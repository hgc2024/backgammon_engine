import torch
import torch.optim as optim
import numpy as np
import os
import time
import random
from collections import deque
import multiprocessing as mp
import csv
import sys
import shutil
import datetime
from tqdm import tqdm
from src.game import BackgammonGame, GamePhase
from src.model_gen5 import BackgammonValueNetGen5
from src.model import BackgammonValueNet # For loading Gen 4 Teacher

# Logging Setup
class DualLogger:
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, "a", encoding='utf-8')
        
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
        
    def flush(self):
        self.terminal.flush()
        self.log.flush()

def setup_logging():
    if not os.path.exists("logs"):
        os.makedirs("logs")
    sys.stdout = DualLogger("logs/training_gen5.log")

def archive_current_state():
    """Archive current critical checkpoints to avoid overwrite."""
    if not os.path.exists("checkpoints"):
        return
        
    targets = ["latest_gen5.pth", "best_so_far_gen5.pth"]
    found = [t for t in targets if os.path.exists(os.path.join("checkpoints", t))]
    
    if found:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_dir = os.path.join("checkpoints", f"archive_gen5_{timestamp}")
        os.makedirs(archive_dir, exist_ok=True)
        print(f">>> Archiving current checkpoints to {archive_dir}...")
        
        for t in found:
            src_path = os.path.join("checkpoints", t)
            dst_path = os.path.join(archive_dir, t)
            shutil.copy2(src_path, dst_path)
            print(f"    Archived: {t}")

def log_metrics(episode, loss, epsilon, win_rate_random=None, win_rate_champion=None):
    file_exists = os.path.isfile("logs/metrics_gen5.csv")
    with open("logs/metrics_gen5.csv", "a", newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Episode", "Loss", "Epsilon", "WinRandom", "WinChampion"])
        writer.writerow([episode, f"{loss:.4f}", f"{epsilon:.4f}", win_rate_random if win_rate_random else "", win_rate_champion if win_rate_champion else ""])

def get_obs_from_state(board, bar, off, perspective_player, score, cube_val, turn, match_target=5):
    """
    Gen 5 Observation:
    192 Board + 2 Bar + 2 Off + 1 Turn + 1 Cube = 198 (Gen 4)
    + 2 MATCH INFO (MyScore/Tgt, OppScore/Tgt) = 200 Total
    """
    # Egocentric observation for `perspective_player`
    if perspective_player == 0:
        my_board = np.maximum(board, 0)
        opp_board = np.abs(np.minimum(board, 0))
        my_score = score[0]
        opp_score = score[1]
    else:
        # P1 (Black/Negative) -> Flip board (0->23 becomes 23->0)
        my_board = np.abs(np.minimum(board, 0))[::-1]
        opp_board = np.maximum(board, 0)[::-1]
        my_score = score[1]
        opp_score = score[0]
        
    features = []
    def encode_point(count):
        f = [0]*4
        if count >= 1: f[0] = 1
        if count >= 2: f[1] = 1
        if count >= 3: f[2] = 1
        if count > 3: f[3] = (count - 3) / 2.0
        return f
        
    for c in my_board: features.extend(encode_point(c))
    for c in opp_board: features.extend(encode_point(c))
    
    # Bar/Off
    my_bar = bar[perspective_player]
    opp_bar = bar[1-perspective_player]
    features.extend([my_bar/2.0, opp_bar/2.0])
    
    my_off = off[perspective_player]
    opp_off = off[1-perspective_player]
    features.extend([my_off/15.0, opp_off/15.0])
    
    # Turn: Always 1.0 (It's my turn in this state)
    features.append(1.0) 
    features.append(cube_val / 64.0)
    
    # --- GEN 5 NEW FEATURES ---
    # Match Score Normalized
    # 0.0 -> Start, 1.0 -> Won Match
    features.append(min(my_score / match_target, 1.0))
    features.append(min(opp_score / match_target, 1.0))
    
    return np.array(features, dtype=np.float32)

def evaluate_vs_random(net, device, n_games=50):
    net.eval()
    wins = 0
    game = BackgammonGame()
    
    for _ in range(n_games):
        game.reset_match()
        game_over = False
        
        while not game_over:
            if game.phase == GamePhase.DECIDE_CUBE_OR_ROLL:
                game.step(0)
            elif game.phase == GamePhase.RESPOND_TO_DOUBLE:
                game.step(0)
            elif game.phase == GamePhase.DECIDE_MOVE:
                moves = game.legal_moves
                if not moves:
                    game.turn = 1 - game.turn
                    game.phase = GamePhase.DECIDE_CUBE_OR_ROLL
                    continue
                
                if game.turn == 0:
                    # Gen 5 Agent
                    best_idx = 0
                    boards = []
                    for seq in moves:
                        b, ba, o = game.get_afterstate(seq)
                        obs = get_obs_from_state(b, ba, o, 0, game.score, game.cube_value, 0)
                        boards.append(obs)
                    
                    if boards:
                        t = torch.tensor(np.array(boards), dtype=torch.float32).to(device)
                        with torch.no_grad():
                            logits, _ = net(t)
                            probs = torch.softmax(logits, dim=1)
                            weights = torch.tensor([-3.0, -2.0, -1.0, 1.0, 2.0, 3.0], device=device)
                            vals = torch.sum(probs * weights, dim=1)
                        best_idx = torch.argmin(vals).item() # Opponent (me) wants to Minimize Opponent's equity? No.
                        # Wait. If I evaluate state, it's MY turn. I want MAX equity.
                        # But get_afterstate returns state where it is OPPONENT turn.
                        # So I want state where Opponent equity is MIN. Correct.
                        best_idx = torch.argmin(vals).item()
                    
                    pts, winner, done = game.step(best_idx)
                    if done and winner == 0: wins += 1
                else:
                    # Random
                    idx = random.randint(0, len(moves)-1)
                    pts, winner, done = game.step(idx)
                    if done and winner == 0: wins += 1
            elif game.phase == GamePhase.GAME_OVER:
                game_over = True
                
    return wins / n_games

def evaluate_vs_champion(candidate_net, champion_net, device, n_games=50):
    candidate_net.eval()
    champion_net.eval()
    wins = 0
    game = BackgammonGame()
    
    # Challenge: Candidate (Model A) vs Champion (Model B)
    # We play n_games. Half as P0, Half as P1 to be fair.
    
    for i in range(n_games):
        game.reset_match()
        
        # Swap sides halfway
        if i < n_games // 2:
            candidate_player = 0
        else:
            candidate_player = 1
            
        game_over = False
        
        while not game_over:
            if game.phase == GamePhase.DECIDE_CUBE_OR_ROLL:
                game.step(0)
            elif game.phase == GamePhase.RESPOND_TO_DOUBLE:
                game.step(0)
            elif game.phase == GamePhase.DECIDE_MOVE:
                moves = game.legal_moves
                if not moves:
                    game.turn = 1 - game.turn
                    game.phase = GamePhase.DECIDE_CUBE_OR_ROLL
                    continue
                
                # Identify Active Net
                current_player = game.turn
                if current_player == candidate_player:
                    active_net = candidate_net
                    style_weights = [-3.0, -2.0, -1.0, 1.0, 2.0, 3.0] # Aggressive
                else:
                    active_net = champion_net
                    style_weights = [-3.0, -2.0, -1.0, 1.0, 2.0, 3.0] # Same style
                
                # Make Move
                best_idx = 0
                boards = []
                for seq in moves:
                    b, ba, o = game.get_afterstate(seq)
                    # Observation from perspective of OPPONENT (who is about to move)
                    # We want to MINIMIZE their equity.
                    opp = 1 - current_player
                    obs = get_obs_from_state(b, ba, o, opp, game.score, game.cube_value, opp)
                    boards.append(obs)
                
                if boards:
                    t = torch.tensor(np.array(boards), dtype=torch.float32).to(device)
                    with torch.no_grad():
                        logits, _ = active_net(t)
                        probs = torch.softmax(logits, dim=1)
                        weights = torch.tensor(style_weights, device=device)
                        vals = torch.sum(probs * weights, dim=1)
                    best_idx = torch.argmin(vals).item()
                
                pts, winner, done = game.step(best_idx)
                if done:
                    if winner == candidate_player: wins += 1
            
            elif game.phase == GamePhase.GAME_OVER:
                game_over = True
                
    return wins / n_games

def main():
    setup_logging()
    archive_current_state()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} (Gen 5)")
    
    net = BackgammonValueNetGen5().to(device)
    
    # Load Teacher (Gen 4) for Knowledge Distillation
    teacher_net = BackgammonValueNet().to(device) # Standard Gen 4 Net
    teacher_net.eval()
    teacher_loaded = False
    
    # Path requested by user
    teacher_path = "best_so_far_gen4.pth" 
    if not os.path.exists(teacher_path):
         teacher_path = "checkpoints/best_so_far.pth" # Fallback
    
    if os.path.exists(teacher_path):
        try:
            print(f">>> Loading Teacher from {teacher_path}...")
            ckpt = torch.load(teacher_path, map_location=device)
            if 'model_state_dict' in ckpt:
                teacher_net.load_state_dict(ckpt['model_state_dict'])
            else:
                teacher_net.load_state_dict(ckpt)
            print(">>> Teacher Gen 4 Loaded successfully.")
            teacher_loaded = True
        except Exception as e:
            print(f"!!! Failed to load teacher: {e}")
    else:
        print(">>> No Teacher found. Training Gen 5 from scratch.")

    optimizer = optim.Adam(net.parameters(), lr=1e-4, weight_decay=1e-5) # Slightly lower LR for Transformer
    
    # Loss Functions
    ce_loss_fn = torch.nn.CrossEntropyLoss()
    mse_loss_fn = torch.nn.MSELoss()
    
    distill_loss_fn = torch.nn.MSELoss() # For matching logits
    
    start_episode = 1
    episodes = 500_000
    best_win_rate = 0.0
    best_win_rate_random = 0.0 # New Tracking
    
    if not os.path.exists("checkpoints"):
        os.makedirs("checkpoints")
      
    # Check for existing checkpoint to RESUME
    resume_candidates = ["checkpoints/latest_gen5.pth", "checkpoints/best_so_far_gen5.pth"]
    resume_path = None
    for p in resume_candidates:
        if os.path.exists(p):
            resume_path = p
            break
            
    # Initialize Scheduler EARLY (Fix for Bug)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50_000, gamma=0.5)

    if resume_path:
        print(f">>> Resuming from {resume_path}...")
        try:
            checkpoint = torch.load(resume_path, map_location=device)
            net.load_state_dict(checkpoint['model_state_dict'])
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            if 'episode' in checkpoint:
                start_episode = checkpoint['episode'] + 1
            if 'best_win_rate' in checkpoint:
                best_win_rate = checkpoint['best_win_rate']
            if 'best_win_rate_random' in checkpoint:
                best_win_rate_random = checkpoint['best_win_rate_random']
            print(f">>> Resumed at Episode {start_episode}")
        except Exception as e:
            print(f"!!! Failed to resume: {e}")
            
    recent_losses = deque(maxlen=100)
    
    game = BackgammonGame()
    
    try:
        for episode in range(start_episode, episodes + 1):
            decay_period = 50_000
            if episode < decay_period:
                epsilon = 1.0 - (0.9 * (episode / decay_period))
            else:
                epsilon = 0.1

            # Match Setup (Gen 5 trains on Matches)
            game.reset_match() # Always new match or continue match? 
            # For simplicity, standard TD trains on games but we input score.
            # We should randomize scores occasionally to teach it "Desperation".
            if random.random() < 0.2:
                 # Randomize Score
                 game.score = [random.randint(0, 4), random.randint(0, 4)]
            
            trajectory = [] 
            game_over = False
            final_win_type = 0
            final_winner = -1
            
            while not game_over:
                current_player = game.turn
                
                if game.phase == GamePhase.DECIDE_CUBE_OR_ROLL:
                    game.step(0)
                elif game.phase == GamePhase.RESPOND_TO_DOUBLE:
                    game.step(0)
                elif game.phase == GamePhase.DECIDE_MOVE:
                    moves = game.legal_moves
                    if not moves:
                        game.turn = 1 - game.turn 
                        game.phase = GamePhase.DECIDE_CUBE_OR_ROLL
                        continue
                    
                    use_random = False
                    if np.random.rand() < epsilon: use_random = True
                    
                    if use_random:
                        best_idx = random.randint(0, len(moves) - 1)
                    else:
                        boards = []
                        for seq in moves:
                            b, ba, o = game.get_afterstate(seq)
                            opponent = 1 - current_player
                            # Gen 5 Obs
                            nxt_obs = get_obs_from_state(b, ba, o, opponent, game.score, game.cube_value, opponent)
                            boards.append(nxt_obs)
                            
                        if boards:
                            t_cand = torch.tensor(np.array(boards), dtype=torch.float32).to(device)
                            with torch.no_grad():
                                logits, _ = net(t_cand)
                                probs = torch.softmax(logits, dim=1)
                                weights = torch.tensor([-3.0, -2.0, -1.0, 1.0, 2.0, 3.0], device=device)
                                vals = torch.sum(probs * weights, dim=1)
                            best_idx = torch.argmin(vals).item()
                        else:
                            best_idx = 0
                            
                    # Store Trajectory
                    obs_curr = get_obs_from_state(game.board, game.bar, game.off, current_player, game.score, game.cube_value, current_player)
                    p0_pip, p1_pip = game.get_pip_counts()
                    if current_player == 0:
                        pip_target = [p0_pip / 100.0, p1_pip / 100.0]
                    else:
                        pip_target = [p1_pip / 100.0, p0_pip / 100.0]
                        
                    trajectory.append((obs_curr, current_player, pip_target))

                    pts, winner, done = game.step(best_idx)
                    if done:
                        final_winner = winner
                        final_win_type = game.get_win_type(winner)
                    
                elif game.phase == GamePhase.GAME_OVER:
                    game_over = True
            
            # Training Step
            states = []
            outcome_targets = []
            pip_targets = []
            
            for (obs, player_at_step, pip_tgt) in trajectory:
                 states.append(obs)
                 pip_targets.append(pip_tgt)
                 if player_at_step == final_winner:
                     tgt_class = 2 + final_win_type
                 else:
                     tgt_class = 3 - final_win_type
                 outcome_targets.append(tgt_class)
                 
            if states:
                 states_t = torch.tensor(np.array(states), dtype=torch.float32).to(device)
                 outcome_t = torch.tensor(np.array(outcome_targets), dtype=torch.long).to(device)
                 pip_t = torch.tensor(np.array(pip_targets), dtype=torch.float32).to(device)
                 
                 logits, pip_preds = net(states_t)
                 loss_outcome = ce_loss_fn(logits, outcome_t)
                 loss_pip = mse_loss_fn(pip_preds, pip_t)
                 loss = loss_outcome + 0.1 * loss_pip
                 
                 # --- KNOWLEDGE DISTILLATION ---
                 if teacher_loaded: 
                     with torch.no_grad():
                         # Teacher expects 198 (No Match Info / Last 2 dims).
                         # Gen 5 State is 200. Slice [:, :-2] to get Gen 4 input.
                         teacher_input = states_t[:, :-2]
                         teacher_logits, _ = teacher_net(teacher_input)
                         
                     # Loss: Minimize MSE between Student Logits and Teacher Logits
                     loss_distill = distill_loss_fn(logits, teacher_logits)
                     
                     # Weight decays as student grows up (Starts strong to clone teacher)
                     # Distill Weight: 1.0 -> 0.1
                     distill_weight = 1.0 * epsilon 
                     loss += distill_weight * loss_distill
                 
                 optimizer.zero_grad()
                 loss.backward()
                 torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                 optimizer.step()
                 recent_losses.append(loss.item())

            scheduler.step()
            
            if episode % 50 == 0:
                avg_loss = np.mean(recent_losses) if recent_losses else 0
                print(f"Ep {episode}: Loss={avg_loss:.4f} | Eps={epsilon:.3f}")
                log_metrics(episode, avg_loss, epsilon)
                
            if episode % 500 == 0:
                 torch.save({
                        'episode': episode,
                        'model_state_dict': net.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'best_win_rate': best_win_rate
                    }, "checkpoints/latest_gen5.pth")

            # Evaluation
            if episode % 250 == 0:
                # 1. Sanity Check vs Random
                win_rate_random = evaluate_vs_random(net, device, n_games=200)
                print(f">>> EVALUATION: Win Rate vs Random: {win_rate_random*100:.1f}%")
                
                trigger_koth = False
                
                # Check for New Personal Best vs Random
                if win_rate_random > best_win_rate_random:
                    best_win_rate_random = win_rate_random
                    print(f"*** NEW RANDOM RECORD! ({best_win_rate_random*100:.1f}%) ***")
                    # Save "Best vs Random" specifically
                    torch.save({
                        'episode': episode,
                        'model_state_dict': net.state_dict(),
                        'best_win_rate_random': best_win_rate_random
                    }, "checkpoints/best_vs_random_gen5.pth")
                    trigger_koth = True # Trigger KOTH ONLY on new record
                    
                # 2. King of the Hill Challenge
                if trigger_koth:
                    champion_path = "checkpoints/best_so_far_gen5.pth"
                    is_new_champion = False
                    
                    if os.path.exists(champion_path):
                        # Load Current King
                        try:
                            champion_net = BackgammonValueNetGen5().to(device)
                            champion_net.load_state_dict(torch.load(champion_path, map_location=device)['model_state_dict'])
                            
                            # Fight!
                            print(">>> CHALLENGER APPEARED! Standard vs King...")
                            win_rate_champ = evaluate_vs_champion(net, champion_net, device, n_games=50)
                            print(f">>> KOTH RESULT: Challenger Win Rate: {win_rate_champ*100:.1f}%")
                            
                            if win_rate_champ > 0.55: # Must beat clearly
                                print("!!! NEW KING CROWNED !!!")
                                is_new_champion = True
                            else:
                                print("... The King defends the throne.")
                        except Exception as e:
                            print(f"!!! Error loading champion for KOTH: {e}")
                            # Fallback: Treat as first champion
                            is_new_champion = True
                    else:
                        # No King yet? You are the King.
                        print("!!! FIRST KING CROWNED !!!")
                        is_new_champion = True
                        
                    if is_new_champion:
                        best_win_rate = win_rate_random # Legacy Verification Metric (or update to Champ Win Rate? Keep random for simplicity of history)
                        torch.save({
                            'episode': episode,
                            'model_state_dict': net.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict(),
                            'best_win_rate': best_win_rate,
                            'best_win_rate_random': best_win_rate_random
                        }, "checkpoints/best_so_far_gen5.pth")
                
                log_metrics(episode, avg_loss if 'avg_loss' in locals() else 0, epsilon, win_rate_random=win_rate_random)

    
    except KeyboardInterrupt:
        print("\n\n!!! KeyboardInterrupt detected.")
        torch.save({
            'episode': episode,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_win_rate': best_win_rate
        }, "checkpoints/latest_gen5.pth")
        print(f"!!! Emergency Save Complete: checkpoints/latest_gen5.pth")

if __name__ == "__main__":
    main()
