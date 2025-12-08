import torch
import torch.optim as optim
import numpy as np
import os
import random
from collections import deque
from src.game import BackgammonGame, GamePhase
from src.model import BackgammonValueNet

def get_obs_from_state(board, bar, off, perspective_player, score, cube_val, turn):
    # Egocentric observation for `perspective_player`
    if perspective_player == 0:
        my_board = np.maximum(board, 0)
        opp_board = np.abs(np.minimum(board, 0))
    else:
        # P1 (Black/Negative) -> Flip board (0->23 becomes 23->0)
        my_board = np.abs(np.minimum(board, 0))[::-1]
        opp_board = np.maximum(board, 0)[::-1]
        
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
    
    return np.array(features, dtype=np.float32)

def evaluate_vs_random(agent_net, device, n_games=50):
    """
    Plays n_games where Agent (using Net) plays P0 and Random plays P1.
    Returns Win Rate for Agent.
    """
    game = BackgammonGame()
    wins = 0
    
    agent_net.eval()
    
    for _ in range(n_games):
        game.reset_match()
        game_over = False
        
        while not game_over:
            if game.phase == GamePhase.DECIDE_CUBE_OR_ROLL:
                game.step(0) # Always Roll
            elif game.phase == GamePhase.RESPOND_TO_DOUBLE:
                game.step(0) # Always Take
            elif game.phase == GamePhase.DECIDE_MOVE:
                moves = game.legal_moves
                if not moves:
                    game.turn = 1 - game.turn
                    game.phase = GamePhase.DECIDE_CUBE_OR_ROLL
                    continue
                
                # Turn logic
                if game.turn == 0:
                    # Agent Turn (Smart)
                    best_val = -float('inf')
                    best_idx = 0
                    
                    # 1-Ply Lookahead
                    # We want to MAXIMIZE "My Win Prob".
                    # get_afterstate returns state after move.
                    # Standard logic: V(afterstate) = P(Opponent Wins)
                    # So we want to MINIMIZE V(afterstate).
                    
                    # Batch eval
                    boards = []
                    for seq in moves:
                        b, ba, o = game.get_afterstate(seq)
                        # Obs for Opponent (P1)
                        obs = get_obs_from_state(b, ba, o, 1, game.score, game.cube_value, 1)
                        boards.append(obs)
                        
                    if not boards:
                        best_idx = 0
                    else:
                        t = torch.tensor(np.array(boards), dtype=torch.float32).to(device)
                        with torch.no_grad():
                            vals = agent_net(t).squeeze(1) # Opponent Win Probs
                        
                        # Argmin Opponent Win Prob = Argmax My Win Prob
                        best_idx = torch.argmin(vals).item()
                        
                    game.step(best_idx)
                else:
                    # Random Turn
                    idx = random.randint(0, len(moves)-1)
                    game.step(idx)
                    
            if game.phase == GamePhase.GAME_OVER:
                # game.score has result.
                # If P0 (Agent) has points > P1?
                # Usually score is updated.
                # Check who got points.
                # game.score is [P0_pts, P1_pts]
                if game.score[0] > game.score[1]:
                    wins += 1
                game_over = True
                
    agent_net.train()
    return wins / n_games

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    net = BackgammonValueNet().to(device)
    # Fine-Tuning: Start with lower LR (5e-5) to avoid breaking existing knowledge
    optimizer = optim.Adam(net.parameters(), lr=5e-5, weight_decay=1e-5) 
    loss_fn = torch.nn.MSELoss()
    
    start_episode = 1
    episodes = 500_000 # Continuous training
    best_win_rate = 0.0
    
    if not os.path.exists("checkpoints"):
        os.makedirs("checkpoints")
    
    # Stability: Scheduler
    # Halve LR every 50,000 episodes
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50_000, gamma=0.5)
    
    # Check for existing
    if os.path.exists("td_backgammon.pth"):
        try:
            checkpoint = torch.load("td_backgammon.pth", map_location=device)
            # Check if it's a full checkpoint or just weights (Legacy)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                net.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                if 'scheduler_state_dict' in checkpoint:
                    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                start_episode = checkpoint.get('episode', 1) + 1
                best_win_rate = checkpoint.get('best_win_rate', 0.0)
                print(f"Loaded Checkpoint: Resuming from Episode {start_episode}")
            else:
                # Legacy: Weights only
                net.load_state_dict(checkpoint)
                # Assume a "Legacy" model is partially trained. 
                # Jump to 100k to lower Epsilon (to ~0.05) and prepare for fine-tuning.
                start_episode = 100_000
                print("Loaded Legacy Model. Assuming mature (Stage 100k).")
                print(f"-> Epsilon will start at ~{(0.1 - (0.09 * (100000/200000))):.3f}")
        except Exception as e:
            print(f"Could not load model: {e}. Starting fresh.")
            
    # Metrics
    recent_losses = deque(maxlen=100)
    recent_magnitudes = deque(maxlen=100) # Track avg |V(s)|
    
    game = BackgammonGame()
    
    # Opponent Net (for Pool Play)
    opponent_net = BackgammonValueNet().to(device)
    opponent_net.eval()
    
    for episode in range(start_episode, episodes + 1):
        # Stability: Epsilon Decay
        decay_progress = min(1.0, episode / 200_000.0)
        epsilon = max(0.01, 0.1 - (0.09 * decay_progress)) # Explicit min clamp
        
        # --- Opponent Selection ---
        # Default: Self-Play (Opponent is Same as Net)
        # 20% Chance: Load a past checkpoint (if available)
        is_self_play = True
        
        # List available checkpoints
        past_models = [f for f in os.listdir("checkpoints") if f.endswith(".pth")]
        
        if past_models and np.random.rand() < 0.20:
             # Play against history
             is_self_play = False
             selected_ckpt = random.choice(past_models)
             try:
                 ckpt_path = os.path.join("checkpoints", selected_ckpt)
                 ckpt_data = torch.load(ckpt_path, map_location=device)
                 if isinstance(ckpt_data, dict) and 'model_state_dict' in ckpt_data:
                     opponent_net.load_state_dict(ckpt_data['model_state_dict'])
                 else:
                     opponent_net.load_state_dict(ckpt_data)
                 # print(f"Vs History: {selected_ckpt}") # Too noisy for every game
             except:
                 # Fallback to self-play if load fails
                 is_self_play = True
        
        game.reset_match()
        
        trajectory = [] 
        game_over = False
        step_count = 0
        
        # Randomize who is Agent and who is Opponent?
        # Standard: Agent is P0, Opponent is P1.
        # But we want Agent to learn both sides.
        # Simple approach: Agent plays BOTH sides if Self-Play.
        # If History-Play: Agent plays P0, History plays P1. 
        # (Or randomize side? Let's keep it simple: Agent=P0, History=P1 for now, or Agent=Turn?)
        
        # Let's say: Agent is ALWAYS the one learning.
        # If Self-Play: Agent makes moves for P0 and P1. We learn from BOTH.
        # If History-Play: Agent plays P0 (Learns). History plays P1 (No Learn).
        
        # Actually, TD-Gammon self-play usually treats the network as "The Player" for whichever turn it is.
        # So "Self Play" means Net is used for P0 and P1.
        # "History Play" means Net is P0, OldNet is P1.
        # We need to randomize so Agent sometimes plays P1 against History P0? 
        # Yes, otherwise it overfits to P0.
        
        agent_side = 0 if is_self_play else random.randint(0, 1) # If History, pick a side.
        
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
                    
                current_player = game.turn
                
                # Logic: Which brain controls this turn?
                # If Self-Play: `net` controls both.
                # If History-Play: 
                #    If current_player == agent_side: `net` controls (and LEARNS).
                #    If current_player != agent_side: `opponent_net` controls (NO LEARN).
                
                is_agent_turn = is_self_play or (current_player == agent_side)
                
                # Select Brain
                active_net = net if is_agent_turn else opponent_net
                
                # 1. Capture State BEFORE Move (S_t)
                # Only need Obs for training if it's Agent's turn
                obs = get_obs_from_state(game.board, game.bar, game.off, current_player, game.score, game.cube_value, current_player)
                
                # 2. Select Move
                best_idx = 0
                
                # Policy: Epsilon Greedy
                # Note: Opponent (History) usually plays greedy (epsilon=0) or low noise?
                # Let's assume Opponent plays best move (Greedy).
                # Agent uses current epsilon.
                
                use_random = False
                if is_agent_turn:
                    if np.random.rand() < epsilon: use_random = True
                else:
                    pass # History plays Greedy
                
                if use_random:
                    best_idx = random.randint(0, len(moves) - 1)
                else:
                    boards = []
                    for seq in moves:
                        b, ba, o = game.get_afterstate(seq)
                        opponent = 1 - current_player
                        nxt_obs = get_obs_from_state(b, ba, o, opponent, game.score, game.cube_value, opponent)
                        boards.append(nxt_obs)
                        
                    if boards:
                        t_cand = torch.tensor(np.array(boards), dtype=torch.float32).to(device)
                        with torch.no_grad():
                            # Important: Use the ACTIVE net (Agent or History)
                            vals = active_net(t_cand).squeeze(1)
                        best_idx = torch.argmin(vals).item()
                
                # 3. Store Trajectory (ONLY if Agent Turn)
                if is_agent_turn:
                    trajectory.append((obs, current_player))
                
                # 4. Step
                game.step(best_idx)
                step_count += 1
                
            elif game.phase == GamePhase.GAME_OVER:
                game_over = True
        
        # Update Weights (Standard Loop) ...
        # (Unchanged Logic below, just processes trajectory)
        winner_idx = 0 if game.score[0] > game.score[1] else 1
        
        states = []
        targets = []
        for (obs, player_at_step) in trajectory:
             states.append(obs)
             tgt = 1.0 if player_at_step == winner_idx else -1.0
             targets.append(tgt)
             
        if states:
             states_t = torch.tensor(np.array(states), dtype=torch.float32).to(device)
             targets_t = torch.tensor(np.array(targets), dtype=torch.float32).unsqueeze(1).to(device)
             preds = net(states_t)
             loss = loss_fn(preds, targets_t)
             optimizer.zero_grad()
             loss.backward()
             optimizer.step()
             recent_losses.append(loss.item())
             recent_magnitudes.append(torch.mean(torch.abs(preds)).item())

        scheduler.step()

        # Logging ... (same)
        if episode % 50 == 0:
            avg_loss = np.mean(recent_losses) if recent_losses else 0
            avg_conf = np.mean(recent_magnitudes) if recent_magnitudes else 0
            current_lr = scheduler.get_last_lr()[0]
            mode_str = "Self" if is_self_play else "Pool"
            print(f"Ep {episode} [{mode_str}]: Loss={avg_loss:.4f} | Eps={epsilon:.3f} | LR={current_lr:.2e}")
            
        # Save Pool Checkpoint every 2,500 (Frequent Diversity)
        if episode % 2_500 == 0:
             ckpt_name = f"checkpoints/td_backgammon_ep{episode}.pth"
             torch.save({
                    'episode': episode,
                    'model_state_dict': net.state_dict(),
                }, ckpt_name)
             print(f"Saved History Checkpoint: {ckpt_name}")

        # Evaluation ... (Unchanged)
        if episode % 250 == 0: # More frequent than 1000 for visibility
            print("Running Evaluation vs Random...")
            win_rate = evaluate_vs_random(net, device, n_games=200) # Increased to reduce noise
            print(f">>> EVALUATION: Win Rate vs Random: {win_rate*100:.1f}%")
            
            # Best Model Checkpoint
            if win_rate > best_win_rate:
                best_win_rate = win_rate
                torch.save({
                    'episode': episode,
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_win_rate': best_win_rate
                }, "td_backgammon_best.pth")
                print(f"*** New Best Model Saved! ({win_rate*100:.1f}%) ***")
            
        if episode % 500 == 0:
             torch.save({
                    'episode': episode,
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_win_rate': best_win_rate
                }, "td_backgammon.pth")

if __name__ == "__main__":
    main()
