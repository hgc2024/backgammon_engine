import torch
import torch.optim as optim
import numpy as np
import os
import time
import random
from collections import deque
import multiprocessing as mp
from tqdm import tqdm
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

def evaluate_vs_random_worker(args):
    """
    Worker: Agent (P0) vs Random (P1)
    args: (agent_state_dict, seed)
    Returns: 1 if Agent wins, 0 otherwise.
    """
    agent_state, seed = args
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    device = torch.device('cpu') 
    
    game = BackgammonGame()
    agent = BackgammonValueNet().to(device)
    agent.load_state_dict(agent_state)
    agent.eval()
    
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
                # Agent Turn
                best_idx = 0
                boards = []
                for seq in moves:
                    b, ba, o = game.get_afterstate(seq)
                    # Agent (P0) looks at P1's future board
                    obs = get_obs_from_state(b, ba, o, 1, game.score, game.cube_value, 1)
                    boards.append(obs)
                    
                if not boards:
                    best_idx = 0
                else:
                    t = torch.tensor(np.array(boards), dtype=torch.float32).to(device)
                    with torch.no_grad():
                        logits, _ = agent(t)
                        probs = torch.softmax(logits, dim=1)
                        weights = torch.tensor([-3.0, -2.0, -1.0, 1.0, 2.0, 3.0], device=device)
                        vals = torch.sum(probs * weights, dim=1)
                    best_idx = torch.argmin(vals).item()
                game.step(best_idx)
            else:
                # Random Turn (P1)
                idx = random.randint(0, len(moves)-1)
                game.step(idx)
                
        if game.phase == GamePhase.GAME_OVER:
            if game.score[0] > game.score[1]:
                return 1
            else:
                return 0
    return 0

def evaluate_vs_random(agent_net, device, n_games=50, num_workers=None):
    if num_workers is None:
        num_workers = max(1, os.cpu_count() - 5)
        
    agent_state = {k: v.cpu() for k, v in agent_net.state_dict().items()}
    
    tasks = []
    base_seed = int(time.time())
    for i in range(n_games):
        tasks.append((agent_state, base_seed + i))
        
    wins = 0
    # On Windows, need to be careful with spawn/fork.
    # We assume 'spawn' is default or handled.
    with mp.Pool(processes=num_workers) as pool:
        results = list(tqdm(pool.imap_unordered(evaluate_vs_random_worker, tasks), total=n_games, desc="Eval vs Random (CPU)", leave=False))
        wins = sum(results)
        
    return wins / n_games

def evaluate_agent_vs_agent(agent_net, opponent_net, device, n_games=50):
    """
    Plays n_games where Agent plays P0 and Opponent plays P1.
    Both use 1-ply greedy search.
    Returns Win Rate for Agent (P0).
    """
    game = BackgammonGame()
    wins = 0
    
    agent_net.eval()
    opponent_net.eval()
    
    for _ in tqdm(range(n_games), desc="Agent vs Agent", leave=False):
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
                    game.turn = 1 - game.turn # Auto-pass
                    game.phase = GamePhase.DECIDE_CUBE_OR_ROLL
                    continue
                
                # Determine Active Net
                current_net = agent_net if game.turn == 0 else opponent_net
                # Perspective of the "Next Player" (Opponent of current turn)
                next_perspective = 1 - game.turn 
                
                # 1-Ply Search
                best_idx = 0
                
                # Batch eval
                boards = []
                for seq in moves:
                    b, ba, o = game.get_afterstate(seq)
                    # We evaluate S' from Next Player's perspective
                    obs = get_obs_from_state(b, ba, o, next_perspective, game.score, game.cube_value, 1)
                    boards.append(obs)
                    
                if not boards:
                    best_idx = 0
                else:
                    t = torch.tensor(np.array(boards), dtype=torch.float32).to(device)
                    with torch.no_grad():
                        logits, _ = current_net(t)
                        probs = torch.softmax(logits, dim=1)
                        weights = torch.tensor([-3.0, -2.0, -1.0, 1.0, 2.0, 3.0], device=device)
                        vals = torch.sum(probs * weights, dim=1)
                    
                    # Current Player wants to MINIMIZE Next Player's Advantage
                    # V(s) = Next Player Advantage (-1 to 1)
                    # So Argmin(V) is best for Current Player.
                    best_idx = torch.argmin(vals).item()
                    
                game.step(best_idx)
                    
            if game.phase == GamePhase.GAME_OVER:
                if game.score[0] > game.score[1]:
                    wins += 1
                game_over = True
                
    agent_net.train()
    return wins / n_games

def play_game_worker(args):
    """
    Worker function for parallel evaluation.
    args: (agent_state_dict, champion_state_dict, seed)
    Returns: 1 if Agent (P0) wins, 0 otherwise.
    """
    agent_state, cham_state, seed = args
    
    # Set seed for reproducibility in this process
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # CPU operation for workers
    device = torch.device('cpu') 
    
    # Init Game and Nets
    game = BackgammonGame()
    agent = BackgammonValueNet().to(device)
    cham = BackgammonValueNet().to(device)
    
    agent.load_state_dict(agent_state)
    cham.load_state_dict(cham_state)
    agent.eval()
    cham.eval()
    
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
            
            # Identify current player
            current_net = agent if game.turn == 0 else cham
            next_perspective = 1 - game.turn
            
            boards = []
            for seq in moves:
                b, ba, o = game.get_afterstate(seq)
                obs = get_obs_from_state(b, ba, o, next_perspective, game.score, game.cube_value, 1)
                boards.append(obs)
            
            if not boards:
                best_idx = 0
            else:
                t = torch.tensor(np.array(boards), dtype=torch.float32).to(device)
                with torch.no_grad():
                    logits, _ = current_net(t)
                    probs = torch.softmax(logits, dim=1)
                    weights = torch.tensor([-3.0, -2.0, -1.0, 1.0, 2.0, 3.0], device=device)
                    vals = torch.sum(probs * weights, dim=1)
                best_idx = torch.argmin(vals).item()
                
            game.step(best_idx)
            
        if game.phase == GamePhase.GAME_OVER:
            if game.score[0] > game.score[1]:
                return 1 # Agent Wins
            else:
                return 0 # Agent Loses
    return 0

def run_challenge(net, champion_net, episode, device, n_games=100, num_workers=None):
    if num_workers is None:
        num_workers = max(1, os.cpu_count() - 5) 
        
    print(f"Running King of the Hill Challenge (Challenger vs Champion) on {num_workers} CPUs...")
    
    agent_state = {k: v.cpu() for k, v in net.state_dict().items()}
    cham_state = {k: v.cpu() for k, v in champion_net.state_dict().items()}
    
    tasks = []
    for i in range(n_games):
        tasks.append((agent_state, cham_state, episode * 10000 + i))
        
    wins = 0
    with mp.Pool(processes=num_workers) as pool:
        results = list(tqdm(pool.imap_unordered(play_game_worker, tasks), total=n_games, desc="Parallel Arena", leave=False))
        wins = sum(results)
        
    challenger_win_rate = wins / n_games
    print(f">>> CHALLENGE: Win Rate vs Champion: {challenger_win_rate*100:.1f}% ({wins}/{n_games})")
    
    if challenger_win_rate > 0.55:
        print(f"*** KING OF THE HILL: PROMOTION! (Win Rate {challenger_win_rate*100:.1f}%) ***")
        torch.save({
            'episode': episode,
            'model_state_dict': net.state_dict(),
        }, "checkpoints/best_so_far.pth")
        
        champion_net.load_state_dict(net.state_dict())
        print(">>> New Champion Crowned.")
        return True
    else:
        print(f">>> Challenge Failed. Champion remains undefeated.")
        return False

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    net = BackgammonValueNet().to(device)
    # FREST START: Higher LR for initial learning
    optimizer = optim.Adam(net.parameters(), lr=2e-4, weight_decay=1e-5) 
    
    # Loss Functions
    ce_loss_fn = torch.nn.CrossEntropyLoss()
    mse_loss_fn = torch.nn.MSELoss()
    
    start_episode = 1
    episodes = 500_000
    best_win_rate = 0.0
    
    if not os.path.exists("checkpoints"):
        os.makedirs("checkpoints")
    
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50_000, gamma=0.5)
    
    recent_losses = deque(maxlen=100)
    
    # Champion Net
    champion_net = BackgammonValueNet().to(device)
    champion_net.eval()
    champion_net.load_state_dict(net.state_dict())

    # Main Loop
    game = BackgammonGame()
    past_models = []
    
    opponent_net = BackgammonValueNet().to(device)
    opponent_net.eval()
    
    for episode in range(start_episode, episodes + 1):
        decay_period = 50_000
        if episode < decay_period:
            epsilon = 1.0 - (0.9 * (episode / decay_period))
        else:
            epsilon = 0.1

        is_self_play = True
        if episode % 1000 == 0:
            past_models = [f for f in os.listdir("checkpoints") if f.endswith(".pth") and "gen2" not in f]
        
        if past_models and np.random.rand() < 0.25:
             is_self_play = False
             selected_ckpt = random.choice(past_models)
             try:
                 ckpt_path = os.path.join("checkpoints", selected_ckpt)
                 ckpt_data = torch.load(ckpt_path, map_location=device)
                 if isinstance(ckpt_data, dict) and 'model_state_dict' in ckpt_data:
                     opponent_net.load_state_dict(ckpt_data['model_state_dict'])
                 else:
                     opponent_net.load_state_dict(ckpt_data)
             except:
                 is_self_play = True
        
        if np.random.rand() < 0.30:
             game.reset_special_endgame()
        else:
             game.reset_match()
        
        trajectory = [] 
        game_over = False
        step_count = 0
        final_win_type = 0
        final_winner = -1
        
        agent_side = 0 if is_self_play else random.randint(0, 1)
        
        while not game_over:
            current_player = game.turn
            
            is_agent_turn = True
            active_net = net
            
            if not is_self_play:
                if current_player != agent_side:
                    is_agent_turn = False
                    active_net = opponent_net
            
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
                if is_agent_turn:
                    if np.random.rand() < epsilon: use_random = True
                else:
                    pass
                
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
                            logits, _ = active_net(t_cand)
                            probs = torch.softmax(logits, dim=1)
                            weights = torch.tensor([-3.0, -2.0, -1.0, 1.0, 2.0, 3.0], device=device)
                            vals = torch.sum(probs * weights, dim=1)
                        best_idx = torch.argmin(vals).item()
                    else:
                        best_idx = 0
                        
                # Store Trajectory (ONLY if Agent Turn)
                if is_agent_turn:
                     obs_curr = get_obs_from_state(game.board, game.bar, game.off, current_player, game.score, game.cube_value, current_player)
                     
                     # Get Pip Counts for Auxiliary Task
                     p0_pip, p1_pip = game.get_pip_counts()
                     # Normalize (divide by 100 for stability)
                     # Perspective: My Pip, Opp Pip
                     if current_player == 0:
                         pip_target = [p0_pip / 100.0, p1_pip / 100.0]
                     else:
                         pip_target = [p1_pip / 100.0, p0_pip / 100.0]
                         
                     trajectory.append((obs_curr, current_player, pip_target))

                pts, winner, done = game.step(best_idx)
                if done:
                    final_winner = winner
                    final_win_type = game.get_win_type(winner) # 1, 2, 3
                step_count += 1
                
            elif game.phase == GamePhase.GAME_OVER:
                game_over = True
        
        # Training Step
        states = []
        outcome_targets = []
        pip_targets = []
        
        for (obs, player_at_step, pip_tgt) in trajectory:
             states.append(obs)
             pip_targets.append(pip_tgt)
             
             # Calculate Target Class (0-5)
             # Classes: 0:LoseBG, 1:LoseG, 2:Lose, 3:Win, 4:WinG, 5:WinBG
             
             if player_at_step == final_winner:
                 # I Won. Map 1->3, 2->4, 3->5.
                 tgt_class = 2 + final_win_type
             else:
                 # I Lost. Map 1->2, 2->1, 3->0.
                 tgt_class = 3 - final_win_type
                 
             outcome_targets.append(tgt_class)
             
        if states:
             states_t = torch.tensor(np.array(states), dtype=torch.float32).to(device)
             outcome_t = torch.tensor(np.array(outcome_targets), dtype=torch.long).to(device)
             pip_t = torch.tensor(np.array(pip_targets), dtype=torch.float32).to(device)
             
             # Forward Pass
             logits, pip_preds = net(states_t)
             
             # Combined Loss
             loss_outcome = ce_loss_fn(logits, outcome_t)
             loss_pip = mse_loss_fn(pip_preds, pip_t)
             
             loss = loss_outcome + 0.1 * loss_pip
             
             optimizer.zero_grad()
             loss.backward()
             torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
             optimizer.step()
             recent_losses.append(loss.item())

        scheduler.step()
        
        if episode % 50 == 0:
            avg_loss = np.mean(recent_losses) if recent_losses else 0
            current_lr = scheduler.get_last_lr()[0]
            print(f"Ep {episode}: Loss={avg_loss:.4f} | Eps={epsilon:.3f}")
            
        if episode % 2_500 == 0:
             ckpt_name = f"checkpoints/td_gen4_ep{episode}.pth"
             torch.save({
                     'episode': episode,
                     'model_state_dict': net.state_dict(),
                 }, ckpt_name)
             print(f"Saved Checkpoint: {ckpt_name}")

        if episode % 500 == 0:
             torch.save({
                    'episode': episode,
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_win_rate': best_win_rate
                }, "checkpoints/best_so_far.pth")

if __name__ == "__main__":
    main()
