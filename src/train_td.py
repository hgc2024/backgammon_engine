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
    optimizer = optim.Adam(net.parameters(), lr=1e-4, weight_decay=1e-5) # Added weight decay
    loss_fn = torch.nn.MSELoss()
    
    # Check for existing
    if os.path.exists("td_backgammon.pth"):
        try:
            net.load_state_dict(torch.load("td_backgammon.pth", map_location=device))
            print("Loaded existing model.")
        except:
            print("Could not load model, starting fresh.")
            
    episodes = 20_000 # Continuous training
    epsilon = 0.1
    
    game = BackgammonGame()
    
    # Metrics
    recent_losses = deque(maxlen=100)
    recent_magnitudes = deque(maxlen=100) # Track avg |V(s)|
    
    for episode in range(1, episodes + 1):
        game.reset_match()
        
        # Trajectory Storage: List of (Observation, PlayerWhoMoved)
        # We need this to assign targets later.
        # But wait: V(S) is P(CurrentPlayer Wins).
        # We perform a move at state S_t -> S_{t+1}.
        # We want to train V(S_t) to predict the eventual Winner.
        trajectory = [] 
        
        game_over = False
        step_count = 0
        
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
                
                # 1. Capture State BEFORE Move (S_t)
                obs = get_obs_from_state(game.board, game.bar, game.off, current_player, game.score, game.cube_value, current_player)
                
                # 2. Select Move
                best_idx = 0
                
                # Evaluate candidates to pick move (Policy)
                if np.random.rand() < epsilon:
                    best_idx = random.randint(0, len(moves) - 1)
                else:
                    boards = []
                    for seq in moves:
                        b, ba, o = game.get_afterstate(seq)
                        # Evaluate Next State (S_{t+1}) from OPPONENT perspective
                        # V(S_{t+1}) = P(Opponent Wins)
                        opponent = 1 - current_player
                        nxt_obs = get_obs_from_state(b, ba, o, opponent, game.score, game.cube_value, opponent)
                        boards.append(nxt_obs)
                        
                    if boards:
                        t_cand = torch.tensor(np.array(boards), dtype=torch.float32).to(device)
                        with torch.no_grad():
                            vals = net(t_cand).squeeze(1)
                        # We want to MINIMIZE P(Opponent Wins)
                        best_idx = torch.argmin(vals).item()
                
                # 3. Store Trajectory for Training
                # We want to train V(S_t) -> Winner.
                trajectory.append((obs, current_player))
                
                # 4. Step
                game.step(best_idx)
                step_count += 1
                
            elif game.phase == GamePhase.GAME_OVER:
                game_over = True

        # --- End of Episode: Monte Carlo Update ---
        # Who won?
        winner_idx = 0 if game.score[0] > game.score[1] else 1
        
        # Prepare Batch
        states = []
        targets = []
        
        for (obs, player_at_step) in trajectory:
            states.append(obs)
            # Target:
            # V(s) is "Prob Current Player Wins".
            # If current player == winner: Target = 1.0
            # If current player != winner: Target = -1.0
            tgt = 1.0 if player_at_step == winner_idx else -1.0
            targets.append(tgt)
            
        if states:
            states_t = torch.tensor(np.array(states), dtype=torch.float32).to(device)
            targets_t = torch.tensor(np.array(targets), dtype=torch.float32).unsqueeze(1).to(device)
            
            # Forward
            preds = net(states_t)
            loss = loss_fn(preds, targets_t)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            recent_losses.append(loss.item())
            recent_magnitudes.append(torch.mean(torch.abs(preds)).item())

        # Logging
        if episode % 50 == 0:
            avg_loss = np.mean(recent_losses) if recent_losses else 0
            avg_conf = np.mean(recent_magnitudes) if recent_magnitudes else 0
            print(f"Ep {episode}: Loss={avg_loss:.4f} | Conf={avg_conf:.4f} | Steps={step_count}")
            
        # Evaluation Arena
        if episode % 250 == 0: # More frequent than 1000 for visibility
            print("Running Evaluation vs Random...")
            win_rate = evaluate_vs_random(net, device, n_games=50)
            print(f">>> EVALUATION: Win Rate vs Random: {win_rate*100:.1f}%")
            
        if episode % 500 == 0:
             torch.save(net.state_dict(), "td_backgammon.pth")

if __name__ == "__main__":
    main()
