import torch
import torch.optim as optim
import numpy as np
import os
from src.game import BackgammonGame, GamePhase
from src.model import BackgammonValueNet
from src.env import BackgammonEnv # Use Env wrapper for obs generation helper?
# Or verify obs generation logic here to avoid overhead of Env wrapper (which is for Gym).
# Ideally share logic. Let's use the helper method from Env if possible or refactor it.
# For now, we will duplicate the _get_obs logic or import it.
# Let's import the CLASS but instantiated lightly.

def get_obs_from_state(board, bar, off, perspective_player, score, cube_val, turn):
    # Re-implement Env logic for static state
    # Egocentric for `perspective_player`
    
    if perspective_player == 0:
        my_board = np.maximum(board, 0)
        opp_board = np.abs(np.minimum(board, 0))
    else:
        # P1 (Black/Negative)
        # Flip board so my home (0-5) is indices 0-5
        # Board: 0 is P1 Home? No, 0 is P0 Home. 23 is P1 Home.
        # Player 1 checkers are negative.
        # Board indices 23->0.
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
    
    # Turn: 1.0 if it's my turn?
    # In 'Afterstate', it's opponent's turn. 
    # But for 'Current State', it is my turn.
    # The network expects [Feature... Turn].
    # Standard: 1 if X is to move, 0 if O is to move?
    # Or "Current Player To Move" is always what we see?
    # Let's stick to "1.0 = It is MY turn".
    # Since we always feed Perspective = Current Player, Turn is always 1.0?
    # Unnecessary feature if always 1? But kept for compatibility.
    features.append(1.0 if turn == perspective_player else 0.0)
    
    features.append(cube_val / 64.0)
    
    return np.array(features, dtype=np.float32)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    net = BackgammonValueNet().to(device)
    optimizer = optim.Adam(net.parameters(), lr=1e-4)
    loss_fn = torch.nn.MSELoss()
    
    # Load checkpoint?
    
    # Training Parameters
    episodes = 100_000
    gamma = 1.0 # Undiscounted for episodic game? Or 0.99? TD-Gammon used lambda=0.7
    epsilon = 0.1 # Exploration
    
    # We need a "Game" instance
    game = BackgammonGame()
    
    for episode in range(episodes):
        game.reset_match()
        
        # Start game loop
        game_over = False
        episode_loss_sum = 0.0
        step_count = 0
        
        # Need to track "Previous State Value" for TD update
        # Standard TD(0): V(s) <- V(s) + alpha * [r + gamma*V(s') - V(s)]
        # We update at every move.
        
        # We need gradients? 
        # In Pytorch, we predict V(s), keep graph.
        # Then observe r, s'. Predict V(s') (no graph).
        # Loss = MSE(V(s), Target). Backward().
        
        last_val = None
        current_player = game.turn # Who made the last move? No, current turn.
        
        # Logic:
        # 1. Roll Dice
        # 2. Get Legal Moves
        # 3. For each move, simulate "Afterstate" (Resulting Board)
        # 4. Predict V(Afterstate) for All.
        # 5. Choose Best.
        # 6. TD Update: The 'Target' for the PREVIOUS move (by same player) is this Best Value.
        
        # Wait, in self-play:
        # White moves -> State S1. Value V(S1).
        # Red moves -> State S2. Value V(S2).
        # White's V(S1) should predict Red's best outcome? 
        # Standard: Reward is from perspective of Current Player.
        
        while not game_over:
            # 1. Roll or Double?
            # Simplified: Auto-roll for now (Skip Cube for first pass of TD)
            # engine handles phases.
            # checks:
            if game.phase == GamePhase.DECIDE_CUBE_OR_ROLL:
                # Always roll
                r, _, done = game.step(0) # 0=Roll
                if done: break
                
            if game.phase == GamePhase.RESPOND_TO_DOUBLE:
                # Always take
                r, _, done = game.step(0)
                if done: break
                
            if game.phase == GamePhase.DECIDE_MOVE:
                 # Generate Afterstates
                 moves = game.legal_moves
                 
                 if not moves:
                     # Must pass
                     game.turn = 1 - game.turn
                     game.phase = GamePhase.DECIDE_CUBE_OR_ROLL
                     continue
                     
                 # Evaluation
                 best_move_idx = 0
                 best_val = -float('inf')
                 
                 # Prepare batch of afterstates
                 boards = []
                 indices = []
                 
                 # Optimization: Batch process
                 for i, seq in enumerate(moves):
                     # Simulate "Afterstate"
                     b, ba, o = game.get_afterstate(seq)
                     
                     # Construct Observation Feature for this state
                     # V(S') is Value of State S' from CURRENT PLAYER perspective
                     # But Wait: After I move, it's Opponent's turn. 
                     # If I use standard encoding, the board flips.
                     # TD-Gammon V(S) = Prob(White Wins).
                     # So if I am Red, I want to MIMIMIZE V(S').
                     # OR: We always predict P(Current Player Wins).
                     # If I am White, I want state where P(White Wins) is High.
                     # If I am Red, I want state where P(Red Wins) is High.
                     
                     # Let's standardize: Net output = P(White Wins) (Player 0) (Range -1 to 1)
                     # White tries to Maximize. Red tries to Minimize.
                     
                     # Feature Extraction must be absolute (not egocentric) OR we handle perspective.
                     # Let's use EGOCENTRIC features + EGOCENTRIC Value?
                     # V(s) = Prob(Current Turn Player Wins).
                     # If I move to S', it becomes Opponent's turn.
                     # So V(S') = Prob(Opponent Wins).
                     # I want to MINIMIZE V(S') (Opponent's win prob).
                     # OR simpler:
                     # My Value of Move = 1 - V(S'_from_opp_perspective)
                     # Or Reward + gamma * -V(S') ?
                     
                     # TD-Gammon: Network always views board from "Current Player" perspective.
                     # Output is "Win Probability for Current Player".
                     # If White moves to S', White wants S' which is BAD for Red.
                     # S' is viewed by Red. V(S') = P(Red Wins).
                     # White chooses Move maximizing (1 - V(S')).
                     
                     # Let's create obs for the NEXT player (opponent)
                     obs = get_obs_from_state(b, ba, o, 1-current_player, game.score, game.cube_value, game.turn)
                     boards.append(obs)
                     indices.append(i)
                 
                 # Tensor-ify
                 if len(boards) > 0:
                     batch_tensor = torch.tensor(np.array(boards), dtype=torch.float32).to(device)
                     
                     with torch.no_grad():
                         values = net(batch_tensor).squeeze(1) # [N]
                         
                     # We want to choose move that leads to MINIMUM opponent win probability
                     # Because V(S') is P(Opponent Wins)
                     
                     best_val_idx = torch.argmin(values).item()
                     best_move_idx = indices[best_val_idx]
                     
                     # The Value of this state S for ME is roughly -V(S') (inverted)
                     current_state_value = -values[best_val_idx].item()
                     
                     # Exploration
                     if np.random.rand() < epsilon:
                         best_move_idx = np.random.choice(indices)
                         # We still use the greedy value for TD target usually? Or SARSA?
                         # Q-Learning: Use max (greedy). SARSA: Use actual.
                         # Let's use actual for simplicity (on-policyish)
                         idx_in_batch = indices.index(best_move_idx)
                         current_state_value = -values[idx_in_batch].item()

                 # Execute Move
                 # apply step
                 r, winner, done = game.step(best_move_idx)
                 
                 # TD Update
                 # Target = Reward + Gamma * Value(Next State)
                 # Value(Next State) is what we just calculated: `current_state_value` (My winning prob)
                 # Wait, Backprop happens on previous step's prediction?
                 
                 # Let's store logic:
                 # State S (My Turn). I predict V(S).
                 # I move -> S' (Opp Turn).
                 # Net predicts V(S') (Opp Win Prob).
                 # My V(S) should approach -V(S') (or 1-V depending on encoding).
                 # If I use Tanh (-1 to +1 where +1 = I win), then V(S) ~= -V(S').
                 
                 # We need to save gradients from V(S) prediction?
                 # No, semi-gradient TD. 
                 # 1. State S. Pred = Net(S).
                 # 2. Determine Best Move -> S'.
                 # 3. Target = -Net(S').detach()
                 # 4. Loss = MSE(Pred, Target).
                 # 5. Backward.
                 
                 # Problem: We need 'S' (State before move).
                 # But we are inside the loop.
                 # Let's track `last_features` and `last_pred`.
                 pass # handled by loops
                 
                 # Actually, simpler:
                 # Just collect Trajectory [S0, S1, S2...]
                 # End of game, Reward is known (+1 / -1).
                 # Train on trajectory? TD(lambda)
                 # Or 1-step TD?
                 
                 # 1-Step TD Implementation here:
                 # We need `obs` BEFORE the move.
                 # We didn't calculate it yet.
                 
                 # Re-structure:
                 # At Start of Turn (DECIDE_MOVE):
                 # 1. Calculate Current Obs.
                 # 2. Predict V(Current Obs) -> `val_pred`.
                 # 3. Find Best Move -> Next State S'.
                 # 4. Predict V(Next State S') -> `next_val`.
                 # 5. Loss = MSE(val_pred, -next_val).
                 # 6. Step.
                 
                 # We need to construct Current Obs
                 curr_obs = get_obs_from_state(game.board, game.bar, game.off, current_player, game.score, game.cube_value, game.turn)
                 curr_obs_t = torch.tensor(curr_obs[None, :], dtype=torch.float32).to(device)
                 val_pred = net(curr_obs_t)
                 
                 optimizer.zero_grad()
                 # Target: -V(S')
                 # If Done, Target = Final Reward.
                 
                 if done:
                     # Winner is Me?
                     # game.step returns reward for ME. 
                     # If I won, reward > 0.
                     # Tanh target: 1.0 (if reward > 0) else -1.0
                     final_target = 1.0 if r > 0 else -1.0
                     target_t = torch.tensor([[final_target]], device=device)
                     loss = loss_fn(val_pred, target_t)
                     loss.backward()
                     optimizer.step()
                     
                     episode_loss_sum += loss.item()
                     step_count += 1
                     break
                 else:
                     # Not done
                     # Target is Derived from Best Child (Evaluating S')
                     target_val = current_state_value # This is -V(S')
                     target_t = torch.tensor([[target_val]], device=device)
                     
                     loss = loss_fn(val_pred, target_t)
                     loss.backward()
                     optimizer.step()
                     
                     episode_loss_sum += loss.item()
                     step_count += 1
        
        if episode % 100 == 0:
            avg_loss = episode_loss_sum / max(1, step_count)
            print(f"Episode {episode} Finished. Avg Loss: {avg_loss:.5f}, Steps: {step_count}")
            
    torch.save(net.state_dict(), "td_backgammon.pth")

def get_obs_from_state(board, bar, off, perspective_player, score, cube_val, turn):
    # Re-implement Env logic for static state
    # Egocentric for `perspective_player`
    
    if perspective_player == 0:
        my_board = np.maximum(board, 0)
        opp_board = np.abs(np.minimum(board, 0))
    else:
        # P1 (Black/Negative)
        # Flip board so my home (0-5) is indices 0-5
        # Board: 0 is P1 Home? No, 0 is P0 Home. 23 is P1 Home.
        # Player 1 checkers are negative.
        # Board indices 23->0.
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
    
    # Turn: 1.0 if it's my turn?
    # In 'Afterstate', it's opponent's turn. 
    # But for 'Current State', it is my turn.
    # The network expects [Feature... Turn].
    # Standard: 1 if X is to move, 0 if O is to move?
    # Or "Current Player To Move" is always what we see?
    # Let's stick to "1.0 = It is MY turn".
    # Since we always feed Perspective = Current Player, Turn is always 1.0?
    # Unnecessary feature if always 1? But kept for compatibility.
    features.append(1.0 if turn == perspective_player else 0.0)
    
    features.append(cube_val / 64.0)
    
    return np.array(features, dtype=np.float32)

if __name__ == "__main__":
    main()
