import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import defaultdict, deque
import pickle
import os
from typing import Dict, List, Tuple, Optional
from game import BackgammonGame, GamePhase, get_obs_from_state
from model import BackgammonValueNet

class InformationSet:
    """
    Represents an information set in CFR for backgammon.
    An information set is defined by the game state from the perspective of a player.
    """
    
    def __init__(self, obs_hash: str, legal_actions: List[int]):
        self.obs_hash = obs_hash
        self.legal_actions = legal_actions
        self.num_actions = len(legal_actions)
        
        # CFR algorithm variables
        self.regret_sum = np.zeros(self.num_actions)
        self.strategy_sum = np.zeros(self.num_actions)
        self.strategy = np.ones(self.num_actions) / self.num_actions
        
    def get_strategy(self, weight: float = 1.0) -> np.ndarray:
        """Get current strategy using regret matching"""
        strategy = np.maximum(self.regret_sum, 0)
        if np.sum(strategy) > 0:
            strategy = strategy / np.sum(strategy)
        else:
            strategy = np.ones(self.num_actions) / self.num_actions
            
        self.strategy = strategy
        self.strategy_sum += weight * strategy
        return strategy
    
    def get_average_strategy(self) -> np.ndarray:
        """Get average strategy for Nash equilibrium computation"""
        if np.sum(self.strategy_sum) > 0:
            return self.strategy_sum / np.sum(self.strategy_sum)
        else:
            return np.ones(self.num_actions) / self.num_actions

class CFRTrainer:
    """
    Counterfactual Regret Minimization trainer for backgammon.
    Implements chance sampling and external sampling for efficiency.
    """
    
    def __init__(
        self,
        device: torch.device = None,
        lr_policy: float = 1e-3,
        lr_value: float = 1e-3,
        gamma: float = 0.99,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.game = BackgammonGame()
        self.lr_policy = lr_policy
        self.lr_value = lr_value
        self.gamma = gamma
        
        # CFR data structures
        self.infosets: Dict[str, InformationSet] = {}
        
        # Neural network for function approximation
        self.policy_net = BackgammonValueNet().to(self.device)
        self.value_net = BackgammonValueNet().to(self.device)
        
        # Optimizers
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr_policy)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=self.lr_value)
        
        # Training parameters
        self.iteration = 0
        self.exploration_prob = 0.0
        self.discount_factor = self.gamma
        
        # Statistics
        self.convergence_history = []
        
    def get_observation_hash(self, board: np.ndarray, bar: np.ndarray, 
                           off: np.ndarray, player: int, dice: Tuple[int, int]) -> str:
        """Create a hash for the current information set"""
        # Create a compact representation of the game state
        obs = get_obs_from_state(board, bar, off, player, self.game.score,
                                self.game.cube_value, player)
        # Include dice in hash — same board with different dice yields different legal moves
        dice_bytes = np.array(sorted(dice), dtype=np.int8).tobytes()
        return hash(obs.tobytes() + dice_bytes)
    
    def get_or_create_infoset(self, obs_hash: str, legal_actions: List[int]) -> InformationSet:
        """Get existing infoset or create new one"""
        if obs_hash not in self.infosets:
            self.infosets[obs_hash] = InformationSet(obs_hash, legal_actions)
        return self.infosets[obs_hash]
    
    def sample_action(self, infoset: InformationSet, explore: bool = False) -> int:
        """Sample action from strategy with exploration"""
        strategy = infoset.get_strategy()
        
        if explore and random.random() < self.exploration_prob:
            # Explore: random action
            return random.choice(range(len(strategy)))
        else:
            # Exploit: sample from strategy
            return np.random.choice(len(strategy), p=strategy)
    
    def cfr_recursive(self, player: int, reach_prob: float, 
                     opp_reach_prob: float) -> Tuple[float, np.ndarray]:
        """
        Recursive CFR implementation with chance sampling.
        Returns the value for the current player and action values.
        """
        # Handle terminal states
        if self.game.phase == GamePhase.GAME_OVER:
            return self._get_terminal_value(player), np.array([0.0])
        
        # Handle chance nodes (dice rolls)
        if self.game.phase == GamePhase.DECIDE_CUBE_OR_ROLL:
            # Always roll for now (simplified)
            self.game.step(0)
            return self.cfr_recursive(player, reach_prob, opp_reach_prob)
        
        # Get current state
        current_player = self.game.turn
        board, bar, off = self.game.board, self.game.bar, self.game.off
        dice = self.game.dice if hasattr(self.game, 'dice') else (0, 0)
        
        # Get information set
        obs_hash = self.get_observation_hash(board, bar, off, current_player, dice)
        legal_actions = list(range(len(self.game.legal_moves)))
        infoset = self.get_or_create_infoset(obs_hash, legal_actions)
        
        # Get strategy for current infoset
        strategy = infoset.get_strategy(reach_prob if current_player == player else opp_reach_prob)
        
        # Initialize action values
        action_values = np.zeros(len(strategy))
        node_value = 0.0
        
        # Iterate over all actions
        for i, action in enumerate(legal_actions):
            # Store current state manually
            saved_board = self.game.board.copy()
            saved_bar = self.game.bar.copy()
            saved_off = self.game.off.copy()
            saved_turn = self.game.turn
            saved_phase = self.game.phase
            saved_cube_value = self.game.cube_value
            saved_cube_owner = self.game.cube_owner
            saved_score = self.game.score.copy()
            saved_dice = self.game.dice.copy()
            saved_current_roll = list(self.game.current_roll)  # Convert tuple to list for copying
            saved_legal_moves = self.game.legal_moves.copy()
            saved_winner = self.game.winner
            
            # Execute action
            self.game.step(action)
            
            # Recursively compute value
            if current_player == player:
                action_value, _ = self.cfr_recursive(player, reach_prob * strategy[i], opp_reach_prob)
            else:
                action_value, _ = self.cfr_recursive(player, reach_prob, opp_reach_prob * strategy[i])
            
            # Restore state manually
            self.game.board = saved_board
            self.game.bar = saved_bar
            self.game.off = saved_off
            self.game.turn = saved_turn
            self.game.phase = saved_phase
            self.game.cube_value = saved_cube_value
            self.game.cube_owner = saved_cube_owner
            self.game.score = saved_score
            self.game.dice = saved_dice
            self.game.current_roll = saved_current_roll
            self.game.legal_moves = saved_legal_moves
            self.game.winner = saved_winner
            
            action_values[i] = action_value
            node_value += strategy[i] * action_value
        
        # Update regrets if this is the current player's node
        if current_player == player:
            for i, action in enumerate(legal_actions):
                regret = action_values[i] - node_value
                infoset.regret_sum[i] += opp_reach_prob * regret
        
        return node_value, action_values
    
    def _get_terminal_value(self, player: int) -> float:
        """Get terminal value for the given player"""
        if self.game.phase != GamePhase.GAME_OVER:
            return 0.0
        
        # Simple win/loss value
        if self.game.score[player] > self.game.score[1-player]:
            return 1.0
        else:
            return -1.0
    
    def train_iteration(self) -> Dict[str, float]:
        """Run one iteration of CFR training"""
        self.iteration += 1
        
        # Update exploration probability (decay over time)
        self.exploration_prob = max(0.0, 0.1 * (0.99 ** self.iteration))
        
        # Reset game
        self.game.reset_match()
        
        # Run CFR for both players
        p0_value, _ = self.cfr_recursive(0, 1.0, 1.0)
        p1_value, _ = self.cfr_recursive(1, 1.0, 1.0)
        
        # Update neural networks with function approximation
        self._update_neural_networks()
        
        # Compute convergence metrics
        avg_regret = self._compute_average_regret()
        strategy_stability = self._compute_strategy_stability()
        
        return {
            'iteration': self.iteration,
            'p0_value': p0_value,
            'p1_value': p1_value,
            'avg_regret': avg_regret,
            'strategy_stability': strategy_stability,
            'num_infosets': len(self.infosets)
        }
    
    def _update_neural_networks(self):
        """Update neural networks using sampled data from infosets"""
        if len(self.infosets) < 100:
            return
        
        # Sample batch of infosets
        sample_size = min(64, len(self.infosets))
        sampled_infosets = random.sample(list(self.infosets.values()), sample_size)
        
        # Only train on infosets that have the same num_actions (shapes must match for batching)
        # Group by action count and pick the largest group
        from collections import defaultdict as _dd
        groups = _dd(list)
        for infoset in sampled_infosets:
            groups[infoset.num_actions].append(infoset)
        largest_group = max(groups.values(), key=len)

        states = []
        target_values = []
        for infoset in largest_group:
            # Placeholder observation — replace with real stored obs when available
            obs_vector = np.random.randn(198)
            states.append(obs_vector)
            # Use the mean strategy value as a scalar training target
            target_values.append([float(np.mean(infoset.get_average_strategy()))])

        states_tensor = torch.tensor(np.array(states), dtype=torch.float32).to(self.device)
        targets_tensor = torch.tensor(np.array(target_values), dtype=torch.float32).to(self.device)

        # Update policy network — value_logits shape (batch, 6); compare against first output dim
        self.policy_optimizer.zero_grad()
        value_logits, _ = self.policy_net(states_tensor)
        policy_loss = nn.MSELoss()(value_logits[:, :1], targets_tensor)
        policy_loss.backward()
        self.policy_optimizer.step()
    
    def _compute_average_regret(self) -> float:
        """Compute average regret across all infosets"""
        if not self.infosets:
            return 0.0
        
        total_regret = 0.0
        total_actions = 0
        
        for infoset in self.infosets.values():
            total_regret += np.sum(np.abs(infoset.regret_sum))
            total_actions += len(infoset.regret_sum)
        
        return total_regret / max(total_actions, 1)
    
    def _compute_strategy_stability(self) -> float:
        """Compute how stable strategies are across iterations"""
        if len(self.convergence_history) < 2:
            return 1.0
        
        # Compare current average strategy with previous iteration
        current_avg_regret = self._compute_average_regret()
        prev_avg_regret = self.convergence_history[-1]['avg_regret']
        
        if prev_avg_regret == 0:
            return 1.0
        
        return min(1.0, abs(current_avg_regret - prev_avg_regret) / prev_avg_regret)
    
    def get_strategy(self, board: np.ndarray, bar: np.ndarray, off: np.ndarray, 
                    player: int, dice: Tuple[int, int]) -> np.ndarray:
        """Get the current strategy for a given state"""
        obs_hash = self.get_observation_hash(board, bar, off, player, dice)
        
        # Get legal actions by temporarily setting up the game state
        saved_state = self.game.save_state()
        # This is simplified - in practice we'd need to properly set the state
        legal_actions = list(range(20))  # Placeholder
        
        infoset = self.get_or_create_infoset(obs_hash, legal_actions)
        return infoset.get_average_strategy()
    
    def save_checkpoint(self, filepath: str):
        """Save training checkpoint"""
        checkpoint = {
            'iteration': self.iteration,
            'infosets': self.infosets,
            'lr_policy': self.lr_policy,
            'lr_value': self.lr_value,
            'gamma': self.gamma,
            'policy_net_state': self.policy_net.state_dict(),
            'value_net_state': self.value_net.state_dict(),
            'policy_optimizer_state': self.policy_optimizer.state_dict(),
            'value_optimizer_state': self.value_optimizer.state_dict(),
            'convergence_history': self.convergence_history
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(checkpoint, f)
    
    def load_checkpoint(self, filepath: str):
        """Load training checkpoint"""
        with open(filepath, 'rb') as f:
            checkpoint = pickle.load(f)
        
        self.iteration = checkpoint['iteration']
        self.infosets = checkpoint['infosets']
        self.lr_policy = checkpoint.get('lr_policy', self.lr_policy)
        self.lr_value = checkpoint.get('lr_value', self.lr_value)
        self.gamma = checkpoint.get('gamma', self.gamma)
        self.discount_factor = self.gamma
        self.policy_net.load_state_dict(checkpoint['policy_net_state'])
        self.value_net.load_state_dict(checkpoint['value_net_state'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state'])
        self.value_optimizer.load_state_dict(checkpoint['value_optimizer_state'])
        self.convergence_history = checkpoint['convergence_history']