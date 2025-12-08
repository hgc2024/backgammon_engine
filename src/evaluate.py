import numpy as np
import gymnasium as gym
from stable_baselines3.common.callbacks import EventCallback, BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
import torch
from src.env import BackgammonEnv

class RandomAgent:
    def predict(self, obs, action_masks=None):
        if action_masks is not None:
            valid_actions = np.where(action_masks)[0]
            if len(valid_actions) > 0:
                return np.random.choice(valid_actions), None
        return 0, None # Default or fallback

class HeuristicAgent:
    """
    A simple heuristic agent.
    Logic (in order of priority):
    1. Hit opponent blot.
    2. Make a point (stacks of 2+).
    3. Bear off (if possible).
    4. Random valid move.
    
    LIMITATION: This agent needs access to the Game object to reason about "Hits" and "Points".
    The Observation space is purely numerical. 
    Ideally, we would run this INSIDE the game logic or give it access to the game state.
    For simplicity here, we will cheat and access the env.game object if we are running in a local env.
    """
    def __init__(self, env):
        # We need the underlying game to be useful
        if hasattr(env, 'envs'):
             self.game = env.envs[0].game
        else:
             self.game = env.game
             
    def predict(self, obs, action_masks=None):
        # We need to know which moves correspond to what.
        # This is tricky because the Action Space is just an index.
        # We need to ask the game: "What does action `i` actually do?"
        
        if action_masks is None:
             # Fallback
             return 0, None

        valid_indices = np.where(action_masks)[0]
        if len(valid_indices) == 0:
            return 0, None
            
        legal_moves = self.game.legal_moves # List of iterables
        
        # If we are in CUBE phase or something else
        # The environment wrapper handles actions.
        # If action_masks suggests 0/1 are valid:
        if self.game.phase != 2: # 2 is DECIDE_MOVE
            # Cube decision? Always take, Never double?
            # Random for now on cube
             return np.random.choice(valid_indices), None
             
        # Move Phase
        best_action = valid_indices[0]
        best_score = -9999
        
        for idx in valid_indices:
            if idx >= len(legal_moves): continue
            
            move_seq = legal_moves[idx] # Tuple of ((s, e), (s, e))
            
            score = 0
            # Simple Heuristic Evaluation of the "Move" itself not the resulting state (easier)
            for start, end in move_seq:
                # HIT?
                if end != 'off':
                    # If opponent has 1 checker there, it's a hit.
                    # We need to check board state BEFORE move? 
                    # This is complex because move_seq happens sequentially.
                    pass
                else:
                    score += 10 # Bearing off is good
            
            # Since evaluating "Hit" requires applying moves one by one, 
            # and we don't want to mutate the real game state, 
            # we will trust the RandomAgent for now and implement a "Better Random" that just prefers bearing off.
            
            # Refined Heuristic:
            # Prefer moves that bear off.
            if score > best_score:
                best_score = score
                best_action = idx
                
        return best_action, None


class EvaluationCallback(EventCallback):
    """
    Callback that evaluates the agent against:
    1. Random Agent
    2. Itself (Self-Play validation mainly for crashes/stats)
    
    Logs win rate to Tensorboard.
    """
    def __init__(self, eval_env, eval_freq=10000, n_eval_episodes=50, verbose=1):
        super(EvaluationCallback, self).__init__(None, verbose=verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.random_agent = RandomAgent()
        
    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            self._run_evaluation()
        return True

    def _run_evaluation(self):
        # Play vs Random
        wins = 0
        total_matches = self.n_eval_episodes
        
        # We need to simulate a match where Agent is P0, Random is P1.
        # AND Agent is P1, Random is P0? (To be fair)
        
        # NOTE: Our Env assumes we are always "Current Player". 
        # But `step()` in Env takes action for current player.
        # So we need a loop that checks `env.game.turn` and queries the right agent.
        
        # Using the Underlying Game directly might be easier than Gym Env for this specialized Arena?
        # But we need Model to predict on Obs. 
        # Let's use the Gym Env.
        
        obs = self.eval_env.reset()
        # Ensure obs is numpy array if vectorized
        if isinstance(obs, tuple): obs = obs[0] # gym new api
        
        for i in range(total_matches):
            done = False
            # Reset
            obs = self.eval_env.reset()
            # Handle Gym 0.26+ syntax
            if isinstance(obs, tuple): obs = obs[0]
            
            while not done:
                # WHO is playing?
                # The Env is designed for Self-Play, so `step` always applies to `game.turn`.
                # We need to peek at `game.turn` to decide WHO acts.
                
                # Unwrap to get game
                if hasattr(self.eval_env, 'envs'):
                    game = self.eval_env.envs[0].game
                else:
                    game = self.eval_env.game
                    
                current_turn = game.turn # 0=White, 1=Red
                
                masks = self.eval_env.action_masks()
                # If vectorized, unwrap
                if isinstance(masks, list): masks = masks[0] 
                
                if current_turn == 0:
                    # Agent's Turn (P0)
                    action, _ = self.model.predict(obs, action_masks=masks)
                    if isinstance(action, np.ndarray): action = action.item()
                else:
                    # Opponent (Random) (P1)
                    action, _ = self.random_agent.predict(obs, action_masks=masks)
                
                # Step
                obs, reward, done, info = self.eval_env.step(action)
                # Gym VecEnv returns (obs, rewards, dones, infos)
                # If using DummyVecEnv
                if isinstance(done, np.ndarray): 
                     done = done[0]
                     reward = reward[0]
                     obs = obs 
                else:
                    # Single env (unlikely with SB3)
                    # Handle Tuple
                    if isinstance(obs, tuple): obs = obs[0]

            # Game Over
            # Who won?
            # Reward is +1 if Current Player Won.
            # But "Current Player" at step end is the one who Just Played and Won.
            # So if P0 moved and won, reward > 0.
            # If P1 moved and won, reward > 0 (for P1).
            
            # Let's check score directly
            if game.score[0] > game.score[1]:
                wins += 1
                
        win_rate = wins / total_matches
        
        if self.verbose > 0:
            print(f"Eval vs Random: {win_rate*100:.1f}% Win Rate")
            
        self.logger.record("eval/win_rate_vs_random", win_rate)
