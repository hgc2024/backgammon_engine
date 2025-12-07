import os
import gymnasium as gym
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
from sb3_contrib.common.wrappers import ActionMasker
import argparse
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from src.env import BackgammonEnv, create_env

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_cpu", type=int, default=20, help="Number of CPUs to use")
    parser.add_argument("--steps", type=int, default=10_000_000, help="Total training steps")
    args = parser.parse_args()
    
    num_cpu = args.n_cpu
    
    # Create vectorized environment
    if num_cpu > 1:
        # ensure create_env is picklable
        env = SubprocVecEnv([create_env for _ in range(num_cpu)])
    else:
        env = DummyVecEnv([create_env])
    
    # Neural Network Architecture
    # Policy: MLP
    # RTX 4050 can handle large networks.
    # Input 198 -> [512, 512] -> Actions
    policy_kwargs = dict(
        net_arch=[512, 512, 256] # Deep network
    )
    
    # PPO Hyperparameters
    # High batch size for GPU efficiency?
    # n_steps * num_cpu = total_steps_per_update
    # 2048 * 20 = 40960 steps per update.
    # Maybe lower n_steps if we want more frequent updates?
    # But PPO likes large batches.
    
    model = MaskablePPO(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        n_steps=1024, # 1024 * 20 = 20480 buffer
        batch_size=2048,
        gamma=0.99, # Discount factor
        ent_coef=0.01, # Exploration
        learning_rate=3e-4,
        tensorboard_log="./backgammon_tensorboard/"
    )
    
    print("Starting training...")
    
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path='./checkpoints/',
        name_prefix='ppu_backgammon'
    )
    
    # Train
    # Set total timesteps huge. Can be stopped manually.
    model.learn(total_timesteps=args.steps, callback=checkpoint_callback)
    
    model.save("backgammon_final")
    print("Training finished.")

if __name__ == "__main__":
    main()
