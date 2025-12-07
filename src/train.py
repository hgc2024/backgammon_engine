import os
import gymnasium as gym
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
from sb3_contrib.common.wrappers import ActionMasker
import argparse
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from src.env import BackgammonEnv, create_env

from src.features import BackgammonResNet

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
    # Policy: MlpPolicy (but with custom feature extractor)
    # The MlpPolicy will put a head on top of our features_dim output.
    policy_kwargs = dict(
        features_extractor_class=BackgammonResNet,
        features_extractor_kwargs=dict(features_dim=512),
        net_arch=[dict(pi=[256, 128], vf=[256, 128])] # Heads
    )
    
    # PPO Hyperparameters
    # High batch size for GPU efficiency?
    # n_steps * num_cpu = total_steps_per_update
    # 2048 * 20 = 40960 steps per update.
    # Maybe lower n_steps if we want more frequent updates?
    # But PPO likes large batches.
    
    # PPO Hyperparameters
    model = MaskablePPO(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        n_steps=1024,
        batch_size=2048,
        gamma=0.99,
        ent_coef=0.01,
        learning_rate=3e-4,
        tensorboard_log="./backgammon_tensorboard/",
        device="cuda" # Explicitly use GPU
    )
    
    print(f"Training on device: {model.device}")
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
