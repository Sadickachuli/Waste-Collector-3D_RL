# training/training.py
import os
import gymnasium as gym
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from environment.custom_env import WasteCollectionEnv

# Create environment (vectorized for stable training)
def create_env():
    return WasteCollectionEnv(grid_size=10, max_steps=200)

# Vectorized version for faster training
env = make_vec_env(create_env, n_envs=4)

# Define model
model = PPO(
    "MultiInputPolicy",  # For Dict observations
    env,
    verbose=1,
    tensorboard_log="./logs",  # Optional: view in TensorBoard
    learning_rate=2.5e-4,
    n_steps=1024,
    batch_size=64,
    gae_lambda=0.95,
    gamma=0.99,
    ent_coef=0.01,
)

# Train model
model.learn(total_timesteps=200_000)

# Save model
save_path = "models/ppo_waste_collector"
os.makedirs(save_path, exist_ok=True)
model.save(os.path.join(save_path, "ppo_model"))

print("✅ Training complete. Model saved to:", save_path)
