import os
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.logger import configure
from environment.custom_env import WasteCollectionEnv

# Custom callback to log rewards and entropy
class TrainingLoggerCallback(BaseCallback):
    def __init__(self, log_dir, verbose=1):
        super(TrainingLoggerCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.entropy_values = []
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

    def _on_step(self) -> bool:
        # Log episode rewards
        if "infos" in self.locals and len(self.locals["infos"]) > 0:
            episode_reward = self.locals["infos"][0].get("episode", {}).get("r")
            if episode_reward is not None:
                self.episode_rewards.append(episode_reward)
                print(f"Episode Reward Logged: {episode_reward}") 

        # Log entropy value from policy
        entropy = float(self.model.logger.name_to_value.get("entropy_loss", 0))
        self.entropy_values.append(entropy)
        print(f"Entropy Logged: {entropy}") 

        return True

    def on_training_end(self):
        # Save logged values to numpy files
        np.save(os.path.join(self.log_dir, "reward_history.npy"), np.array(self.episode_rewards, dtype=np.float32))
        np.save(os.path.join(self.log_dir, "entropy_history.npy"), np.array(self.entropy_values, dtype=np.float32))

def train():
    env = WasteCollectionEnv(grid_size=5, max_steps=100)
    models_dir = "models/pg/"
    log_dir = "logs/"

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    model = PPO(
        "MultiInputPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        gamma=0.99,
        batch_size=64,
        n_steps=2048,
        n_epochs=10,
        ent_coef=0.01,
        tensorboard_log=log_dir 
    )

    # Set up TensorBoard logger
    new_logger = configure(log_dir, ["stdout", "tensorboard"])
    model.set_logger(new_logger)

    # Initialize custom logging callback
    training_logger = TrainingLoggerCallback(log_dir)
    
    # Evaluation callback
    eval_callback = EvalCallback(env, best_model_save_path=models_dir, log_path=log_dir, eval_freq=5000, deterministic=True, render=False)

    # Train model with both callbacks
    model.learn(total_timesteps=100000, callback=[training_logger, eval_callback])
    
    model.save(os.path.join(models_dir, "ppo_collection(3)"))
    print("Training completed and model saved.")

if __name__ == '__main__':
    train()