# test_model.py
import time
import pygame
from stable_baselines3 import PPO
from environment.custom_env import WasteCollectionEnv

# Load the trained PPO model
model_path = "models/ppo_waste_collector/ppo_model"  # Make sure the model exists
model = PPO.load(model_path)

# Initialize environment with rendering
env = WasteCollectionEnv(render_mode="human")
obs, _ = env.reset()

# Initialize Pygame window for rendering
pygame.init()
screen = pygame.display.set_mode((800, 600), pygame.OPENGL | pygame.DOUBLEBUF)
clock = pygame.time.Clock()

done = False
truncated = False

while not (done or truncated):
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True
            break

    # Predict action using the trained model
    action, _states = model.predict(obs, deterministic=True)

    # Step environment
    obs, reward, done, truncated, info = env.step(action)

    # Render environment
    env.render()

    clock.tick(2)  # Limit FPS to match render_fps

pygame.quit()
print("🏁 Testing finished.")
