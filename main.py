# play.py
import pygame
import time
import random
from environment.custom_env import WasteCollectionEnv
from environment.rendering import render_waste_env

class GameState:
    def __init__(self):
        self.total_reward = 0.0
        self.steps = 0
        self.episode = 1
        self.ep_start_time = time.time()

def log_episode(state):
    elapsed = time.time() - state.ep_start_time
    avg_reward = state.total_reward / state.steps if state.steps > 0 else 0
    print(f"Episode {state.episode} finished: Total Reward: {state.total_reward:.2f} | "
          f"Steps: {state.steps} | Avg Reward/Step: {avg_reward:.2f} | Time: {elapsed:.2f}s")
    print("-" * 60)

def play():
    pygame.init()
    window_size = 600
    screen = pygame.display.set_mode((window_size, window_size), pygame.OPENGL | pygame.DOUBLEBUF)
    pygame.display.set_caption("3D Waste Collection")

    env = WasteCollectionEnv(grid_size=7, max_steps=100, render_mode='human')
    obs, _ = env.reset()
    clock = pygame.time.Clock()
    state = GameState()

    running = True
    while running:
        clock.tick(5)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        action = random.choice([0, 1, 2, 3, 4])
        obs, reward, terminated, truncated, _ = env.step(action)

        state.total_reward += reward
        state.steps += 1
        render_waste_env(env, screen)

        if terminated or truncated:
            log_episode(state)
            obs, _ = env.reset()
            state = GameState()

    pygame.quit()

if __name__ == '__main__':
    play()
