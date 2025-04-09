# main.py
import pygame
from environment.custom_env import WasteCollectionEnv

def main():
    pygame.init()
    screen = pygame.display.set_mode((800, 600), pygame.OPENGL | pygame.DOUBLEBUF)
    
    env = WasteCollectionEnv(grid_size=10, max_steps=200, render_mode='human')
    obs, _ = env.reset()
    
    running = True
    clock = pygame.time.Clock()
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        action = env.action_space.sample()  # For testing, choose random actions.
        obs, reward, terminated, truncated, info = env.step(action)
        
        from environment.rendering import render_waste_env
        render_waste_env(env, screen)
        
        if terminated or truncated:
            obs, _ = env.reset()
        
        clock.tick(10)  # Limit to 10 FPS.

    pygame.quit()

if __name__ == '__main__':
    main()
