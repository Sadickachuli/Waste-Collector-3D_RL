# environment/custom_env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random

class WasteCollectionEnv(gym.Env):
    metadata = {'render_modes': ['human'], 'render_fps': 4}
    
    def __init__(self, grid_size=5, max_steps=100, render_mode=None):
        super(WasteCollectionEnv, self).__init__()
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.render_mode = render_mode

        # Action space: 0-3=movement, 4=pickup/drop
        self.action_space = spaces.Discrete(5)
        
        # Observation space: agent, waste, bin positions + carrying status
        self.observation_space = spaces.Dict({
            'agent': spaces.Box(0, grid_size-1, shape=(2,), dtype=np.int32),
            'waste': spaces.Box(0, grid_size-1, shape=(2,), dtype=np.int32),
            'bin': spaces.Box(0, grid_size-1, shape=(2,), dtype=np.int32),
            'carrying': spaces.Discrete(2)
        })

        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        self.agent_pos = np.array([0, 0], dtype=np.int32)
        self.carrying_waste = False
        
        # Place waste and bin ensuring they're not at agent's start
        self.waste_pos = self._random_position(exclude=[self.agent_pos])
        self.bin_pos = self._random_position(exclude=[self.agent_pos, self.waste_pos])
        
        return self._get_obs(), {}

    def _random_position(self, exclude=[]):
        while True:
            pos = np.array([random.randint(0, self.grid_size-1), 
                           random.randint(0, self.grid_size-1)], dtype=np.int32)
            if all(not np.array_equal(pos, e) for e in exclude):
                return pos

    def _get_obs(self):
        return {
            'agent': self.agent_pos,
            'waste': self.waste_pos if not self.carrying_waste else np.array([-1, -1]),
            'bin': self.bin_pos,
            'carrying': 1 if self.carrying_waste else 0
        }

    def step(self, action):
        self.steps += 1
        terminated = False
        truncated = self.steps >= self.max_steps
        reward = -0.1  # Step penalty
        
        # Movement actions
        if action <= 3:
            new_pos = self.agent_pos.copy()
            if action == 0: new_pos[1] = max(0, new_pos[1]-1)          # Up
            elif action == 1: new_pos[1] = min(self.grid_size-1, new_pos[1]+1) # Down
            elif action == 2: new_pos[0] = max(0, new_pos[0]-1)        # Left
            elif action == 3: new_pos[0] = min(self.grid_size-1, new_pos[0]+1) # Right
            
            if not np.array_equal(new_pos, self.agent_pos):
                self.agent_pos = new_pos
                # Reward based on moving toward target: if carrying, toward bin; else, toward waste.
                if self.carrying_waste:
                    prev_dist = np.abs(self.agent_pos - self.bin_pos).sum()
                    new_dist = np.abs(new_pos - self.bin_pos).sum()
                    reward += (prev_dist - new_dist) * 0.2
                else:
                    prev_dist = np.abs(self.agent_pos - self.waste_pos).sum()
                    new_dist = np.abs(new_pos - self.waste_pos).sum()
                    reward += (prev_dist - new_dist) * 0.1
            else:
                reward -= 0.5  # Wall hit penalty

        # Pickup/drop action
        elif action == 4:
            if not self.carrying_waste:
                if np.array_equal(self.agent_pos, self.waste_pos):
                    self.carrying_waste = True
                    reward += 10  # Pickup reward
                else:
                    reward -= 1  # Invalid pickup attempt
            else:
                if np.array_equal(self.agent_pos, self.bin_pos):
                    terminated = True
                    reward += 20  # Successful drop reward
                else:
                    reward -= 1  # Invalid drop attempt

        # Check for episode termination
        if terminated or truncated:
            if not terminated:
                reward -= 5

        return self._get_obs(), reward, terminated, truncated, {}

    def render(self):
        if self.render_mode == 'human':
            from environment.rendering import render_waste_env
            render_waste_env(self)