# environment/custom_env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random

class WasteCollectionEnv(gym.Env):
    metadata = {'render_modes': ['human'], 'render_fps': 4}
    
    def __init__(self, grid_size=10, max_steps=200, render_mode=None):
        super(WasteCollectionEnv, self).__init__()
        self.grid_size = grid_size  # Wider environment
        self.max_steps = max_steps
        self.render_mode = render_mode

        # Actions: 0: forward (+z), 1: backward (-z), 2: right (+x), 3: left (-x), 4: pickup/drop.
        self.action_space = spaces.Discrete(5)
        
        # Observations: positions are in (x, y, z) with y fixed to 0 (on the ground), plus a carrying flag.
        self.observation_space = spaces.Dict({
            'agent': spaces.Box(0, grid_size-1, shape=(3,), dtype=np.int32),
            'waste': spaces.Box(0, grid_size-1, shape=(3,), dtype=np.int32),
            'bin': spaces.Box(0, grid_size-1, shape=(3,), dtype=np.int32),
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
        # Place the agent at the origin on the ground.
        self.agent_pos = np.array([0, 0, 0], dtype=np.int32)
        self.carrying_waste = False
        
        # Place waste and bin randomly on the ground (y = 0), ensuring no overlap with agent.
        self.waste_pos = self._random_position(exclude=[self.agent_pos])
        self.bin_pos = self._random_position(exclude=[self.agent_pos, self.waste_pos])
        
        # Generate houses. They will change positions every episode.
        self.houses = self._generate_houses(num_houses=3)
        
        return self._get_obs(), {}

    def _random_position(self, exclude=[]):
        while True:
            pos = np.array([random.randint(0, self.grid_size-1), 
                            0,  # Fixed on the ground
                            random.randint(0, self.grid_size-1)], dtype=np.int32)
            if all(not np.array_equal(pos, e) for e in exclude):
                return pos

    def _generate_houses(self, num_houses=3):
        """
        Generate random house positions (with a random size) ensuring that houses
        do not occupy the same cell as the waste product.
        """
        houses = []
        attempts = 0
        while len(houses) < num_houses and attempts < 1000:
            pos = np.array([random.randint(0, self.grid_size-1),
                            0,
                            random.randint(0, self.grid_size-1)], dtype=np.int32)
            # Ensure the house is not in the same cell as the waste.
            if np.array_equal(pos, self.waste_pos):
                attempts += 1
                continue
            # Check for duplicate positions among houses.
            duplicate = False
            for h in houses:
                if np.array_equal(pos, h['pos']):
                    duplicate = True
                    break
            if duplicate:
                attempts += 1
                continue
            # Random size for the house (adjust the range as desired).
            size = random.uniform(1.2, 2.0)
            houses.append({'pos': pos, 'size': size})
            attempts += 1
        return houses

    def _get_obs(self):
        return {
            'agent': self.agent_pos,
            'waste': self.waste_pos if not self.carrying_waste else np.array([-1, -1, -1]),
            'bin': self.bin_pos,
            'carrying': 1 if self.carrying_waste else 0
        }

    def step(self, action):
        self.steps += 1
        terminated = False
        truncated = self.steps >= self.max_steps
        reward = -0.1  # Step penalty

        # Movement actions (move on the x-z plane).
        if action < 4:
            new_pos = self.agent_pos.copy()
            # Action mapping:
            # 0: forward (+z), 1: backward (-z), 2: right (+x), 3: left (-x)
            if action == 0:
                new_pos[2] = min(self.grid_size-1, new_pos[2] + 1)
            elif action == 1:
                new_pos[2] = max(0, new_pos[2] - 1)
            elif action == 2:
                new_pos[0] = min(self.grid_size-1, new_pos[0] + 1)
            elif action == 3:
                new_pos[0] = max(0, new_pos[0] - 1)

            if not np.array_equal(new_pos, self.agent_pos):
                target = self.bin_pos if self.carrying_waste else self.waste_pos
                # Compute Manhattan distance on (x,z) only.
                prev_dist = np.abs(self.agent_pos[[0, 2]] - target[[0, 2]]).sum()
                new_dist = np.abs(new_pos[[0, 2]] - target[[0, 2]]).sum()
                self.agent_pos = new_pos
                if self.carrying_waste:
                    reward += (prev_dist - new_dist) * 0.2
                else:
                    reward += (prev_dist - new_dist) * 0.1
            else:
                reward -= 0.5  # Penalty for hitting a boundary.

        # Pickup/drop action.
        elif action == 4:
            if not self.carrying_waste:
                if np.array_equal(self.agent_pos, self.waste_pos):
                    self.carrying_waste = True
                    reward += 10  # Pickup reward.
                else:
                    reward -= 1  # Invalid pickup.
            else:
                if np.array_equal(self.agent_pos, self.bin_pos):
                    terminated = True
                    reward += 20  # Successful drop reward.
                else:
                    reward -= 1  # Invalid drop.

        # ---------- NEW: Check for collisions with buildings ----------
        # If the agent is in the same grid cell as any house, end the episode.
        if hasattr(self, 'houses'):
            for house in self.houses:
                if np.array_equal(self.agent_pos, house['pos']):
                    reward -= 20  # Heavy penalty for collision.
                    terminated = True
                    print("Collision with a building at position:", house['pos'])
                    break

        # Check if episode should end based on max steps.
        if terminated or truncated:
            if not terminated:
                reward -= 5

        return self._get_obs(), reward, terminated, truncated, {}

    def render(self):
        if self.render_mode == 'human':
            from environment.rendering import render_waste_env
            render_waste_env(self)
