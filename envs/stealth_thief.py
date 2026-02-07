import numpy as np
import pygame
import random
import os

class StealthThiefEnv:
    def __init__(self, grid_size=10, render_mode=None, max_steps=200):
        self.grid_size = grid_size
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.cell_size = 60
        
        # Start and Goal positions (Mapping 01 to (0,0) and 95 to (9,4))
        self.start_pos = (0, 0)
        self.goal_pos = (9, 4)
        
        # Police Station (Center 4 grids: 44, 45, 54, 55 -> (4,4), (4,5), (5,4), (5,5))
        self.station_zone = [(4, 4), (4, 5), (5, 4), (5, 5)]
        # Search radius (1 grid perimeter around station)
        self.search_zone = []
        for r in range(3, 7):
            for c in range(3, 7):
                if (r, c) not in self.station_zone:
                    self.search_zone.append((r, c))
        
        # Hardcoded Walls (as per user request: "block the straight ways")
        self.walls = [
            (1, 1), (1, 2), (1, 3), (1, 4), (1, 5),
            (3, 7), (4, 7), (5, 7), (6, 7),
            (8, 1), (8, 2), (8, 3), (8, 4)
        ]
        
        self.num_police = 4
        self.police_positions = []
        
        self.action_space = 4 # 0: U, 1: D, 2: L, 3: R
        # State: Agent Pos (2) + Goal Pos (2) + Station Center (2) + Police Pos (2*num_police)
        self.state_size = 2 + 2 + 2 + (2 * self.num_police)
        self.observation_space = self.state_size
        
        self._init_rendering()
        self.reset()

    def _init_rendering(self):
        self.assets_loaded = False
        if self.render_mode == "human":
            pygame.init()
            self.screen = pygame.display.set_mode((self.grid_size * self.cell_size, self.grid_size * self.cell_size))
            pygame.display.set_caption("Stealth Thief Escape")
            self.clock = pygame.time.Clock()
            self._load_assets()

    def _load_assets(self):
        try:
            asset_dir = os.path.join(os.path.dirname(__file__), '..', 'assets')
            self.thief_img = pygame.transform.scale(pygame.image.load(os.path.join(asset_dir, 'agent.png')), (self.cell_size, self.cell_size))
            self.wall_img = pygame.transform.scale(pygame.image.load(os.path.join(asset_dir, 'wall.png')), (self.cell_size, self.cell_size))
            self.police_img = pygame.transform.scale(pygame.image.load(os.path.join(asset_dir, 'police.png')), (self.cell_size, self.cell_size))
            # Load getaway vehicle (car)
            vehicle_path = os.path.join(asset_dir, 'car.png')
            if os.path.exists(vehicle_path):
                self.vehicle_img = pygame.transform.scale(pygame.image.load(vehicle_path), (self.cell_size, self.cell_size))
            self.assets_loaded = True
        except Exception as e:
            print(f"Warning: Assets not loaded ({e})")
            self.assets_loaded = False

    def reset(self):
        self.agent_pos = self.start_pos
        self.current_step = 0
        self._spawn_police()
        return self._get_state()

    def _spawn_police(self):
        self.police_positions = []
        forbidden = set([self.start_pos, self.goal_pos] + self.walls + self.station_zone)
        while len(self.police_positions) < self.num_police:
            pos = (random.randint(0, self.grid_size-1), random.randint(0, self.grid_size-1))
            if pos not in forbidden and pos not in self.police_positions:
                self.police_positions.append(pos)

    def _get_state(self):
        state = [
            self.agent_pos[0] / self.grid_size, self.agent_pos[1] / self.grid_size,
            self.goal_pos[0] / self.grid_size, self.goal_pos[1] / self.grid_size,
            4.5 / self.grid_size, 4.5 / self.grid_size # Station center
        ]
        for p in self.police_positions:
            state.extend([p[0] / self.grid_size, p[1] / self.grid_size])
        return np.array(state, dtype=np.float32)

    def step(self, action):
        self.current_step += 1
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        dx, dy = moves[action]
        
        new_x = max(0, min(self.grid_size-1, self.agent_pos[0] + dx))
        new_y = max(0, min(self.grid_size-1, self.agent_pos[1] + dy))
        new_pos = (new_x, new_y)
        
        reward = -0.1
        done = False
        info = {'status': 'active'}

        # Wall Collision
        if new_pos in self.walls:
            reward = -1.0
            new_pos = self.agent_pos # Stay in place
        else:
            self.agent_pos = new_pos

        # Goal reached
        if self.agent_pos == self.goal_pos:
            reward = 100.0
            done = True
            info['status'] = 'escaped'
        
        # Caught by Police
        elif self.agent_pos in self.police_positions:
            reward = -100.0
            done = True
            info['status'] = 'caught_by_police'
            
        # Entering Police Station or Search Radius (Busted)
        elif self.agent_pos in self.station_zone or self.agent_pos in self.search_zone:
            reward = -100.0
            done = True
            info['status'] = 'busted_by_surveillance'
            
        # Max steps
        if self.current_step >= self.max_steps:
            done = True
            info['status'] = 'timed_out'

        if self.render_mode == "human":
            self.render()

        return self._get_state(), reward, done, info

    def render(self):
        if not hasattr(self, 'screen'): return
        self.screen.fill((20, 20, 20)) # Dark city background
        
        # Draw search zone radius (faint red)
        for r, c in self.search_zone:
            rect = pygame.Rect(c * self.cell_size, r * self.cell_size, self.cell_size, self.cell_size)
            pygame.draw.rect(self.screen, (60, 0, 0), rect)

        # Draw police station (bright red)
        for r, c in self.station_zone:
            rect = pygame.Rect(c * self.cell_size, r * self.cell_size, self.cell_size, self.cell_size)
            pygame.draw.rect(self.screen, (150, 0, 0), rect)

        # Draw Walls
        for r, c in self.walls:
            if self.assets_loaded:
                self.screen.blit(self.wall_img, (c * self.cell_size, r * self.cell_size))
            else:
                pygame.draw.rect(self.screen, (100, 100, 100), (c * self.cell_size, r * self.cell_size, self.cell_size, self.cell_size))

        # Draw Police
        for r, c in self.police_positions:
            if self.assets_loaded:
                self.screen.blit(self.police_img, (c * self.cell_size, r * self.cell_size))
            else:
                pygame.draw.circle(self.screen, (0, 0, 255), (c * self.cell_size + 30, r * self.cell_size + 30), 20)

        # Draw Goal (Vehicle)
        gr, gc = self.goal_pos
        if self.assets_loaded and hasattr(self, 'vehicle_img'):
            self.screen.blit(self.vehicle_img, (gc * self.cell_size, gr * self.cell_size))
        else:
            pygame.draw.rect(self.screen, (255, 255, 0), (gc * self.cell_size + 10, gr * self.cell_size + 10, 40, 40))

        # Draw Thief
        tr, tc = self.agent_pos
        if self.assets_loaded:
            self.screen.blit(self.thief_img, (tc * self.cell_size, tr * self.cell_size))
        else:
            pygame.draw.rect(self.screen, (0, 255, 0), (tc * self.cell_size + 15, tr * self.cell_size + 15, 30, 30))

        pygame.display.flip()
        self.clock.tick(10)

    def close(self):
        if hasattr(self, 'screen'):
            pygame.quit()
