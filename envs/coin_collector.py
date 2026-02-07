import numpy as np
import pygame
import random
import os

class CoinCollectorEnv:
    def __init__(self, grid_size=10, num_coins=3, render_mode=None, max_steps=200):
        self.grid_size = grid_size
        self.num_coins = num_coins
        self.render_mode = render_mode
        self.max_steps = max_steps  # Maximum steps per episode
        self.current_step = 0  # Track current step count
        self.coins_collected = 0  # Track coins collected in current episode
        self.grid = np.zeros((grid_size, grid_size))
        self.agent_pos = (0, 0)
        self.coins = []
        self.action_space = 4  # Up, Down, Left, Right
        self.state_size = grid_size * grid_size * 2 + num_coins * 2
        self.observation_space = self.state_size
        
        # Initialize rendering assets
        self._init_rendering()
        
        self.reset()
    
    def _load_assets(self):
        """Load game assets with error handling"""
        try:
            # Default assets directory
            asset_dir = os.path.join(os.path.dirname(__file__), '..', 'assets')
                    
            # Load coin image (fallback to circle)
            coin_path = os.path.join(asset_dir, 'coin.png')
            if os.path.exists(coin_path):
                self.coin_img = pygame.image.load(coin_path).convert_alpha()
                self.coin_img = pygame.transform.scale(
                    self.coin_img,
                    (self.cell_size//2, self.cell_size//2))
            
            # Load agent image (fallback to rectangle)
            agent_path = os.path.join(asset_dir, 'agent.png')
            if os.path.exists(agent_path):
                self.agent_img = pygame.image.load(agent_path).convert_alpha()
                self.agent_img = pygame.transform.scale(
                    self.agent_img,
                    (self.cell_size, self.cell_size))
            
            self.assets_loaded = True
        except Exception as e:
            print(f"Warning: Could not load assets ({e}). Using fallback rendering.")
            self.assets_loaded = False


    def _init_rendering(self):
        """Initialize rendering-related attributes"""
        self.assets_loaded = False
        self.cell_size = 60
        if self.render_mode == "human":
            pygame.init()
            self.screen = pygame.display.set_mode(
                (self.grid_size * self.cell_size, self.grid_size * self.cell_size))
            pygame.display.set_caption("Coin Collector RL")
            self.clock = pygame.time.Clock()
            self._load_assets()

    def reset(self):
        """Reset the environment and return initial state."""
        self.grid.fill(0)
        self.agent_pos = (0, 0)
        self.coins = self._spawn_coins()
        self.current_step = 0
        self.coins_collected = 0
        
        # Update grid representation
        self.grid[self.agent_pos] = 2  # Agent
        for coin in self.coins:
            self.grid[coin] = 1  # Coins
            
        if self.render_mode == "human":
            self.render()
            
        return self._get_state()

    def step(self, action):
        """Execute action and return (next_state, reward, done, info)."""
        self.current_step += 1
        
        # Action mapping: 0=Up, 1=Down, 2=Left, 3=Right
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        dx, dy = moves[action]
        
        # Calculate new position
        new_x = max(0, min(self.grid_size-1, self.agent_pos[0] + dx))
        new_y = max(0, min(self.grid_size-1, self.agent_pos[1] + dy))
        new_pos = (new_x, new_y)
        
        # Update grid
        self.grid[self.agent_pos] = 0  # Clear old position
        self.agent_pos = new_pos
        self.grid[self.agent_pos] = 2  # Mark new position
        
        # Initialize return values
        reward = -0.1  # Small penalty for each step
        done = False
        collected = False
        
        # Check for coin collection
        if self.agent_pos in self.coins:
            self.coins.remove(self.agent_pos)
            reward += 10
            collected = True
            self.coins_collected += 1
            
            if len(self.coins) == 0:  # All coins collected
                reward += 50  # Completion bonus
                done = True
            else:
                # Spawn new coin
                new_coin = self._spawn_coins()[0]
                self.coins.append(new_coin)
                self.grid[new_coin] = 1
        
        # Check if max steps reached
        if self.current_step >= self.max_steps:
            done = True
        
        # Get next state
        next_state = self._get_state()
        
        # Prepare info dictionary
        info = {
            'collected': collected,
            'coins_collected': self.coins_collected,
            'steps_remaining': self.max_steps - self.current_step
        }
        
        # Render if human mode
        if self.render_mode == "human":
            self.render()
            
        return next_state, reward, done, info

    def _spawn_coins(self):
        """Spawn coins at random unique positions."""
        coins = []
        while len(coins) < self.num_coins:
            pos = (random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1))
            if pos != self.agent_pos and pos not in coins:
                coins.append(pos)
        return coins

    def _get_state(self):
        """Return flattened observation matching state_size."""
        agent_grid = np.zeros((self.grid_size, self.grid_size))
        agent_grid[self.agent_pos] = 1
        
        coins_grid = np.zeros((self.grid_size, self.grid_size))
        for coin in self.coins:
            coins_grid[coin] = 1
            
        coin_coords = []
        for coin in self.coins:
            coin_coords.extend([coin[0] / self.grid_size, coin[1] / self.grid_size])
        
        # Ensure constant size for coin_coords
        while len(coin_coords) < self.num_coins * 2:
            coin_coords.extend([-1.0, -1.0])
            
        return np.concatenate([
            agent_grid.flatten(),
            coins_grid.flatten(),
            np.array(coin_coords[:self.num_coins * 2])
        ]).astype(np.float32)

    def close(self):
        """Clean up resources."""
        if self.render_mode == "human":
            pygame.quit()

    def render(self):
        """Render the environment with additional info."""
        if self.render_mode == "human":
            self.screen.fill((0, 0, 0))
            
            # Draw game elements
            for r in range(self.grid_size):
                for c in range(self.grid_size):
                    rect = pygame.Rect(c * self.cell_size, r * self.cell_size, self.cell_size, self.cell_size)
                    pygame.draw.rect(self.screen, (30, 30, 30), rect, 1) # Grid lines
                    
            # Draw coins
            for coin in self.coins:
                if self.assets_loaded and hasattr(self, 'coin_img'):
                    self.screen.blit(self.coin_img, (coin[1] * self.cell_size + self.cell_size//4, coin[0] * self.cell_size + self.cell_size//4))
                else:
                    pygame.draw.circle(self.screen, (255, 215, 0), 
                                     (coin[1] * self.cell_size + self.cell_size//2, coin[0] * self.cell_size + self.cell_size//2), 
                                     self.cell_size//4)
            
            # Draw agent
            if self.assets_loaded and hasattr(self, 'agent_img'):
                self.screen.blit(self.agent_img, (self.agent_pos[1] * self.cell_size, self.agent_pos[0] * self.cell_size))
            else:
                pygame.draw.rect(self.screen, (0, 255, 0), 
                               (self.agent_pos[1] * self.cell_size + 5, self.agent_pos[0] * self.cell_size + 5, 
                                self.cell_size - 10, self.cell_size - 10))
                
            # Add HUD with performance info
            font = pygame.font.SysFont('Arial', 20)
            coins_text = font.render(f'Coins: {self.coins_collected}/{self.num_coins}', True, (255, 255, 255))
            steps_text = font.render(f'Steps: {self.current_step}/{self.max_steps}', True, (255, 255, 255))
            
            self.screen.blit(coins_text, (10, 10))
            self.screen.blit(steps_text, (10, 40))
            
            pygame.display.flip()
            self.clock.tick(60)