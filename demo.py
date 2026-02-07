import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from envs.coin_collector import CoinCollectorEnv
import random
import pygame

def run_demo():
    env = CoinCollectorEnv(render_mode="human")
    state = env.reset()
    
    running = True
    clock = pygame.time.Clock()
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        action = random.randint(0, 3)  # Random policy
        next_state, reward, done, info = env.step(action)
        
        print(f"Action: {action}, Reward: {reward}, Collected: {info['collected']}")
        
        if done:
            print("All coins collected! Resetting...")
            state = env.reset()
        
        clock.tick(5)  # Control simulation speed
    
    env.close()

if __name__ == "__main__":
    run_demo()