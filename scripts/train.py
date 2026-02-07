import sys
import os
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.stealth_thief import StealthThiefEnv
from agents.dqn import DQNAgent
import torch
import pygame

def train():
    # Paths
    model_dir = "models"
    data_dir = "data"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    
    model_path = os.path.join(model_dir, "stealth_thief_latest.pth")
    buffer_path = os.path.join(data_dir, "stealth_replay_buffer.pkl")

    # Initialize environment
    env = StealthThiefEnv(render_mode="human")
    
    # Initialize agent
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = DQNAgent(
        state_size=env.observation_space,
        action_size=env.action_space,
        device=device
    )
    
    # Load previous state if exists
    if os.path.exists(model_path):
        try:
            agent.load(model_path)
            print("Resuming from previous model weights.")
        except Exception as e:
            print(f"Starting fresh: {e}")
            
    if os.path.exists(buffer_path):
        agent.memory.load(buffer_path)

    # Training parameters
    batch_size = 64
    min_buffer_size = 1000
    
    epsilon = 1.0 if len(agent.memory) < min_buffer_size else 0.3
    epsilon_min = 0.05
    epsilon_decay = 0.999 # Slower decay for more exploration in complex env
    
    print(f"Stealth Thief Training on {device}. Press 'Q' in game window to Save and Quit.")
    
    episode = 0
    running = True
    
    while running:
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]
            
        total_reward = 0
        done = False
        step_count = 0
        
        while not done and running:
            # Check for quit event in Pygame
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
                    running = False
                    print("Quitting and saving progress...")
            
            if not running: break

            # Get action
            action = agent.act(state, epsilon=epsilon)
            
            # Step environment
            next_state, reward, done, info = env.step(action)
            
            # Store experience
            agent.memory.push(state, action, reward, next_state, done)
            
            # Train if enough samples
            if len(agent.memory) > min_buffer_size:
                agent.train(batch_size)
                if epsilon > epsilon_min:
                    epsilon *= epsilon_decay
            
            total_reward += reward
            state = next_state
            step_count += 1
            
            if step_count > 500: break
        
        # Target net update and logging
        if episode % 10 == 0:
            agent.update_target_net()
            print(f"Ep {episode} | Reward: {total_reward:.1f} | Steps: {step_count} | Epsilon: {epsilon:.3f} | Buffer: {len(agent.memory)}")
            agent.save(model_path)
            agent.memory.save(buffer_path)
            
        episode += 1
    
    # Final save
    agent.save(model_path)
    agent.memory.save(buffer_path)
    env.close()
    print("Training finished and saved.")

if __name__ == "__main__":
    train()