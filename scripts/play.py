import sys
import os
import pygame
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from envs.stealth_thief import StealthThiefEnv
from agents.dqn import DQNAgent
import torch

def load_agent(model_path, state_size, action_size, device="cuda" if torch.cuda.is_available() else "cpu"):
    """Load a trained DQN agent with automatic CUDA detection"""
    agent = DQNAgent(state_size, action_size, device)
    
    if os.path.exists(model_path):
        agent.load(model_path)
    else:
        print(f"Warning: Model not found at {model_path}")
    
    agent.update_target_net()
    agent.q_net.eval()
    agent.target_net.eval()
    
    return agent

def play_stealth():
    env = StealthThiefEnv(render_mode="human")
    agent = load_agent("models/stealth_thief_latest.pth", env.state_size, 4)

    clock = pygame.time.Clock()
    running = True

    while running:
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]
            
        done = False
        while not done and running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            action = agent.act(state, epsilon=0.01) # Use exploitation
            state, reward, done, info = env.step(action)
            
            if done:
                print(f"Outcome: {info['status']}")
                pygame.time.wait(1000) # Pause to see result
            
            clock.tick(10)

    pygame.quit()

if __name__ == "__main__":
    play_stealth()