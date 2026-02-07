import sys
import os
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.coin_collector import CoinCollectorEnv
from agents.dqn import DQNAgent
import torch

def train():
    # Paths
    model_dir = "models"
    data_dir = "data"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    
    model_path = os.path.join(model_dir, "dqn_latest.pth")
    buffer_path = os.path.join(data_dir, "replay_buffer.pkl")

    # Initialize environment
    env = CoinCollectorEnv(render_mode="human")
    
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
            print("Successfully loaded previous model weights.")
        except Exception as e:
            print(f"Could not load previous model: {e}")
            
    if os.path.exists(buffer_path):
        agent.memory.load(buffer_path)

    # Training parameters
    batch_size = 64
    min_buffer_size = 2000
    max_episodes = 500
    
    epsilon = 1.0 if len(agent.memory) < min_buffer_size else 0.2
    epsilon_min = 0.05
    epsilon_decay = 0.995
    
    print(f"Starting/Resuming training on {device}...")
    
    for episode in range(max_episodes):
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]
            
        total_reward = 0
        done = False
        step_count = 0
        
        while not done:
            # Get action
            action = agent.act(state, epsilon=epsilon)
            
            # Step environment
            next_state, reward, done, info = env.step(action)
            
            # Store experience
            agent.memory.push(state, action, reward, next_state, done)
            
            # Train if enough samples
            if len(agent.memory) > min_buffer_size:
                loss = agent.train(batch_size)
                # Epsilon decay
                if epsilon > epsilon_min:
                    epsilon *= epsilon_decay
            
            total_reward += reward
            state = next_state
            step_count += 1
            
            if step_count > 500: # tighter limit for faster episodes
                break
        
        # Target net update frequency
        if episode % 5 == 0:
            agent.update_target_net()
            
        # Logging and Checkpointing
        if episode % 10 == 0:
            print(f"Ep {episode} | Reward: {total_reward:.1f} | Steps: {step_count} | Epsilon: {epsilon:.3f} | Buffer: {len(agent.memory)}")
            agent.save(model_path)
            # Save periodic checkpoints too
            if episode % 50 == 0:
                agent.save(os.path.join(model_dir, f"dqn_checkpoint_ep{episode}.pth"))
                agent.memory.save(buffer_path)
    
    # Final save
    agent.save(model_path)
    agent.memory.save(buffer_path)
    print("Training completed")

if __name__ == "__main__":
    train()