# Stealth Thief Escape RL

A Reinforcement Learning project where an agent (Thief) learns to navigate a hazardous 10x10 city grid to reach a getaway vehicle while avoiding police officers, hardcoded walls, and a central police station's surveillance zone.

## Key Features
- **Environment**: Custom Gymnasium-compatible environment (`StealthThiefEnv`) with:
    - 10x10 grid with specified coordinate mapping (01 to 100).
    - Randomly spawned Police officers.
    - Central Police Station (44, 45, 54, 55) with a 1-grid "Busted" search radius.
    - Static Wall obstacles.
- **Agent**: Advanced Dueling Double DQN architecture for stable and efficient learning.
- **Persistence**: 
    - Automatic saving and loading of model weights (`models/stealth_thief_latest.pth`).
    - Persistent Replay Buffer (`data/stealth_replay_buffer.pkl`) to preserve experience across training sessions.
- **Continuous Training**: Training script runs in a loop, resetting episodes automatically but allowing manual termination via the 'Q' key in the game window, which triggers a final persistent save.

## Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Assets: Ensure the `assets/` directory contains:
   - `agent.png` (Thief)
   - `police.png` (Police)
   - `wall.png` (Walls)
   - `car.png` (Getaway Vehicle)

## How to Run
### Train the Agent
Watch the agent explore the environment and learn optimal escape routes:
```bash
python scripts/train.py
```
*Press **'Q'** while focused on the game window to save progress and exit.*

### Evaluate the Agent
Watch the best-trained model attempt the escape:
```bash
python scripts/play.py
```

## Disclaimer
This project is developed for game simulation and research purposes. Please refer to [DISCLAIMER.md](DISCLAIMER.md) for more information.
