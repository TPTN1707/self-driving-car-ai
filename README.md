# Self-Driving Car AI - Reinforcement Learning (DQN)

An advanced 2D autonomous driving simulator that teaches an AI car to steer and navigate a custom racetrack using Deep Q-Learning (DQN). Built with Python, PyTorch, Pygame, and Matplotlib.

## Key Advancements (Over Snake AI)

This project introduces several complex features representing real-world autonomous vehicle mechanics:
- **Continuous Normalized States:** Instead of simple binary inputs, the AI receives 5 continuous distance values, normalized between `0.0` (critical danger) and `1.0` (clear road) for highly stable neural network training.
- **Ray-Casting Distance Sensors:** Simulates LIDAR-like proximity sensors by casting 5 laser rays (`-90°`, `-45`, `0°`, `45°`, `90°` relative to heading) and calculating real-time line-segment intersections with the track walls using vector cross products.
- **Rotational Car Physics:** Implements continuous trigonometric heading angle rotation (`math.cos` and `math.sin`) to calculate realistic 360-degree car steering and movement.
- **Persistent Record Saving:** Saves the absolute historical high-score to `record.txt`. The model is *only* overwritten when the agent breaks this historical record, protecting your best-trained models from accidental overwrites during subsequent runs.
- **Minimum Epsilon Safety Net:** Prevents policy collapse and local minima traps by enforcing a minimum 5% exploration rate (`epsilon = max(..., 10)`), allowing the agent to self-correct and recover from bad training states.

## Project Directory Structure

```text
self-driving-car-ai/
├── data/                       # Output checkpoints and charts
│   ├── checkpoints/            # Saved model weights (.pth) & high score (record.txt)
│   └── plots/                  # Auto-saved dynamic performance graphs (progress.png)
├── src/                        # Main source code
│   ├── __init__.py
│   ├── config.py               # Physics, colors, and RL hyperparameters
│   ├── track.py                # Generates the closed-loop racetrack walls
│   ├── environment.py          # Car physics, ray-casting math, and collision checks
│   ├── agent.py                # RL Agent (state normalization, epsilon-greedy)
│   ├── model.py                # Deep Q-Network and Q-Trainer (PyTorch)
│   └── utils/
│       ├── __init__.py
│       └── helper.py           # Real-time Matplotlib plotting helper (starts from Game 1)
├── main.py                     # Training orchestrator and entry point
└── requirements.txt            # Dependency list
```

## Requirements

- Python 3.11 (Recommended for stability with PyTorch and Pygame)
- Windows OS (Required for standard `pygame` and interactive `matplotlib` threads)

## Installation & Execution

1. **Install dependencies:**
   Using `uv` (recommended):
   ```bash
   uv pip install -r requirements.txt
   ```
   Or standard pip:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the simulator:**
   ```bash
   uv run main.py
   ```
   *(Or `python main.py`)*

## Neural Network & Reward System

- **Neural Network:** A 3-layer feedforward network with 5 inputs (normalized sensors), 256 hidden neurons, and 3 outputs (steering commands: [Straight, Turn Right, Turn Left]).
- **Reward Function:**
  - Survive one frame: `+0.1` (Encourages driving continuously).
  - Collision with track walls: `-10` (Heavily penalizes crashes).
  - No timeout limit (Allows the fully-trained agent to drive indefinitely).