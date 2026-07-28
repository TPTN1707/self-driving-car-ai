from collections import deque
import os
import random
import numpy as np
import torch
from src.config import BATCH_SIZE, GAMMA, LEARNING_RATE, MAX_MEMORY, SENSOR_LENGTH
from src.model import Linear_QNet, QTrainer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0  # Randomness controller
        self.gamma = GAMMA
        self.memory = deque(maxlen=MAX_MEMORY)

        # Input size is 5 (5 sensor readings), output size is 3 (3 actions)
        self.model = Linear_QNet(5, 256, 3).to(device)

        model_path = "./data/checkpoints/model.pth"
        if os.path.exists(model_path):
            print(
                f"\n[INFO] Found saved car model at '{model_path}'. Loading..."
            )
            try:
                self.model.load_state_dict(
                    torch.load(model_path, map_location=device)
                )
                self.n_games = 80  # Skip exploration phase
            except Exception as e:
                print(
                    f"[WARNING] Could not load model: {e}. Starting fresh."
                )

        self.trainer = QTrainer(
            self.model, lr=LEARNING_RATE, gamma=self.gamma
        )

    def get_state(self, game):
        """Extracts the 5 sensor readings and normalizes them between 0.0 and 1.0."""
        # Get raw sensor readings: list of (dist, intersect, ray_end)
        sensor_data = game.car.get_sensor_readings(game.track.walls)

        # Extract distances and normalize by dividing by the max sensor length (150)
        state = []
        for dist, _, _ in sensor_data:
            normalized_dist = dist / SENSOR_LENGTH
            state.append(normalized_dist)

        return np.array(state, dtype=float)

    def remember(self, state, action, reward, next_state, done):
        """Saves a single experience step to replay memory."""
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        """Samples a random batch from memory and trains the neural network (Experience Replay)."""
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        """Trains the neural network on the immediate single step."""
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        """Decides the next action based on epsilon-greedy policy."""
        # Exploration vs Exploitation tradeoff
        self.epsilon = max(80 - self.n_games, 10)
        final_move = [0, 0, 0]

        if random.randint(0, 200) < self.epsilon:
            # Explore: Take a random steering action
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            # Exploit: Query the neural network for prediction
            state0 = torch.tensor(state, dtype=torch.float).to(device)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move