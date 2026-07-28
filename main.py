import os
import torch

torch.set_num_threads(1)

from src.agent import Agent
from src.environment import CarGameAI
from src.utils.helper import plot

RECORD_FILE_PATH = "./data/checkpoints/record.txt"


def load_historical_record():
    """Loads the absolute highest score record from a physical text file."""
    if os.path.exists(RECORD_FILE_PATH):
        try:
            with open(RECORD_FILE_PATH, "r") as f:
                content = f.read().strip()
                return int(content) if content else 0
        except Exception:
            return 0
    return 0


def save_historical_record(record):
    """Saves the new absolute high-score record to a physical text file."""
    os.makedirs(os.path.dirname(RECORD_FILE_PATH), exist_ok=True)
    with open(RECORD_FILE_PATH, "w") as f:
        f.write(str(record))


def train():
    """Main training loop orchestrating the self-driving car agent and environment."""
    plot_scores = []
    plot_mean_scores = []
    total_score = 0

    agent = Agent()
    game = CarGameAI()

    model_exists = os.path.exists("./data/checkpoints/model.pth")
    record = load_historical_record() if model_exists else 0
    print(f"[INFO] Initialized with Historical Record: {record}")

    while True:
        # 1. Get the current continuous normalized state
        state_old = agent.get_state(game)

        # 2. Query the agent for the steering action
        final_move = agent.get_action(state_old)

        # 3. Step the physics simulation and get feedback
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # 4. Train the neural network on short-term single-step experience
        agent.train_short_memory(
            state_old, final_move, reward, state_new, done
        )

        # 5. Store the transition in replay memory
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # Episode ended: reset environment and update metrics
            game.reset()
            agent.n_games += 1

            # Train long-term memory (Experience Replay)
            agent.train_long_memory()

            # Save the model weights ONLY if the car breaks the absolute historical record
            if score > record:
                record = score
                agent.model.save()
                save_historical_record(record)  # Save record to file
                print(f"--> NEW HISTORICAL RECORD SET: {record}! Saved model.")

            print(
                f"Game: {agent.n_games} | Score: {score} | Record: {record}"
            )

            # Append metrics for real-time visualization
            plot_scores.append(score)
            total_score += score
            mean_score = total_score / len(plot_scores)
            plot_mean_scores.append(mean_score)

            # Dynamically update the training plot PNG
            plot(plot_scores, plot_mean_scores)


if __name__ == "__main__":
    train()