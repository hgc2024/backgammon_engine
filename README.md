# Backgammon RL Engine

A Reinforcement Learning engine for Backgammon supporting **Doubling Cube**, **Crawford Rule**, and **Match Play (First to 15)**. Uses PPO (Proximal Policy Optimization) with masking for invalid moves.

## Setup

### Prerequisites
- Python 3.8+
- Windows (recommended for this setup script) or Linux

### Installation
1.  Run the setup script to create a virtual environment and install dependencies:
    ```cmd
    setup_venv.bat
    ```
    This will create a `.venv` directory.

2.  **GPU Support (Optional but Recommended)**:
    By default, `torch` might install the CPU version. To enable GPU support (for your RTX 4050), install the appropriate CUDA version manually inside the venv:
    ```cmd
    .venv\Scripts\activate
    pip install torch --index-url https://download.pytorch.org/whl/cu118
    ```

## Usage

### Training (TD-Gammon)
To start training the agent using the Temporal Difference (TD) learning algorithm (state-of-the-art for Backgammon):
```cmd
.venv\Scripts\activate
python -m src.train_td
```
This script implements:
- **King of the Hill Evaluation**: Every 1,000 episodes, the trainee challenges the current "Champion" (`checkpoints/best_so_far.pth`). If it wins >55% of 100 games, it dethrones the champion.
- **Opponent Pool (League Training)**: 20% of games are played against random past checkpoints to prevent forgetting and ensure robustness.
- **Stability Features**: Epsilon decay, Learning Rate scheduling, and full state checkpointing.

Checkpoints are saved in `checkpoints/`. The best model is always copied to `td_backgammon_best.pth`.

### Web UI
To play against the agent in a graphical interface:
```cmd
.venv\Scripts\activate
streamlit run src/app.py
```
- **Engine Strength**: Choose between 1-Ply (Fast) and 2-Ply (Stronger).
- **Debug Features**: View the engine's win probability confidence in real-time.

## Project Structure
- `src/train_td.py`: **Main Training Script** (TD-Lambda, Self-Play + League).
- `src/game.py`: Core Backgammon logic.
- `src/model.py`: Neural Network Architecture (Tanh output).
- `src/search.py`: Expectiminimax Agent for inference/playing.
- `src/app.py`: Streamlit Web UI.

## Training Metrics
- **Win Rate vs Random**: Sanity check every 250 episodes.
- **Win Rate vs Champion**: The true test of progress (every 1,000 episodes).
- **Loss**: TD-Error (difference between current state value and next state value).
