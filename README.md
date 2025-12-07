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

### Training
To start training the agent (Self-Play):
```cmd
.venv\Scripts\activate
python -m src.train
```
This uses 20 CPU cores for parallel environment simulation and the GPU for model updates.
Checkpoints are saved in `checkpoints/`.

### Playing
To play against the trained agent (or a random agent if no model is found):
```cmd
.venv\Scripts\activate
python -m src.play
```

## Project Structure
- `src/game.py`: Core Backgammon logic (Doubling, Rules).
- `src/env.py`: Gymnasium Environment wrapper.
- `src/train.py`: Training script (PPO).
- `src/play.py`: Interactive play script.
- `tests/`: Unit tests.

## Interpreting Training Metrics

When running `src.train`, you will see real-time log outputs. Here is how to read them:

- **`fps`**: Frames Per Second. Measures training speed. Higher is better.
  - On your 20-CPU setup, ~700-1100 FPS is expected.
- **`explained_variance`**: How well the Value function predicts rewards.
  - Range: `0` (Random guessing) to `1` (Perfect prediction).
  - *Good Signal*: It should trend upwards from near 0 to positive values (e.g., 0.2, 0.5, 0.8) as the agent learns the game flow.
- **`value_loss`**: Error in the Value function's prediction.
  - This may increase initially as the agent explores deeper games with higher rewards, but should eventually stabilize.
- **`entropy_loss`**: A measure of randomness in the Policy.
  - Slightly negative values (e.g., `-1.6`).
  - *Stable* is good initially (exploring). *Decreasing* (closer to 0) means the agent is becoming certain about its strategy.
- **`approx_kl`**: Divergence between the old and new policy.
  - Measures how much the agent changed its mind in this update. Large spikes > 0.1 might indicate unstable learning.