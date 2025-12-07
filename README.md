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