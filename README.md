# Backgammon RL Engine (Gen 3)

A high-performance Reinforcement Learning engine for Backgammon, implementing **TD-Gammon** with modern training enhancements (League Training, Curriculum Learning, and Quality Control).

It features a **Full-Stack Application** with a React Frontend and FastAPI Backend for interactive play.

## 🚀 Key Methodologies (Gen 3)

### 1. Model Architecture
- **ResNet-style 1D CNN**: Processes the board state (encoded as 198 features) relative to the active player.
- **Output**: Scalar `tanh` value (`-1` to `1`) representing the **Equity** (Expected Win Value). `1.0` = Guaranteed Win, `-1.0` = Guaranteed Loss.

### 2. Training Loop (`src/train_td.py`)
- **Outcome-Based Learning**: The model learns to predict the final game result (Monte Carlo evaluation), minimizing the MSE between its prediction `V(s)` and the actual outcome.
- **Self-Play with Pool**:
    - **Self-Play**: 75% of games are played against itself.
    - **Opponent Pool**: 25% of games are played against **Historical Checkpoints** (including the **Gen 2 Champion**) to prevent regression and cyclic learning strategies.
- **Curriculum Learning (Endgame)**: 
    - **30%** of games start in a **Late-Game Scenario**:
        - **Race**: Checkers clustered in range 0-8.
        - **Bear-Off**: Checkers all in Home Board (0-5), forcing the AI to master the guaranteed win/safe play.
- **Quality Control (QC)**:
    - Before saving a checkpoint, the model must achieve **>40% Win Rate** against a static **"Gatekeeper"** (The Gen 2 Champion). This filters out collapsed models.
- **King of the Hill**:
    - Every 1,000 episodes, the trainee challenges the reigning **Champion**.
    - If it achieves **>55% Win Rate** (over 50-100 games), it typically dethrones the champion and becomes the new standard (`checkpoints/best_so_far.pth`).

## 🛠️ Usage

### 1. Play the Game
To launch the full stack (Backgammon UI + Engine API):
```cmd
start_app.bat
```
This opens:
- **Backend**: FastAPI server at `http://localhost:8000`
- **Frontend**: React App at `http://localhost:5173`

**Features**:
- **Drag & Drop** Interface with Valid Move Highlighting (Green/Yellow).
- **Bear Off**: Drag checkers to the "Off" zone (Right side) to score when home is full.
- **Bar Re-entry**: Drag checkers from the "Bar" (Center) to re-enter when hit.
- **AI Strength Toggle**: 1-Ply (Fast) vs 2-Ply (Strong).
- **Move Log**: Detailed history with AI Win Confidence (e.g., `Win Est: 58.2%`).
- **Auto-AI**: CPU plays automatically on its turn.

### 2. Train the Model
To start the Gen 3 training loop:
```cmd
.venv\Scripts\activate
python -m src.train_td
```
- **Checkpoints**: Saved to `checkpoints/`.
- **Best Model**: Continuously updated at `checkpoints/best_so_far.pth`.

## 📂 Project Structure

- **`src/train_td.py`**: The core Training Script. Implements the League, Curriculum, and QC logic.
- **`src/game.py`**: The Backgammon Rules Engine. Fast, NumPy-based implementation supporting Splits, Hits, Bearing Off, and Dice Logic.
- **`src/api.py`**: FastAPI backend serving the game state and AI predictions.
- **`src/search.py`**: The Inference Agent (Expectiminimax).
- **`frontend/`**: React + TypeScript application.

## 📈 Gen 3 Goals
- Solve the "Endgame Randomness" issue by forcing high-frequency exposure to Bear-Off states.
- Stabilize learning using the "Gatekeeper" to ensure every saved version is at least competitive with Gen 2.
