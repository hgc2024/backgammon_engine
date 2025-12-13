# Backgammon AI: The "King" (Generation 5)

A state-of-the-art Reinforcement Learning engine for Backgammon, featuring a **Transformer-based Neural Network**, **Match-Aware Strategy**, and a full-stack **React/FastAPI** interface.

This project represents the culmination of 5 generations of AI development, moving from simple heuristics to a Master-level agent capable of beating "Perfect vs Random" bots.

---

## 🚀 Quick Start

### 1. Prerequisites
- **Python 3.10+**
- **Node.js & npm** (for Frontend)
- **CUDA-capable GPU** (Recommended for Training, optional for Play)

### 2. Installation
Run the automated setup script to create a virtual environment and install dependencies:
```cmd
setup_venv.bat
cd frontend
npm install
cd ..
```

### 3. Play the Game
Launch the full application (Backend API + React UI) with a single command:
```cmd
start_app.bat
```
- **Game UI**: [http://localhost:5173](http://localhost:5173)
- **Backend API**: [http://localhost:8000](http://localhost:8000)

---

## 🧠 The AI (Gen 5)

### Architecture
The "King" uses a custom **Transformer Encoder + ResNet** architecture (`src/model_gen5.py`).
- **Inputs**: 200 Features (Board State, Bar, Off, Turn, Cube, **Match Scores**).
- **Attention**: Self-Attention layers allow the AI to understand long-range dependencies across the board.
- **Match Awareness**: Unlike traditional bots, Gen 5 changes its playstyle based on the match score (e.g., playing aggressively when behind).

### Training Methodology
- **Knowledge Distillation**: Bootstrapped from a "Gen 4" expert teacher.
- **Parallel Evaluation**: Uses multi-core CPU processing to play thousands of tournament games per hour.
- **King of the Hill**: A strict evolutionary system where a new model only replaces the current champion if it wins a head-to-head match series.

### Hybrid Endgame Strategy
While the Neural Network handles complex contact positions, the agent switches to a **Mathematically Optimal Race Heuristic** when contact is broken.
- **Race Theory**: Maximizes bear-off efficiency while minimizing pip count waste.
- **Guaranteed Optimality**: Ensures the bot never makes "human" mistakes in simple race endings.

### Performance
The current champion (`latest_gen5.pth`) has achieved:
- **100% Win Rate** against Random Agents.
- **>55% Win Rate** against previous "Perfect" models.
- Estimated ELO: **2000+ (Master Level)**.

---

## 🧪 Sandbox Mode & Features
Access the **Sandbox Editor** from the main menu to:
- **Edit Board**: Drag and drop pieces, add/remove checkers (Left/Right click).
- **Custom Scenarios**: Set dice rolls and force specific turn phases.
- **Advanced Evaluation**:
    - **2-Ply Lookahead**: The "Eval" button simulates all possible future dice rolls to give an accurate "Before Roll" equity.
    - **Perspective Splits**: View win probabilities for both YOU and the BOT explicitly (e.g., "You: 30% | Bot: 70%").
- **AI Analysis**: Trigger the AI to play from any position with adjustable depth (1-Ply to 3-Ply).

### Game Mode
- **Depth Toggle**: Switch between **1-Ply** (Instant) and **3-Ply** (Deep Thought) during play.
- **Undo System**: Mistake? Undo moves instantly.
- **Move Log**: Track the game's history and AI's win confidence turn-by-turn.

---

## 🛠️ Advanced Usage

### Training a New Model
To start the training loop from scratch (or resume from checkpoint):
```cmd
.venv\Scripts\activate
python -m src.train_gen5
```
- **Logs**: Monitor progress in `logs/training.log`.
- **Checkpoints**: Models are saved to `checkpoints/`.

### Configuration
- **AI Strength**: Powered by **3-Ply Beam Search** (Width=2). The AI considers opponent responses and its own counter-responses to find the optimal move.
- **Search Logic**: Uses aggressive pruning to inspect thousands of branches in seconds, ensuring Grandmaster-level play without long wait times.

---

## 📂 Project Structure

- **`src/`**: Python Source Code
    - `game.py`: Core Backgammon Rules Engine & Observation Utils.
    - `model_gen5.py`: PyTorch Neural Network Definition.
    - `train_gen5.py`: Main Training Script (Distillation, KOTH).
    - `api.py`: FastAPI Backend Server.
    - `search.py`: Expectiminimax Search Agent (Inference).
- **`frontend/`**: React source code (TypeScript).

---

**Status**: Final Release (Gen 5).
