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

### Performance
The current champion (`latest_gen5.pth`) has achieved:
- **100% Win Rate** against Random Agents.
- **>55% Win Rate** against previous "Perfect" models.
- Estimated ELO: **2000+ (Master Level)**.

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
- **AI Strength**: Hardcoded to **2-Ply Search** (Strongest) for the best user experience.
- **Playstyle**: Hardcoded to **Aggressive/Match-Optimized**.

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
