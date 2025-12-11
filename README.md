# Backgammon RL Engine (Gen 5: The King)

A state-of-the-art Reinforcement Learning engine for Backgammon, implementing **Gen 5 Transformer Architecture** with Knowledge Distillation and Match-Aware Value Estimation.

It features a **High-Performance Full-Stack Application** with a React Frontend and FastAPI Backend for interactive play.

## 👑 Key Innovations (Gen 5)

### 1. Transformer Architecture (`src/model_gen5.py`)
- **Self-Attention Mechanism**: Uses a custom **Transformer Encoder Block** combined with wider ResNet layers to capture long-range board dependencies.
- **Match Awareness**: Unlike previous generations, Gen 5 takes the **Match Score** as a direct input, allowing it to dynamically adjust its risk appetite (Optimizing for Gammons when behind, playing safe when ahead).
- **Feature Space**: 200 Input Features (Board, Bar, Off, Turn, Cube, Match Scores).

### 2. Knowledge Distillation & Training (`src/train_gen5.py`)
- **Teacher-Student Learning**: Gen 5 bootstraps its learning by distilling knowledge from the **Gen 4 Champion**.
- **Dual Loss Function**: Minimizes specific game outcome error (MSE) + Teacher Logit matching (KL Divergence).
- **Parallel Evaluation**: Utilizes **Multiprocessing (15 CPU Cores)** to run massive parallel tournaments during training (200 games/batch), accelerating validation by 15x.

### 3. "King of the Hill" Evaluation
- The training process is a continuous tournament.
- **Strict Promotion**: A new model snapshot is only promoted to "Champion" status if it defeats the reigning King in a head-to-head 100-game match with **>55% Win Rate**.
- **Periodic Challenges**: Automatic challenges occur every 2,500 episodes or whenever a new "Best vs Random" record is set.

---

## 🛠️ Usage

### 1. Play vs The AI
Launch the full stack application (Backend + Frontend):
```cmd
start_app.bat
```
This opens:
- **Game UI**: [http://localhost:5173](http://localhost:5173)
- **Backend API**: [http://localhost:8000](http://localhost:8000)

**Features**:
- **Gen 5 AI**: Automatically loads the strongest available model (`best_so_far_gen5.pth`).
- **Drag & Drop**: Intuitive board interface.
- **Visual Aids**: Valid move highlighting, pip counts, and win probability estimation.
- **Aggressive Play**: The AI is hardcoded to maximize equity (Money Play style).

### 2. Train the Gen 5 Model
To start the training loop (requires Python 3.10+):
```cmd
.venv\Scripts\activate
python -m src.train_gen5
```
- **Checkpoints**: Saved to `checkpoints/`.
- **Logs**: Training metrics saved to `logs/training.log`.
- **Performance**: Capable of training ~10,000 episodes/hour on standard hardware thanks to parallel evaluation.

---

## 📂 Project Structure

- **`src/train_gen5.py`**: The Gen 5 Training Script (Distillation, Parallel Eval, KOTH).
- **`src/model_gen5.py`**: The Transformer + ResNet Neural Network definition.
- **`src/game.py`**: The Backgammon Rules Engine (NumPy-based).
- **`src/api.py`**: FastAPI backend serving the game.
- **`src/search.py`**: 2-Ply Search Agent (Expectiminimax) with Alpha-Beta pruning.
- **`frontend/`**: React + TypeScript application.

## 📈 Roadmap
- [x] Gen 1-4: TD-Gammon Replacements (Completed).
- [x] Gen 5: Transformer & Match Awareness (Completed).
- [ ] Gen 6: Zero-Knowledge Self-Play (AlphaZero approach).
