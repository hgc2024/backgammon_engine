# Backgammon AI: The "King" (Generation 6)

The **Council of Experts** release.

This project implements a state-of-the-art **Agentic AI Workflow** for Backgammon, combining a powerful **Transformer-based Neural Network** (Tactics) with a **Large Language Model** (Strategy).

---

## 🚀 Quick Start

### 1. Prerequisites
- **Python 3.10+**
- **Node.js & npm** (for Frontend)
- **Ollama** (Optional, but required for Gen 6 Agentic Reasoning)
  - Download: [https://ollama.com/](https://ollama.com/)
  - Pull Model: `ollama pull llama3.2`

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

## 🧠 The AI: "The Council" (Gen 6)

The Gen 6 Agent employs a hybrid **"Council of Experts"** architecture to solve the complex game of Backgammon.

### 1. The Tactician (Gen 5 Neural Network)
- **Role**: Calculate ELO-optimal moves and precise win probabilities.
- **Tech**: Transformer Encoder + ResNet trained via Self-Play and Knowledge Distillation.
- **Strength**: 2000+ ELO (Master Level). Handles 95% of standard moves purely on instinct.

### 2. The Strategist (LLM / Mistral / Llama)
- **Role**: Break ties in complex dilemmas where tactical equity is nearly identical.
- **Tech**: Integrating local LLMs (via **Ollama**). default `llama3.2`.
- **Dilemma Protocol**: The Strategist is ONLY summoned when the equity difference between top moves is `< 0.03`. It receives a natural language description of the board and provides purely strategic reasoning (e.g., "Preserve your anchor," "Attack now due to race deficit").

### 3. The Orchestrator
- **Role**: Manages the flow.
  1.  Runs Gen 5 (2-Ply).
  2.  Checks for specific Dilemma criteria.
  3.  If Dilemma -> Consults LLM -> Parses reasoning -> Executes move.
  4.  Updates UI with **"Agentic Reasoning"** (Collapsible panel in Sidebar).

---

## ⚙️ Configuration

### Disabling the LLM
If you do not wish to use Ollama or prefer a lightweight setup, you can disable the Agentic Workflow entirely:
1.  Open `src/api.py`.
2.  Set `ENABLE_GEN6_AGENT = False`.

The system also gracefully degrades: if Ollama is taking too long or is unreachable, the Agent automatically falls back to the standard Gen 5 decision.

---

## 🧪 features

### Sandbox Editor
- **Edit Board**: Drag and drop pieces, add/remove checkers (Left/Right click).
- **Custom Scenarios**: Set dice rolls and force specific turn phases.
- **Evaluation**: Real-time equity analysis with 2-Ply lookahead.

### Game Mode
- **Standard Play**: Default play against the AI.
- **Undo System**: Mistake? Undo moves instantly.
- **Move Log**: Track game history.
- **Collapsible Analysis**: View the AI's "Thoughts" in the sidebar when it faces a tough decision.

---

## 📂 Project Structure

- **`src/`**: Python Source Code
    - `agent_gen6.py`: The "Council" Agentic Workflow & LLM Integration.
    - `game.py`: Core Backgammon Rules Engine.
    - `model_gen5.py`: PyTorch Neural Network Definition.
    - `train_gen5.py`: Main Training Script.
    - `api.py`: FastAPI Backend Server.
    - `search.py`: Expectiminimax Search Algorithms.

---

## 🏆 Performance
- **Win Rate**: ~100% vs Random, >55% vs Previous Generations.
- **Speed**: <500ms per move (Standard), ~2-4s (with LLM reasoning).
