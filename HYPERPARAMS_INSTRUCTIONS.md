# How to Invoke Hyperparameters in the CFR Framework

This guide explains how to pass hyperparameters to the CFR training and evaluation scripts.

## 1. Training (`src/train_cfr.py`)

The training script uses `argparse` to accept command-line arguments. To add a new hyperparameter:

### Step 1: Add the argument to the parser
Edit `src/train_cfr.py` and add an `add_argument` call before `parser.parse_args()`.

Example:
```python
parser.add_argument(
    "--lr_policy",
    type=float,
    default=1e-3,
    help="Learning rate for the policy network (Adam)",
)
```

### Step 2: Retrieve the argument in `main()`
After `args = parser.parse_args()`, access the value via `args.lr_policy`.

### Step 3: Pass the argument to the CFRTrainer constructor
When creating the trainer, pass the hyperparameter:
```python
cfr = CFRTrainer(
    device=device,
    lr_policy=args.lr_policy,
    lr_value=args.lr_value,
    gamma=args.gamma,
    # ... other arguments ...
)
```

### Step 4: Store and use the hyperparameter inside `CFRTrainer`
In `src/cfr.py`, modify the `__init__` method to accept and store the hyperparameter:
```python
def __init__(self, device, lr_policy=1e-3, lr_value=1e-3, gamma=0.99, ...):
    self.lr_policy = lr_policy
    self.lr_value = lr_value
    self.gamma = gamma
    # ... etc.
```
Then use the stored attribute where needed, e.g., in the optimizer:
```python
self.policy_optimizer = optim.Adam(
    self.policy_net.parameters(),
    lr=self.lr_policy,
    betas=(0.9, 0.999),
    eps=1e-8,
)
```

## 2. Evaluation (`src/cfr_evaluate.py`)

The evaluation script follows the same pattern.

### Step 1: Add the argument to the parser
Edit `src/cfr_evaluate.py` and add an `add_argument` call.

Example for a temperature parameter:
```python
parser.add_argument(
    "--eval_temp",
    type=float,
    default=1.0,
    help="Temperature for softmax when sampling actions during evaluation",
)
```

### Step 2: Retrieve the argument in `main()`
Access via `args.eval_temp`.

### Step 3: Pass the argument to the `CFREvaluator` constructor
```python
evaluator = CFREvaluator(
    cfr=cfr,
    device=args.device,
    n_games=args.eval_games,
    temp=args.eval_temp,   # <-- NEW
)
```

### Step 4: Store and use the hyperparameter inside `CFREvaluator`
In `src/cfr_evaluate.py`, modify the `__init__` method:
```python
def __init__(self, cfr_trainer: CFRTrainer, device: torch.device = None, temp: float = 1.0):
    self.cfr = cfr_trainer
    self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.game = BackgammonGame()
    self.temp = temp   # <-- store it
```
Then use `self.temp` where needed, e.g., in action selection:
```python
probs = torch.softmax(logits / self.temp, dim=-1)
```

## 3. Example Command Lines

### Training with custom learning rates and discount factor
```bash
python src/train_cfr.py \
    --lr_policy 0.0005 \
    --lr_value 0.001 \
    --gamma 0.95 \
    --iterations 50000 \
    --device auto
```

### Evaluation with custom temperature and number of games
```bash
python src/cfr_evaluate.py \
    --eval_temp 0.8 \
    --eval_games 200 \
    --device auto
```

## 4. Tips

- Keep default values sensible; change them only when you have a specific reason.
- If you want a hyperparameter to change over time (e.g., learning rate decay), modify the training loop after retrieving the trainer's attribute.
- Always run a quick sanity check with the default values before launching extensive experiments.

---