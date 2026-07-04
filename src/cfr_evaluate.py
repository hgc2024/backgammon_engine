import torch
import numpy as np
import multiprocessing as mp
from tqdm import tqdm
from typing import Dict, List, Tuple
import os
from game import BackgammonGame, GamePhase, get_obs_from_state
from cfr import CFRTrainer
import random
import time

class CFREvaluator:
    """
    Evaluation utilities for CFR-trained backgammon agents.
    """
    
    def __init__(self, cfr_trainer: CFRTrainer, device: torch.device = None):
        self.cfr = cfr_trainer
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.game = BackgammonGame()
    
    def evaluate_vs_random(self, n_games: int = 100) -> Dict[str, float]:
        """Evaluate CFR agent against random opponent"""
        wins = 0
        total_moves = 0
        good_moves = 0

        with tqdm(range(n_games), desc="CFR vs Random", unit="game") as pbar:
            for game_idx in pbar:
                self.game.reset_match()
                game_over = False

                while not game_over:
                    if self.game.phase == GamePhase.DECIDE_CUBE_OR_ROLL:
                        self.game.step(0)  # Always roll
                    elif self.game.phase == GamePhase.RESPOND_TO_DOUBLE:
                        self.game.step(0)  # Always take
                    elif self.game.phase == GamePhase.DECIDE_MOVE:
                        moves = self.game.legal_moves
                        if not moves:
                            self.game.turn = 1 - self.game.turn
                            self.game.phase = GamePhase.DECIDE_CUBE_OR_ROLL
                            continue

                        if self.game.turn == 0:  # CFR agent
                            action = self._get_cfr_action()
                            total_moves += 1

                            # Evaluate move quality
                            if self._is_good_move(action):
                                good_moves += 1
                        else:  # Random opponent
                            action = random.randint(0, len(moves) - 1)

                        self.game.step(action)

                    elif self.game.phase == GamePhase.GAME_OVER:
                        if self.game.score[0] > self.game.score[1]:
                            wins += 1
                        game_over = True

                if (game_idx + 1) % max(1, n_games // 10) == 0:
                    pbar.set_postfix({
                        "win_rate": f"{wins / (game_idx + 1):.3f}",
                        "move_q": f"{(good_moves / max(total_moves, 1)):.3f}",
                    })
        
        win_rate = wins / n_games
        move_quality = good_moves / max(total_moves, 1)

        print(
            f"[Eval Summary] CFR vs Random | games={n_games} | "
            f"win_rate={win_rate:.3f} | move_quality={move_quality:.3f}"
        )
        
        return {
            'win_rate': win_rate,
            'move_quality': move_quality,
            'total_games': n_games
        }
    
    def evaluate_vs_td(self, td_model_path: str, n_games: int = 100) -> Dict[str, float]:
        """Evaluate CFR agent against TD-trained opponent"""
        from model import BackgammonValueNet
        
        # Load TD model
        td_net = BackgammonValueNet().to(self.device)
        if os.path.exists(td_model_path):
            checkpoint = torch.load(td_model_path, map_location=self.device)
            td_net.load_state_dict(checkpoint['model_state_dict'])
        td_net.eval()
        
        wins = 0
        total_moves = 0
        cfr_moves = 0
        td_moves = 0
        
        for game_idx in tqdm(range(n_games), desc="CFR vs TD"):
            self.game.reset_match()
            game_over = False
            
            while not game_over:
                if self.game.phase == GamePhase.DECIDE_CUBE_OR_ROLL:
                    self.game.step(0)
                elif self.game.phase == GamePhase.RESPOND_TO_DOUBLE:
                    self.game.step(0)
                elif self.game.phase == GamePhase.DECIDE_MOVE:
                    moves = self.game.legal_moves
                    if not moves:
                        self.game.turn = 1 - self.game.turn
                        self.game.phase = GamePhase.DECIDE_CUBE_OR_ROLL
                        continue
                    
                    if self.game.turn == 0:  # CFR agent
                        action = self._get_cfr_action()
                        cfr_moves += 1
                    else:  # TD agent
                        action = self._get_td_action(td_net, moves)
                        td_moves += 1
                    
                    total_moves += 1
                    self.game.step(action)
                
                elif self.game.phase == GamePhase.GAME_OVER:
                    if self.game.score[0] > self.game.score[1]:
                        wins += 1
                    game_over = True
        
        win_rate = wins / n_games
        
        return {
            'win_rate': win_rate,
            'cfr_moves': cfr_moves,
            'td_moves': td_moves,
            'total_games': n_games
        }
    
    def evaluate_exploitability(self, n_samples: int = 1000) -> Dict[str, float]:
        """
        Evaluate exploitability of the current strategy.
        Lower exploitability means closer to Nash equilibrium.
        """
        total_exploitability = 0.0
        
        with tqdm(range(n_samples), desc="Computing Exploitability", unit="sample") as pbar:
            for sample_idx in pbar:
            # Sample a random game state
                self._sample_random_state()
            
            # Get CFR strategy
                cfr_strategy = self._get_current_strategy()
            
            # Compute best response value
                best_response_value = self._compute_best_response_value()
            
            # CFR value
                cfr_value = self._compute_cfr_value(cfr_strategy)
            
            # Exploitability is the gap
                exploitability = best_response_value - cfr_value
                total_exploitability += exploitability

                if (sample_idx + 1) % max(1, n_samples // 10) == 0:
                    pbar.set_postfix({
                        "avg_exploitability": f"{(total_exploitability / (sample_idx + 1)):.4f}"
                    })
        
        avg_exploitability = total_exploitability / n_samples

        print(
            f"[Eval Summary] Exploitability | samples={n_samples} | "
            f"avg={avg_exploitability:.4f}"
        )
        
        return {
            'exploitability': avg_exploitability,
            'samples': n_samples
        }
    
    def evaluate_convergence(self, window_size: int = 100) -> Dict[str, float]:
        """Evaluate convergence metrics"""
        if len(self.cfr.convergence_history) < window_size:
            return {'convergence_rate': 0.0, 'stability': 0.0}
        
        recent_history = self.cfr.convergence_history[-window_size:]
        
        # Compute regret reduction rate
        regrets = [h['avg_regret'] for h in recent_history]
        if len(regrets) > 1:
            convergence_rate = (regrets[0] - regrets[-1]) / max(regrets[0], 1e-6)
        else:
            convergence_rate = 0.0
        
        # Compute strategy stability
        stabilities = [h['strategy_stability'] for h in recent_history]
        avg_stability = np.mean(stabilities)
        
        return {
            'convergence_rate': convergence_rate,
            'stability': avg_stability,
            'current_regret': regrets[-1] if regrets else 0.0
        }
    
    def _get_cfr_action(self) -> int:
        """Get action from CFR strategy"""
        board, bar, off = self.game.board, self.game.bar, self.game.off
        player = self.game.turn
        dice = self.game.dice if hasattr(self.game, 'dice') else (0, 0)
        
        strategy = self.cfr.get_strategy(board, bar, off, player, dice)
        
        # Sample from strategy
        return np.random.choice(len(strategy), p=strategy)
    
    def _get_td_action(self, td_net, moves: List) -> int:
        """Get action from TD network"""
        boards = []
        for seq in moves:
            b, ba, o = self.game.get_afterstate(seq)
            obs = get_obs_from_state(b, ba, o, 1 - self.game.turn, 
                                    self.game.score, self.game.cube_value, 1)
            boards.append(obs)
        
        if boards:
            t = torch.tensor(np.array(boards), dtype=torch.float32).to(self.device)
            with torch.no_grad():
                logits, _ = td_net(t)
                probs = torch.softmax(logits, dim=1)
                weights = torch.tensor([-3.0, -2.0, -1.0, 1.0, 2.0, 3.0], device=self.device)
                vals = torch.sum(probs * weights, dim=1)
            return torch.argmin(vals).item()
        else:
            return 0
    
    def _is_good_move(self, action: int) -> bool:
        """Simple heuristic to evaluate move quality"""
        # This is a simplified evaluation - in practice you'd use more sophisticated metrics
        moves = self.game.legal_moves
        if len(moves) <= 1:
            return True
        
        # Check if move captures opponent pieces or advances own pieces
        saved_state = self.game.save_state()
        self.game.step(action)
        
        # Simple heuristic: check if we improved our position
        p0_pip_before, p1_pip_before = self._get_pip_counts_from_state(saved_state)
        p0_pip_after, p1_pip_after = self.game.get_pip_counts()
        
        self.game.restore_state(saved_state)
        
        if self.game.turn == 0:
            return p0_pip_after < p0_pip_before
        else:
            return p1_pip_after < p1_pip_before
    
    def _get_pip_counts_from_state(self, state) -> Tuple[int, int]:
        """Extract pip counts from saved state"""
        # This is simplified - would need proper state extraction
        return 150, 150  # Placeholder
    
    def _sample_random_state(self):
        """Sample a random game state for evaluation"""
        # Play random moves to get to a random state
        self.game.reset_match()
        for _ in range(random.randint(10, 30)):
            if self.game.phase == GamePhase.DECIDE_MOVE:
                moves = self.game.legal_moves
                if moves:
                    action = random.randint(0, len(moves) - 1)
                    self.game.step(action)
                else:
                    break
            elif self.game.phase == GamePhase.DECIDE_CUBE_OR_ROLL:
                self.game.step(0)
            elif self.game.phase == GamePhase.RESPOND_TO_DOUBLE:
                self.game.step(0)
            elif self.game.phase == GamePhase.GAME_OVER:
                break
    
    def _get_current_strategy(self) -> np.ndarray:
        """Get current CFR strategy"""
        board, bar, off = self.game.board, self.game.bar, self.game.off
        player = self.game.turn
        dice = self.game.dice if hasattr(self.game, 'dice') else (0, 0)
        
        return self.cfr.get_strategy(board, bar, off, player, dice)
    
    def _compute_best_response_value(self) -> float:
        """Compute best response value against current strategy"""
        # This is simplified - would need full best response computation
        return 0.0  # Placeholder
    
    def _compute_cfr_value(self, strategy: np.ndarray) -> float:
        """Compute CFR value for current strategy"""
        # This is simplified - would need proper value computation
        return 0.0  # Placeholder

def parallel_evaluate_cfr(cfr_trainer: CFRTrainer, n_games: int = 100, 
                         num_workers: int = None) -> Dict[str, float]:
    """Parallel evaluation of CFR agent"""
    if num_workers is None:
        num_workers = max(1, mp.cpu_count() - 2)
    
    def evaluate_worker(args):
        cfr_state, games, seed = args
        random.seed(seed)
        np.random.seed(seed)
        
        # Create evaluator with CFR state
        evaluator = CFREvaluator(cfr_state)
        return evaluator.evaluate_vs_random(games)
    
    # Prepare CFR state for workers
    cfr_state = {
        'infosets': cfr_trainer.infosets,
        'iteration': cfr_trainer.iteration
    }
    
    # Split games among workers
    games_per_worker = n_games // num_workers
    tasks = []
    for i in range(num_workers):
        games = games_per_worker + (1 if i < n_games % num_workers else 0)
        tasks.append((cfr_state, games, int(time.time()) + i))
    
    # Run evaluation
    with mp.Pool(processes=num_workers) as pool:
        results = list(tqdm(pool.imap_unordered(evaluate_worker, tasks), 
                          total=num_workers, desc="Parallel Evaluation"))
    
    # Aggregate results
    total_wins = sum(r['win_rate'] * r['total_games'] for r in results)
    total_games = sum(r['total_games'] for r in results)
    avg_move_quality = np.mean([r['move_quality'] for r in results])
    
    return {
        'win_rate': total_wins / total_games,
        'move_quality': avg_move_quality,
        'total_games': total_games
    }