import sys
import os
import pytest
import numpy as np

# Adjust path to find src
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.game import BackgammonGame

def test_initial_board():
    game = BackgammonGame()
    # Check board setup
    assert game.board[23] == 2
    assert game.board[0] == -2
    assert game.bar == [0, 0]
    assert game.off == [0, 0]

def test_roll_dice():
    game = BackgammonGame()
    d1, d2 = game.roll_dice()
    assert 1 <= d1 <= 6
    assert 1 <= d2 <= 6

def test_get_legal_moves_opening():
    game = BackgammonGame()
    # Mock roll 3, 1
    # Player 0 (White).
    # Moves from 23->...
    # Possible moves:
    # 23->20 (3), 23->22 (1).
    # 12->9 (3), 12->11 (1).
    # 7->4 (3), 7->6 (1).
    # 5->2 (3), 5->4 (1).
    
    # We expect sequences of length 2.
    moves = game.get_legal_moves((3, 1))
    assert len(moves) > 0
    # Check one known move: 7->4, 5->4 (making point 4?)
    # 5->4 is move 1. 7->4 is move 3.
    # Note: moves are ordered? (sequence).
    
    found = False
    for seq in moves:
        if len(seq) == 2:
            s1, s2 = seq[0], seq[1]
            if (s1 == (7, 4) and s2 == (5, 4)) or (s1 == (5, 4) and s2 == (7, 4)):
                found = True
                break
    assert found

def test_bearing_off():
    game = BackgammonGame()
    # Setup board for bearing off
    game.board = np.zeros(24, dtype=int)
    # 2 checkers on point 0 (Player 0 home, dist 0..hmm wait.)
    # Player 0 moves 23 -> 0. Home is 0-5.
    # Point 0 is furthest from Start? No.
    # Let's re-read Game Logic comment:
    # "Player 0 perspective - moving from 23 down to 0"
    # "2 at 23, 5 at 12, 3 at 7, 5 at 5"
    # So P0 moves indices 23 -> 22 -> ... -> 0.
    # Home board is 0, 1, 2, 3, 4, 5.
    # Off is < 0.
    
    game.board[0] = 2 # On point 0 (deepest home point)
    game.board[1] = 0
    # Roll 1
    # Move 0 -> -1 (Off).
    
    moves = game.get_legal_moves((1, 2))
    # Should be able to bear off.
    
    off_moves = []
    for seq in moves:
        for m in seq:
            if m[1] == 'off':
                off_moves.append(m)
    
    assert len(off_moves) > 0
    
def test_doubling_initial():
    game = BackgammonGame()
    # Initial state phase
    assert game.phase.name == "DECIDE_CUBE_OR_ROLL"
    # Should be able to double (if not crawford, which is initial)
    assert game._can_double(0)
