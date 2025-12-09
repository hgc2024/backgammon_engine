import React, { useEffect, useState } from 'react';
import axios from 'axios';
import { Point } from './Point';

// Types
interface GameState {
    board: number[];
    bar: number[];
    off: number[];
    turn: number;
    dice: number[];
    legal_moves: any[];
    phase: string;
    score: number[];
}

const API_URL = 'http://localhost:8000';

const Board: React.FC = () => {
    const [gameState, setGameState] = useState<GameState | null>(null);
    const [legalMoves, setLegalMoves] = useState<any[]>([]); // List of [from, to] (to can be 'off')
    const [message, setMessage] = useState<string>("");

    const fetchState = async () => {
        try {
            const res = await axios.get(`${API_URL}/gamestate`);
            setGameState(res.data);

            // Also fetch moves
            const movesRes = await axios.get(`${API_URL}/moves`);
            setLegalMoves(movesRes.data);
        } catch (e) {
            console.error(e);
            setMessage("Failed to connect to backend");
        }
    };

    useEffect(() => {
        fetchState();
        const interval = setInterval(fetchState, 1000); // Poll every second for AI moves
        return () => clearInterval(interval);
    }, []);

    const handleRoll = async () => {
        await axios.post(`${API_URL}/roll`);
        fetchState();
    };

    const handleUndo = async () => {
        await axios.post(`${API_URL}/undo`);
        fetchState();
    };

    const handleMove = async (fromIdx: number, toIdx: number) => {
        if (!gameState) return;

        console.log(`Attempt Move ${fromIdx} -> ${toIdx}`);
        try {
            await axios.post(`${API_URL}/step`, {
                move: [fromIdx, toIdx]
            });
            fetchState();
        } catch (e) {
            console.error("Move Failed", e);
            setMessage("Invalid Move");
        }
    };

    // Check if point is valid dest for start (if dragging) - NOT IMPLEMENTED (Need start point)
    // Highlighting Logic: Point asks "Am I a legal destination for *any* active move?"
    // OR: "Can *any* checker move here?"
    // This is useful for "Drop Target" highlighting.
    const isLegalDestination = (idx: number) => {
        return legalMoves.some(m => m[1] === idx);
    };

    if (!gameState) return <div>Loading... {message}</div>;

    return (
        <div style={{ padding: '20px', maxWidth: '800px', margin: '0 auto' }}>
            {/* HUD */}
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '10px' }}>
                <div>Score: You {gameState.score[0]} - CPU {gameState.score[1]}</div>
                <div>Turn: {gameState.turn === 0 ? "You (White)" : "CPU (Red)"}</div>
                <div>Dice: {JSON.stringify(gameState.dice)}</div>
            </div>

            {message && <div style={{ color: 'red' }}>{message}</div>}

            {/* Controls */}
            <div style={{ marginBottom: '20px', display: 'flex', gap: '10px' }}>
                <button onClick={handleRoll} disabled={gameState.phase !== "DECIDE_CUBE_OR_ROLL"}>Roll Dice</button>
                <button onClick={handleUndo}>Undo</button>
                <button onClick={() => axios.post(`${API_URL}/start`).then(fetchState)}>New Game</button>
                <button onClick={() => axios.post(`${API_URL}/ai-move`).then(fetchState)}>AI Move</button>
            </div>

            {/* Board Visual */}
            <div style={{ backgroundColor: '#f0d9b5', padding: '20px', borderRadius: '5px', border: '5px solid #6b4c35' }}>

                {/* Top Row (12-23) */}
                <div style={{ display: 'flex', height: '150px', borderBottom: '2px solid #6b4c35' }}>
                    {Array.from({ length: 12 }, (_, i) => 12 + i).map(idx => (
                        <Point
                            key={idx}
                            index={idx}
                            checkers={gameState.board[idx]}
                            onDropChecker={handleMove}
                            canMoveTo={isLegalDestination(idx)}
                        />
                    ))}
                </div>

                {/* Bar Area (Middle) */}
                <div style={{ height: '20px', backgroundColor: '#6b4c35', color: 'white', textAlign: 'center' }}>
                    BAR: White({gameState.bar[0]}) | Red({gameState.bar[1]})
                </div>

                {/* Bot Row (11-0) -- Reversed order for visual correctness (11 left, 0 right) */}
                <div style={{ display: 'flex', height: '150px', borderTop: '2px solid #6b4c35' }}>
                    {Array.from({ length: 12 }, (_, i) => 11 - i).map(idx => (
                        <Point
                            key={idx}
                            index={idx}
                            checkers={gameState.board[idx]}
                            onDropChecker={handleMove}
                            canMoveTo={isLegalDestination(idx)}
                        />
                    ))}
                </div>
            </div>

            {/* Off Tray */}
            <div style={{ marginTop: '10px', textAlign: 'right' }}>
                OFF: White({gameState.off[0]}) | Red({gameState.off[1]})
            </div>

        </div>
    );
};

export default Board;
