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
    history: string[];
}

const API_URL = 'http://localhost:8000';

const Board: React.FC = () => {
    const [gameState, setGameState] = useState<GameState | null>(null);
    const [legalMoves, setLegalMoves] = useState<any[]>([]);
    const [message, setMessage] = useState<string>("");

    // UI State
    const [aiDepth, setAiDepth] = useState<number>(1);
    const [startPlayer, setStartPlayer] = useState<number>(-1);

    const fetchState = async () => {
        try {
            const res = await axios.get(`${API_URL}/gamestate`);
            setGameState(res.data);
            const movesRes = await axios.get(`${API_URL}/moves`);
            setLegalMoves(movesRes.data);
        } catch (e) {
            console.error(e);
            setMessage("Failed to connect to backend");
        }
    };

    useEffect(() => {
        fetchState();
        const interval = setInterval(fetchState, 1000);
        return () => clearInterval(interval);
    }, []);

    const handleRoll = async () => {
        await axios.post(`${API_URL}/roll`);
        fetchState();
    };

    const handleAIMove = async () => {
        await axios.post(`${API_URL}/ai-move`, { depth: aiDepth });
        fetchState();
    };

    // Auto-AI Hook
    useEffect(() => {
        if (!gameState) return;
        const isCpuTurn = gameState.turn === 1;
        const isGameOver = gameState.phase === "GAME_OVER";

        if (isCpuTurn && !isGameOver) {
            // Add delay for UX so it doesn't feel instant/broken
            const timer = setTimeout(() => {
                handleAIMove();
            }, 1000);
            return () => clearTimeout(timer);
        }
    }, [gameState, aiDepth]); // Re-run if gamestate updates or depth changes

    const handleUndo = async () => {
        await axios.post(`${API_URL}/undo`);
        fetchState();
    };

    const handleNewGame = async () => {
        await axios.post(`${API_URL}/start`, { first_player: startPlayer });
        fetchState();
    };

    const handlePass = async () => {
        try {
            await axios.post(`${API_URL}/pass`);
            fetchState();
        } catch (e: any) {
            console.error("Pass Failed", e);
            if (e.response && e.response.data && e.response.data.error) {
                setMessage(e.response.data.error);
            }
        }
    };

    const handleMove = async (fromIdx: number, toIdx: number) => {
        if (!gameState) return;
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

    const isLegalDestination = (idx: number) => {
        return legalMoves.some(m => m[1] === idx);
    };

    if (!gameState) return <div>Loading... {message}</div>;

    return (
        <div style={{ display: 'flex', flexDirection: 'row', gap: '20px', padding: '20px', maxWidth: '1400px', margin: '0 auto' }}>

            {/* LEFT: Game Board */}
            <div style={{ flex: 3 }}>
                {/* HUD */}
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '10px', fontSize: '1.2em', fontWeight: 'bold' }}>
                    <div>Score: You {gameState.score[0]} - CPU {gameState.score[1]}</div>
                    <div style={{ color: gameState.turn === 0 ? 'black' : 'red' }}>
                        Turn: {gameState.turn === 0 ? "You (White)" : "CPU (Red)"}
                    </div>
                    <div>Dice: {JSON.stringify(gameState.dice)}</div>
                </div>

                {message && <div style={{ color: 'red', fontWeight: 'bold' }}>{message}</div>}

                {/* Controls Area */}
                <div style={{ marginBottom: '20px', padding: '15px', backgroundColor: '#eee', borderRadius: '8px', display: 'flex', gap: '15px', alignItems: 'center', flexWrap: 'wrap', border: '1px solid #ccc' }}>

                    {/* Action Buttons */}
                    <button onClick={handleRoll} disabled={gameState.phase !== "DECIDE_CUBE_OR_ROLL" || gameState.turn === 1} style={{ padding: '10px 20px', fontSize: '1em', cursor: 'pointer', fontWeight: 'bold' }}>Roll Dice</button>
                    <button onClick={handleUndo} disabled={gameState.turn === 1} style={{ padding: '10px 20px', fontSize: '1em', cursor: 'pointer' }}>Undo</button>
                    <button onClick={handlePass} disabled={gameState.turn === 1} style={{ padding: '10px 20px', fontSize: '1em', cursor: 'pointer', color: 'darkblue', fontWeight: 'bold', border: '1px solid blue' }}>End Turn (Pass)</button>

                    {/* AI Controls - Just Depth now */}
                    <div style={{ display: 'flex', alignItems: 'center', gap: '5px', borderLeft: '1px solid #999', paddingLeft: '15px' }}>
                        {/* Button hidden/debug only now */}
                        {/* <button onClick={handleAIMove} style={{ padding: '10px 20px', cursor: 'pointer', backgroundColor: '#ddd', fontWeight: 'bold' }}>AI Move</button> */}
                        <label>AI Depth:</label>
                        <select value={aiDepth} onChange={(e) => setAiDepth(Number(e.target.value))} style={{ padding: '10px' }}>
                            <option value={1}>1-Ply (Fast)</option>
                            <option value={2}>2-Ply (Strong)</option>
                        </select>
                    </div>

                    {/* New Game Controls */}
                    <div style={{ borderLeft: '1px solid #999', paddingLeft: '15px', display: 'flex', gap: '5px', alignItems: 'center' }}>
                        <select value={startPlayer} onChange={(e) => setStartPlayer(Number(e.target.value))} style={{ padding: '10px' }}>
                            <option value={-1}>Random Start</option>
                            <option value={0}>You Start</option>
                            <option value={1}>CPU Starts</option>
                        </select>
                        <button onClick={handleNewGame} style={{ padding: '10px 15px', backgroundColor: '#f99', color: 'white', fontWeight: 'bold', border: 'none', borderRadius: '4px', cursor: 'pointer' }}>New Game</button>
                    </div>
                </div>

                {/* Board Visual */}
                <div style={{ backgroundColor: '#f0d9b5', padding: '20px', borderRadius: '5px', border: '10px solid #6b4c35', minHeight: '520px', boxShadow: '0 4px 8px rgba(0,0,0,0.2)' }}>

                    {/* Top Row (12-23) */}
                    <div style={{ display: 'flex', height: '240px', borderBottom: '2px solid #6b4c35' }}>
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
                    <div style={{ height: '40px', backgroundColor: '#6b4c35', color: 'white', textAlign: 'center', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '1.2em', fontWeight: 'bold' }}>
                        BAR -- White: {gameState.bar[0]} | Red: {gameState.bar[1]}
                    </div>

                    {/* Bot Row (11-0) */}
                    <div style={{ display: 'flex', height: '240px', borderTop: '2px solid #6b4c35' }}>
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

                {/* Off Tray - Prominent Display */}
                <div style={{ marginTop: '15px', padding: '15px', backgroundColor: '#e0e0e0', borderRadius: '8px', textAlign: 'center', fontSize: '1.4em', fontWeight: 'bold', border: '2px solid #bbb' }}>
                    OFF BOARD: <span style={{ color: 'black' }}>White ({gameState.off[0]})</span> | <span style={{ color: 'red' }}>Red ({gameState.off[1]})</span>
                </div>
            </div>

            {/* RIGHT: Sidebar / Move Log */}
            <div style={{ flex: 1, backgroundColor: '#f9f9f9', padding: '15px', borderRadius: '5px', maxHeight: '800px', overflowY: 'auto', border: '1px solid #ccc', minWidth: '250px' }}>
                <h3 style={{ borderBottom: '1px solid #ccc', paddingBottom: '10px', marginTop: 0 }}>Move History</h3>
                <ul style={{ listStyleType: 'none', padding: 0, fontSize: '0.9em', color: '#333' }}>
                    {gameState.history && gameState.history.slice().reverse().map((msg, i) => (
                        <li key={i} style={{ padding: '8px 0', borderBottom: '1px solid #eee' }}>{msg}</li>
                    ))}
                    {!gameState.history && <li>No history yet.</li>}
                </ul>
            </div>

        </div>
    );
};

export default Board;
