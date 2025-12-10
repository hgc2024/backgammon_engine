import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Point } from './Point';

// Types
interface GameState {
    board: number[];
    bar: number[];
    off: number[];
    turn: number;
    dice: number[];
    cube_value: number;
    phase: string;
    score: number[];
    winner: number;
    history: string[];
}

const API_URL = "http://localhost:8000";

export const Board: React.FC = () => {
    // --- STATE ---
    const [gameState, setGameState] = useState<GameState | null>(null);
    const [legalMoves, setLegalMoves] = useState<number[][]>([]);
    const [aiDepth, setAiDepth] = useState<number>(2);
    const [isLoading, setIsLoading] = useState<boolean>(false);
    const [message, setMessage] = useState<string>("");

    // --- API HELPERS ---
    const fetchState = async () => {
        try {
            const res = await axios.get(`${API_URL}/gamestate`);
            setGameState(res.data);
            const movesRes = await axios.get(`${API_URL}/moves`);
            setLegalMoves(movesRes.data);
            setMessage(""); // Clear errors on success
        } catch (e) {
            console.error(e);
            setMessage("Connection Error - Is backend running?");
        }
    };

    // Initial Load Only
    useEffect(() => {
        fetchState();
    }, []);

    // --- HANDLERS ---
    const withLoading = async (fn: () => Promise<void>) => {
        if (isLoading) return;
        setIsLoading(true);
        try {
            await fn();
            await fetchState(); // Always Refresh after action
        } catch (e: any) {
            console.error(e);
            if (e.response?.data?.error) setMessage(e.response.data.error);
        } finally {
            setIsLoading(false);
        }
    };

    const handleStart = (player: number) => withLoading(async () => {
        await axios.post(`${API_URL}/start`, { first_player: player });
    });

    const handleRoll = () => withLoading(async () => {
        await axios.post(`${API_URL}/roll`);
    });

    const handleAIMove = () => withLoading(async () => {
        await axios.post(`${API_URL}/ai-move`, { depth: aiDepth });
    });

    const handleMove = (fromIdx: number, toIdx: number) => withLoading(async () => {
        await axios.post(`${API_URL}/step`, { move: [fromIdx, toIdx] });
    });

    const handlePass = () => withLoading(async () => {
        await axios.post(`${API_URL}/pass`);
    });

    const handleUndo = () => withLoading(async () => {
        await axios.post(`${API_URL}/undo`);
    });

    // --- RENDER ---
    const isLegalDestination = (idx: number) => legalMoves.some(m => m[1] === idx);

    if (!gameState) return <div style={{ padding: 20 }}>Loading Game... {message}</div>;

    const isCpuTurn = gameState.turn === 1;
    const isHumanTurn = gameState.turn === 0;
    const canRoll = gameState.phase === "DECIDE_CUBE_OR_ROLL" && isHumanTurn;
    const isGameOver = gameState.phase === "GAME_OVER";

    // Auto-scroll logs
    const logs = [...gameState.history].reverse();

    return (
        <div style={{ display: 'flex', minHeight: '100vh', fontFamily: 'sans-serif', backgroundColor: '#eef1f5', color: '#1a1a1a' }}>

            {/* --- SIDEBAR (Streamlit Style) --- */}
            <div style={{ width: '380px', backgroundColor: '#ffffff', padding: '25px', borderRight: '1px solid #ddd', display: 'flex', flexDirection: 'column', gap: '25px', boxShadow: '2px 0 10px rgba(0,0,0,0.05)' }}>
                <div>
                    <h2 style={{ marginTop: 0, marginBottom: '5px', color: '#1f1f1f' }}>Backgammon AI</h2>
                    <div style={{ fontSize: '0.95em', color: '#555', fontWeight: '500' }}>Gen 3 Engine (TD-Gammon)</div>
                </div>

                {/* Score Card */}
                <div style={{ padding: '15px', backgroundColor: '#f8f9fa', borderRadius: '8px', border: '1px solid #e0e0e0' }}>
                    <div style={{ fontWeight: 'bold', marginBottom: '10px', color: '#333' }}>Score</div>
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '1.1em' }}>
                        <span style={{ color: '#2c3e50' }}>You: <b>{gameState.score[0]}</b></span>
                        <span style={{ color: '#c0392b' }}>CPU: <b>{gameState.score[1]}</b></span>
                    </div>
                </div>

                {/* Controls */}
                <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
                    <label style={{ fontWeight: 'bold', color: '#333' }}>Difficulty</label>
                    <select value={aiDepth} onChange={e => setAiDepth(Number(e.target.value))} style={{ padding: '10px', borderRadius: '4px', border: '1px solid #ccc', backgroundColor: 'white', color: '#000' }}>
                        <option value={1}>1-Ply (Fast)</option>
                        <option value={2}>2-Ply (Grandmaster)</option>
                    </select>
                </div>

                {message && <div style={{ backgroundColor: '#ffebee', color: '#c62828', padding: '10px', borderRadius: '4px', fontSize: '0.9em', border: '1px solid #ffcdd2' }}>{message}</div>}

                <div style={{ display: 'flex', gap: '8px' }}>
                    <button onClick={fetchState} disabled={isLoading} style={{ flex: 1, padding: '8px', cursor: 'pointer', backgroundColor: '#fff', border: '1px solid #ccc', borderRadius: '4px', color: '#333' }}>Refresh</button>
                    <button onClick={handleUndo} disabled={isLoading} style={{ flex: 1, padding: '8px', cursor: 'pointer', backgroundColor: '#fff', border: '1px solid #ccc', borderRadius: '4px', color: '#333' }}>Undo</button>
                </div>

                <hr style={{ width: '100%', border: 'none', borderTop: '1px solid #eee' }} />

                {/* Action Buttons */}
                <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                    {isGameOver ? (
                        <>
                            <button onClick={() => handleStart(0)} className="btn-primary">New API Match (You)</button>
                            <button onClick={() => handleStart(1)} className="btn-primary">New API Match (CPU)</button>
                            <button onClick={() => handleStart(-1)} className="btn-primary">New API Match (Random)</button>
                        </>
                    ) : (
                        <>
                            {canRoll && <button onClick={handleRoll} className="btn-green" disabled={isLoading}>🎲 Roll Dice</button>}

                            {isCpuTurn && (
                                <button onClick={handleAIMove} className="btn-blue" disabled={isLoading}>
                                    {isLoading ? "Thinking..." : "🤖 Trigger AI Move"}
                                </button>
                            )}

                            {!canRoll && isHumanTurn && legalMoves.length === 0 && (
                                <button onClick={handlePass} className="btn-red" disabled={isLoading}>Pass Turn (No Moves)</button>
                            )}
                        </>
                    )}
                </div>

                <hr style={{ width: '100%', border: 'none', borderTop: '1px solid #eee' }} />

                {/* Logs */}
                <div style={{ flex: 1, overflowY: 'auto', fontSize: '0.9em', color: '#111', backgroundColor: '#fafafa', padding: '10px', borderRadius: '4px', border: '1px solid #eee' }}>
                    <b style={{ display: 'block', marginBottom: '8px', color: '#333' }}>Action Log</b>
                    <ul style={{ paddingLeft: '20px', marginTop: '0', marginBottom: 0 }}>
                        {logs.map((log, i) => <li key={i} style={{ marginBottom: '4px' }}>{log}</li>)}
                    </ul>
                </div>

            </div>

            {/* --- MAIN CONTENT (Board) --- */}
            <div style={{ flex: 1, padding: '40px', display: 'flex', flexDirection: 'column', alignItems: 'center', backgroundColor: '#eef1f5' }}>

                <div style={{ marginBottom: '30px', fontSize: '1.6em', textAlign: 'center', backgroundColor: 'white', padding: '15px 30px', borderRadius: '10px', boxShadow: '0 2px 5px rgba(0,0,0,0.05)' }}>
                    <span style={{ color: '#555' }}>Phase:</span> <b style={{ color: '#000' }}>{gameState.phase}</b>
                    <span style={{ margin: '0 20px', color: '#ccc' }}>|</span>
                    <span style={{ color: '#555' }}>Turn:</span> <span style={{ color: gameState.turn === 0 ? '#27ae60' : '#c0392b', fontWeight: 'bold' }}>{gameState.turn === 0 ? "YOU (White)" : "CPU (Red)"}</span>
                    <div style={{ marginTop: '8px', fontSize: '0.9em', color: '#444' }}>Dice: <b>{JSON.stringify(gameState.dice)}</b></div>
                </div>

                {/* Visual Board - Scaled UP */}
                <div style={{ width: '800px', position: 'relative', backgroundColor: '#f5deb3', border: '15px solid #6d4c41', borderRadius: '15px', minHeight: '600px', boxShadow: '0 15px 35px rgba(0,0,0,0.15)' }}>

                    {/* Top Row */}
                    <div style={{ display: 'flex', height: '270px', borderBottom: '6px solid #8d6e63' }}>
                        {Array.from({ length: 12 }, (_, i) => 12 + i).map(i => (
                            <Point key={i} index={i} checkers={gameState.board[i]} onDropChecker={handleMove} canMoveTo={isLegalDestination(i)} />
                        ))}
                    </div>

                    {/* Bar Area */}
                    <div style={{ height: '60px', backgroundColor: '#6d4c41', color: 'white', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '1.3em', fontWeight: 'bold', textShadow: '1px 1px 2px rgba(0,0,0,0.3)' }}>
                        BAR <span style={{ fontSize: '0.8em', marginLeft: '15px', fontWeight: 'normal', opacity: 0.9 }}>You: {gameState.bar[0]} | CPU: {gameState.bar[1]}</span>
                    </div>

                    {/* Bottom Row */}
                    <div style={{ display: 'flex', height: '270px', borderTop: '6px solid #8d6e63' }}>
                        {Array.from({ length: 12 }, (_, i) => 11 - i).map(i => (
                            <Point key={i} index={i} checkers={gameState.board[i]} onDropChecker={handleMove} canMoveTo={isLegalDestination(i)} />
                        ))}
                    </div>

                    {/* Bear Off Zone */}
                    <div style={{ position: 'absolute', right: -90, top: 0, bottom: 0, width: 80, display: 'flex', flexDirection: 'column', padding: 10, background: '#fff8e1', border: '2px solid #8d6e63', fontWeight: 'bold', color: '#4e342e', textAlign: 'center' }}>
                        <div>Off<br />You<br /><span style={{ fontSize: '1.5em' }}>{gameState.off[0]}</span></div>
                        <div style={{ marginTop: 'auto' }}>Off<br />CPU<br /><span style={{ fontSize: '1.5em' }}>{gameState.off[1]}</span></div>
                    </div>

                </div>

            </div>

            <style>{`
                .btn-primary { padding: 12px; background: #ff4b4b; color: white; border: none; borderRadius: 6px; cursor: pointer; font-weight: 500; width: 100%; text-align: left; transition: background 0.2s; font-size: 1rem; }
                .btn-primary:hover { background: #ff3333; }
                
                .btn-green { padding: 12px; background: #27ae60; color: white; border: none; borderRadius: 6px; cursor: pointer; font-weight: bold; width: 100%; font-size: 1.1rem; box-shadow: 0 4px 0 #219150; }
                .btn-green:active { transform: translateY(2px); box-shadow: 0 2px 0 #219150; }

                .btn-red { padding: 12px; background: #c0392b; color: white; border: none; borderRadius: 6px; cursor: pointer; font-weight: bold; width: 100%; font-size: 1.1rem; }
                
                .btn-blue { padding: 12px; background: #2980b9; color: white; border: none; borderRadius: 6px; cursor: pointer; font-weight: bold; width: 100%; font-size: 1.1rem; box-shadow: 0 4px 0 #2471a3; }
                .btn-blue:active { transform: translateY(2px); box-shadow: 0 2px 0 #2471a3; }

                button:disabled { opacity: 0.5; cursor: not-allowed; box-shadow: none !important; transform: none !important; }
            `}</style>
        </div>
    );
};

export default Board;
