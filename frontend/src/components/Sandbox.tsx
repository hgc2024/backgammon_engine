import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { useDrop } from 'react-dnd';
import { Point } from './Point';
import { Checker } from './Checker';

// Types (Same as Board)
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
    pips: number[];
    device: string;
    history: string[];
}

const API_URL = "http://localhost:8000";

// --- Sub-Component for Bear Off (Simplified for Sandbox) ---
interface BearOffProps {
    owner: string;
    count: number;
    onDrop: () => void;
}

const SandboxBearOffZone: React.FC<BearOffProps> = ({ owner, count, onDrop }) => {
    const [{ isOver }, drop] = useDrop(() => ({
        accept: 'CHECKER',
        drop: () => onDrop(),
        collect: (monitor) => ({
            isOver: !!monitor.isOver(),
        }),
    }), [onDrop]);

    return (
        <div ref={drop as any} style={{
            flex: 1,
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            backgroundColor: isOver ? 'yellow' : 'transparent',
            padding: 5
        }}>
            <div>Off<br />{owner}</div>
            <span style={{ fontSize: '1.5em' }}>{count}</span>
        </div>
    );
};

// --- Sub-Component for Bar Drop (Drag to Bar) ---
interface BarZoneProps {
    color: number; // 1 for Player, -1 for CPU
    onDrop: () => void;
    children: React.ReactNode;
}

const SandboxBarZoneWrapper: React.FC<BarZoneProps> = ({ color, onDrop, children }) => {
    const [{ isOver, canDrop }, drop] = useDrop(() => ({
        accept: 'CHECKER',
        // Allow drop if dragging SAME color
        canDrop: (item: { color: number }) => item.color === color,
        drop: () => onDrop(),
        collect: (monitor) => ({
            isOver: !!monitor.isOver(),
            canDrop: !!monitor.canDrop(),
        }),
    }), [color, onDrop]);

    const highlight = isOver && canDrop ? 'yellow' : (canDrop ? 'rgba(0, 255, 0, 0.2)' : 'transparent');

    return (
        <div ref={drop as any} style={{ position: 'relative', backgroundColor: highlight, padding: '5px', borderRadius: '4px' }}>
            {children}
        </div>
    );
};

export const Sandbox: React.FC = () => {
    // --- STATE ---
    // Start with empty/default state
    const [gameState, setGameState] = useState<GameState>({
        board: Array(24).fill(0),
        bar: [0, 0],
        off: [0, 0],
        turn: 0,
        dice: [0, 0],
        cube_value: 1,
        phase: "DECIDE_MOVE",
        score: [0, 0],
        winner: -1,
        pips: [0, 0],
        device: "N/A",
        history: []
    });
    const [diceInput, setDiceInput] = useState<string>("3,1");
    const [evalResult, setEvalResult] = useState<{ equity: number, win_prob: number } | null>(null);
    const [isLoading, setIsLoading] = useState<boolean>(false);

    // --- API SYNC ---
    const fetchState = async () => {
        try {
            const res = await axios.get(`${API_URL}/gamestate`);
            setGameState(res.data);
            setDiceInput(`${res.data.dice[0]},${res.data.dice[1] || 0}`);
        } catch (e) {
            console.error(e);
        }
    };

    useEffect(() => {
        fetchState();
    }, []);

    const pushState = async (newState: GameState) => {
        setGameState(newState); // Optimistic update
        try {
            await axios.post(`${API_URL}/set-state`, {
                board: newState.board,
                bar: newState.bar,
                off: newState.off,
                turn: newState.turn,
                score: newState.score
            });
        } catch (e) {
            console.error("Failed to sync state", e);
        }
    };

    const handleSetDice = async () => {
        try {
            const parts = diceInput.split(',').map(x => parseInt(x.trim()));
            if (parts.length >= 1 && !isNaN(parts[0])) {
                const d1 = parts[0];
                const d2 = parts.length > 1 ? parts[1] : 0;

                // Validate Range
                if (d1 < 1 || d1 > 6 || (d2 !== 0 && (d2 < 1 || d2 > 6))) {
                    alert("Dice must be between 1 and 6.");
                    return;
                }

                await axios.post(`${API_URL}/set-dice`, { dice: [d1, d2] });
                fetchState();
            }
        } catch (e) {
            alert("Invalid Format. Use '3,1' or '6,6'");
        }
    };

    const handleEvaluate = async () => {
        setIsLoading(true);
        try {
            const res = await axios.post(`${API_URL}/evaluate`);
            setEvalResult(res.data);
        } catch (e) {
            console.error(e);
            alert("Evaluation Failed");
        } finally {
            setIsLoading(false);
        }
    };

    const triggerAI = async () => {
        setIsLoading(true);
        try {
            await axios.post(`${API_URL}/ai-move`, {});
            await fetchState();
        } catch (e) {
            console.error(e);
        } finally {
            setIsLoading(false);
        }
    };

    const clearBoard = () => {
        const reset: GameState = {
            ...gameState,
            board: Array(24).fill(0),
            bar: [0, 0],
            off: [0, 0]
        };
        pushState(reset);
    };

    const resetStandard = async () => {
        await axios.post(`${API_URL}/start`, { first_player: 0, reset_score: true });
        fetchState();
    };

    // --- SANDBOX MOVES ---
    // In Sandbox, "moves" are free. 
    // Moving from A to B: Decrement A, Increment B.
    // We infer Color based on content of A.
    // If A is off/bar, we handle that.

    const handleFreeMove = (fromIdx: number | 'bar' | 'bar_cpu', toIdx: number | 'off' | 'bar') => {
        const newState = { ...gameState };
        const board = [...newState.board];
        const bar = [...newState.bar];
        const off = [...newState.off];

        let color = 0; // 1 (White/Player) or -1 (Red/CPU)

        // 1. REMOVE FROM SOURCE
        if (fromIdx === 'bar') {
            if (bar[0] > 0) { bar[0]--; color = 1; }
            else return;
        } else if (fromIdx === 'bar_cpu') {
            if (bar[1] > 0) { bar[1]--; color = -1; }
            else return;
        } else if (typeof fromIdx === 'number') {
            const val = board[fromIdx];
            if (val === 0) return;
            color = val > 0 ? 1 : -1;

            if (val > 0) board[fromIdx]--;
            else board[fromIdx]++; // Remove negative checker (-2 -> -1)
        }

        // 2. ADD TO DESTINATION
        if (toIdx === 'off') {
            if (color === 1) off[0]++;
            else off[1]++;
        } else if (toIdx === 'bar') {
            // Added to Bar (dragged from board to bar)
            if (color === 1) bar[0]++;
            else bar[1]++;
        } else if (typeof toIdx === 'number') {
            // Check collision? In sandbox, we allow "messy" states?
            // Or force replace?
            // Let's implement logic: 
            // If dest has Same Color or Empty -> Add.
            // If dest has Opponent -> Hit (Send to bar).

            const destVal = board[toIdx];
            const sameColor = (color === 1 && destVal >= 0) || (color === -1 && destVal <= 0);

            if (sameColor) {
                if (color === 1) board[toIdx]++;
                else board[toIdx]--;
            } else {
                // Hit!
                // Remove opponent
                if (destVal > 0) {
                    // White was here, Red Hit
                    board[toIdx] = -1;
                    bar[0] += destVal; // Send all to bar? usually 1.
                } else {
                    // Red was here, White Hit
                    board[toIdx] = 1;
                    bar[1] += Math.abs(destVal);
                }
            }
        }

        newState.board = board;
        newState.bar = bar;
        newState.off = off;
        pushState(newState);
    };

    // Click Handlers for "Painting"
    const handlePointClick = (idx: number, e: React.MouseEvent) => {
        // Left Click: Add White (+1)
        // Right Click: Add Red (-1)
        // Ctrl+Click: Remove (towards 0)
        e.preventDefault();

        const newState = { ...gameState };
        const board = [...newState.board];

        if (e.ctrlKey) {
            // Remove
            if (board[idx] > 0) board[idx]--;
            if (board[idx] < 0) board[idx]++;
        } else if (e.type === 'contextmenu') {
            // Right Click -> Add Red
            if (board[idx] <= 0) board[idx]--;
            else {
                // Was White, Switch to Red 1
                board[idx] = -1;
            }
        } else {
            // Left Click -> Add White
            if (board[idx] >= 0) board[idx]++;
            else {
                // Was Red, switch to White 1
                board[idx] = 1;
            }
        }

        // update
        newState.board = board;
        pushState(newState);
    };

    return (
        <div style={{ display: 'flex', minHeight: '100vh', fontFamily: 'sans-serif', backgroundColor: '#eef1f5', color: '#1a1a1a' }}>

            {/* --- EDITOR SIDEBAR --- */}
            <div style={{ width: '300px', backgroundColor: '#333', color: '#fff', padding: '20px', display: 'flex', flexDirection: 'column', gap: '20px' }}>
                <h2 style={{ borderBottom: '1px solid #555', paddingBottom: '10px' }}>Sandbox Mode</h2>

                <div style={{ fontSize: '0.9em', color: '#aaa' }}>
                    <b>How to Edit:</b>
                    <ul style={{ paddingLeft: '20px', marginTop: '5px' }}>
                        <li>Drag & Drop freely</li>
                        <li><b>L-Click</b> Point: Add White</li>
                        <li><b>R-Click</b> Point: Add Red</li>
                        <li><b>Ctrl+Click</b>: Remove from Board</li>
                        <li><b>L-Click</b> Bar/Off: Add/Inc</li>
                        <li><b>R-Click</b> Bar/Off: Remove/Dec</li>
                    </ul>
                </div>

                {/* Validation Status */}
                {(() => {
                    let whiteCount = gameState.bar[0] + gameState.off[0];
                    let redCount = gameState.bar[1] + gameState.off[1];
                    gameState.board.forEach(p => {
                        if (p > 0) whiteCount += p;
                        else if (p < 0) redCount += Math.abs(p);
                    });

                    const whiteValid = whiteCount === 15;
                    const redValid = redCount === 15;

                    if (whiteValid && redValid) return null;

                    return (
                        <div style={{ backgroundColor: '#e74c3c', color: 'white', padding: '10px', borderRadius: '4px', fontSize: '0.9em' }}>
                            <b>⚠️ Invalid Board Limit</b>
                            {!whiteValid && <div>White pieces: {whiteCount} (Need 15)</div>}
                            {!redValid && <div>Red pieces: {redCount} (Need 15)</div>}
                        </div>
                    );
                })()}

                <div style={{ display: 'flex', flexDirection: 'column', gap: '10px', backgroundColor: '#444', padding: '10px', borderRadius: '4px' }}>
                    <label>Dice Override (d1, d2)</label>
                    <div style={{ display: 'flex', gap: '5px' }}>
                        <input value={diceInput} onChange={e => setDiceInput(e.target.value)} style={{ flex: 1, padding: 5 }} />
                        <button onClick={handleSetDice} style={{ cursor: 'pointer' }}>Set</button>
                    </div>
                </div>

                <div style={{ display: 'flex', flexDirection: 'column', gap: '10px', backgroundColor: '#444', padding: '10px', borderRadius: '4px' }}>
                    <label>Current Turn</label>
                    <div style={{ display: 'flex', gap: '5px' }}>
                        <button
                            onClick={() => pushState({ ...gameState, turn: 0 })}
                            style={{ flex: 1, backgroundColor: gameState.turn === 0 ? '#2ecc71' : '#555', border: 'none', padding: 8, color: 'white', cursor: 'pointer' }}>
                            You (White)
                        </button>
                        <button
                            onClick={() => pushState({ ...gameState, turn: 1 })}
                            style={{ flex: 1, backgroundColor: gameState.turn === 1 ? '#e74c3c' : '#555', border: 'none', padding: 8, color: 'white', cursor: 'pointer' }}>
                            CPU (Red)
                        </button>
                    </div>
                </div>

                <button
                    onClick={handleEvaluate}
                    disabled={isLoading}
                    style={{ padding: '10px', backgroundColor: '#8e44ad', color: 'white', border: 'none', borderRadius: '4px', cursor: 'pointer', marginBottom: '10px', fontWeight: 'bold' }}>
                    ⚖️ Evaluate Win Chance
                </button>

                {evalResult && (
                    <div style={{ backgroundColor: '#2c3e50', color: '#f1c40f', padding: '10px', borderRadius: '4px', marginBottom: '10px', fontSize: '0.9em' }}>
                        <div>Equity: {evalResult.equity.toFixed(3)}</div>
                        <div><b>Win Probability: {evalResult.win_prob.toFixed(1)}%</b></div>
                    </div>
                )}

                <div style={{ backgroundColor: '#222', color: '#0f0', padding: '10px', borderRadius: '4px', fontSize: '0.85em', fontFamily: 'monospace', minHeight: '40px', wordBreak: 'break-word' }}>
                    {gameState.history.length > 0 ? gameState.history[gameState.history.length - 1] : "Waiting for move..."}
                </div>

                <button onClick={triggerAI} disabled={isLoading} style={{ padding: '15px', backgroundColor: '#3498db', color: 'white', border: 'none', borderRadius: '4px', fontWeight: 'bold', fontSize: '1.1em', cursor: 'pointer' }}>
                    {isLoading ? "Thinking..." : "▶ Trigger AI"}
                </button>

                <div style={{ marginTop: 'auto', display: 'flex', flexDirection: 'column', gap: '10px' }}>
                    <button onClick={clearBoard} style={{ padding: '10px', backgroundColor: '#e67e22', border: 'none', color: 'white', cursor: 'pointer' }}>Clear Board</button>
                    <button onClick={resetStandard} style={{ padding: '10px', backgroundColor: '#c0392b', border: 'none', color: 'white', cursor: 'pointer' }}>Reset to Standard</button>
                </div>

            </div>

            {/* --- BOARD AREA (Simplified) --- */}
            <div style={{ flex: 1, padding: '20px', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                <div style={{ width: '800px', marginRight: '100px', position: 'relative', backgroundColor: '#f5deb3', border: '15px solid #555', borderRadius: '15px', minHeight: '660px', boxShadow: '0 15px 35px rgba(0,0,0,0.5)' }}>

                    {/* Top Row: 12-17, 18-23 */}
                    <div style={{ display: 'flex', height: '300px', borderBottom: '6px solid #777' }}>
                        {Array.from({ length: 6 }, (_, i) => 12 + i).map(i => (
                            <div key={i} style={{ flex: 1, position: 'relative' }}
                                onContextMenu={(e) => handlePointClick(i, e)}
                                onClick={(e) => handlePointClick(i, e)}
                            >
                                <Point index={i} checkers={gameState.board[i]} onDropChecker={handleFreeMove} legalMoves={[]} isSandbox={true} />
                            </div>
                        ))}
                        <div style={{ width: '25px', backgroundColor: '#444' }} />
                        {Array.from({ length: 6 }, (_, i) => 18 + i).map(i => (
                            <div key={i} style={{ flex: 1, position: 'relative' }}
                                onContextMenu={(e) => handlePointClick(i, e)}
                                onClick={(e) => handlePointClick(i, e)}
                            >
                                <Point index={i} checkers={gameState.board[i]} onDropChecker={handleFreeMove} legalMoves={[]} isSandbox={true} />
                            </div>
                        ))}
                    </div>

                    {/* Bar */}
                    <div style={{ height: '60px', backgroundColor: '#333', color: 'white', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                        <div style={{ marginRight: 20 }}>BAR</div>
                        {/* Player Bar (Click to add, Right-Click to remove) */}
                        <div style={{ cursor: 'pointer' }}
                            onClick={() => {
                                const newState = { ...gameState }; newState.bar[0]++; pushState(newState);
                            }}
                            onContextMenu={(e) => {
                                e.preventDefault();
                                if (gameState.bar[0] > 0) {
                                    const newState = { ...gameState }; newState.bar[0]--; pushState(newState);
                                }
                            }}
                        >
                            <SandboxBarZoneWrapper color={1} onDrop={() => handleFreeMove('bar', 'bar')}>
                                <Checker color={1} count={gameState.bar[0]} pointIndex="bar" canDrag={true} />
                            </SandboxBarZoneWrapper>
                        </div>
                        <div style={{ width: 50 }}></div>
                        {/* CPU Bar */}
                        <div style={{ cursor: 'pointer' }}
                            onClick={() => {
                                const newState = { ...gameState }; newState.bar[1]++; pushState(newState);
                            }}
                            onContextMenu={(e) => {
                                e.preventDefault();
                                if (gameState.bar[1] > 0) {
                                    const newState = { ...gameState }; newState.bar[1]--; pushState(newState);
                                }
                            }}
                        >
                            <SandboxBarZoneWrapper color={-1} onDrop={() => handleFreeMove('bar_cpu', 'bar')}>
                                <Checker color={-1} count={gameState.bar[1]} pointIndex="bar_cpu" canDrag={true} />
                            </SandboxBarZoneWrapper>
                        </div>
                    </div>

                    {/* Bottom Row: 11-6, 5-0 */}
                    <div style={{ display: 'flex', height: '300px', borderTop: '6px solid #777' }}>
                        {Array.from({ length: 6 }, (_, i) => 11 - i).map(i => (
                            <div key={i} style={{ flex: 1, position: 'relative' }}
                                onContextMenu={(e) => handlePointClick(i, e)}
                                onClick={(e) => handlePointClick(i, e)}
                            >
                                <Point index={i} checkers={gameState.board[i]} onDropChecker={handleFreeMove} legalMoves={[]} isSandbox={true} />
                            </div>
                        ))}
                        <div style={{ width: '25px', backgroundColor: '#444' }} />
                        {Array.from({ length: 6 }, (_, i) => 5 - i).map(i => (
                            <div key={i} style={{ flex: 1, position: 'relative' }}
                                onContextMenu={(e) => handlePointClick(i, e)}
                                onClick={(e) => handlePointClick(i, e)}
                            >
                                <Point index={i} checkers={gameState.board[i]} onDropChecker={handleFreeMove} legalMoves={[]} isSandbox={true} />
                            </div>
                        ))}
                    </div>

                    {/* Bear Off Zone */}
                    <div style={{ position: 'absolute', right: -90, top: 0, bottom: 0, width: 80, display: 'flex', flexDirection: 'column', height: '100%' }}>
                        <div style={{ flex: 1, display: 'flex', border: '2px solid #8d6e63', background: '#fff8e1', marginBottom: 5, cursor: 'pointer' }}
                            onClick={() => { const newState = { ...gameState }; newState.off[1]++; pushState(newState); }}
                            onContextMenu={(e) => { e.preventDefault(); if (gameState.off[1] > 0) { const newState = { ...gameState }; newState.off[1]--; pushState(newState); } }}
                        >
                            <SandboxBearOffZone owner="CPU" count={gameState.off[1]} onDrop={() => handleFreeMove('bar_cpu', 'off')} />
                        </div>
                        <div style={{ flex: 1, display: 'flex', border: '2px solid #8d6e63', background: '#fff8e1', cursor: 'pointer' }}
                            onClick={() => { const newState = { ...gameState }; newState.off[0]++; pushState(newState); }}
                            onContextMenu={(e) => { e.preventDefault(); if (gameState.off[0] > 0) { const newState = { ...gameState }; newState.off[0]--; pushState(newState); } }}
                        >
                            <SandboxBearOffZone owner="You" count={gameState.off[0]} onDrop={() => handleFreeMove('bar', 'off')} />
                        </div>
                    </div>

                </div>
            </div>
        </div>
    );
};
