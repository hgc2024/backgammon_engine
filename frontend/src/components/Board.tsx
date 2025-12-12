import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { useDrop } from 'react-dnd';
import { Point } from './Point';
import { Checker } from './Checker';

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
    pips: number[]; // Gen 4
    device: string;
    history: string[];
}

const API_URL = "http://localhost:8000";

// --- Sub-Component for Bear Off ---
interface BearOffProps {
    owner: string;
    count: number;
    legalMoves: any[][];
    onDrop: (fromIndex: number) => void;
}

const BearOffZone: React.FC<BearOffProps> = ({ owner, count, legalMoves, onDrop }) => {
    // Only Player (White, 0) bears off to 25/off.
    // If owner is 'You', we check if 'off' is a legal move from dragging checker's index.

    // Note: CPU doesn't drag, so we only care about Player bearing off.
    const isPlayer = owner === 'You';

    const [{ isOver, canDrop }, drop] = useDrop(() => ({
        accept: 'CHECKER',
        canDrop: (item: { pointIndex: number, color: number }) => {
            if (!isPlayer) return false;
            // Check if ANY move allows this checker to go 'off'
            return legalMoves.some(m => m[0] === item.pointIndex && m[1] === 'off');
        },
        drop: (item: { pointIndex: number }) => onDrop(item.pointIndex),
        collect: (monitor) => ({
            isOver: !!monitor.isOver(),
            canDrop: !!monitor.canDrop(),
        }),
    }), [legalMoves, onDrop, isPlayer]);

    const highlight = isOver && canDrop ? 'yellow' : (canDrop ? 'rgba(0, 255, 0, 0.2)' : 'transparent');

    return (
        <div ref={drop as any} style={{
            flex: 1,
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            backgroundColor: highlight,
            padding: 5
        }}>
            <div>Off<br />{owner}</div>
            <span style={{ fontSize: '1.5em' }}>{count}</span>
        </div>
    );
};


export const Board: React.FC = () => {
    // --- STATE ---
    const [gameState, setGameState] = useState<GameState | null>(null);
    const [legalMoves, setLegalMoves] = useState<any[][]>([]); // Can contain 'off'
    // const [aiDepth, setAiDepth] = useState<number>(2); // Removed
    // const [aiStyle, setAiStyle] = useState<string>("aggressive"); // Removed
    const [isLoading, setIsLoading] = useState<boolean>(false);
    const [message, setMessage] = useState<string>("");
    const [startOption, setStartOption] = useState<number>(-1); // -1: Random, 0: Player, 1: CPU

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

    const handleStart = (player: number, resetScore: boolean) => withLoading(async () => {
        await axios.post(`${API_URL}/start`, { first_player: player, reset_score: resetScore });
    });

    const handleRoll = () => withLoading(async () => {
        await axios.post(`${API_URL}/roll`);
    });

    const handleAIMove = () => withLoading(async () => {
        await axios.post(`${API_URL}/ai-move`, {}); // depth/style removed
    });

    const handleMove = (fromIdx: number, toIdx: number | 'off') => withLoading(async () => {
        await axios.post(`${API_URL}/step`, { move: [fromIdx, toIdx] });
    });

    const handleUndo = () => withLoading(async () => {
        await axios.post(`${API_URL}/undo`);
    });

    const handlePass = () => withLoading(async () => {
        await axios.post(`${API_URL}/pass`);
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
                    <div style={{ fontSize: '0.95em', color: '#555', fontWeight: '500' }}>Gen 5 Engine (Transformer)</div>
                    <div style={{ fontSize: '0.8em', color: '#888', marginTop: '2px' }}>Hardware: {gameState.device}</div>
                </div>

                {/* Score Card */}
                <div style={{ padding: '15px', backgroundColor: '#f8f9fa', borderRadius: '8px', border: '1px solid #e0e0e0' }}>
                    <div style={{ fontWeight: 'bold', marginBottom: '10px', color: '#333' }}>Score</div>
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '1.1em' }}>
                        <span style={{ color: '#2c3e50' }}>You: <b>{gameState.score[0]}</b> <small style={{ color: '#777' }}>({gameState.pips ? gameState.pips[0] : '-'} pips)</small></span>
                        <span style={{ color: '#c0392b' }}>CPU: <b>{gameState.score[1]}</b> <small style={{ color: '#777' }}>({gameState.pips ? gameState.pips[1] : '-'} pips)</small></span>
                    </div>
                </div>

                {/* Game Settings */}
                <div style={{ display: 'flex', flexDirection: 'column', gap: '15px' }}>

                    {/* Difficulty Removed (Hardcoded to 2-Ply) */}
                    {/* 
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '5px' }}>
                        <label style={{ fontWeight: 'bold', color: '#333', fontSize: '0.9em' }}>Difficulty</label>
                        <select value={aiDepth} onChange={e => setAiDepth(Number(e.target.value))} style={{ padding: '8px', borderRadius: '4px', border: '1px solid #ccc', backgroundColor: 'white', color: '#000' }}>
                            <option value={1}>1-Ply (Fast)</option>
                            <option value={2}>2-Ply (Strong)</option>
                        </select>
                    </div> 
                    */}

                    {/* Starting Player Selection (Streamlit Style) */}
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '5px' }}>
                        <label style={{ fontWeight: 'bold', color: '#333' }}>Who Starts?</label>
                        <select
                            value={startOption}
                            onChange={e => setStartOption(Number(e.target.value))}
                            style={{ padding: '10px', borderRadius: '4px', border: '1px solid #ccc', backgroundColor: 'white', color: '#000' }}
                        >
                            <option value={-1}>Random Start</option>
                            <option value={0}>You (White)</option>
                            <option value={1}>CPU (Red)</option>
                        </select>
                        <div style={{ fontSize: '0.8em', color: '#777', fontStyle: 'italic' }}>Select who makes the first move.</div>
                    </div>
                </div>

                {message && <div style={{ backgroundColor: '#ffebee', color: '#c62828', padding: '10px', borderRadius: '4px', fontSize: '0.9em', border: '1px solid #ffcdd2' }}>{message}</div>}

                <div style={{ display: 'flex', gap: '8px' }}>
                    <button onClick={fetchState} disabled={isLoading} style={{ flex: 1, padding: '8px', cursor: 'pointer', backgroundColor: '#fff', border: '1px solid #ccc', borderRadius: '4px', color: '#333' }}>Refresh</button>
                    <button onClick={handleUndo} disabled={isLoading} style={{ flex: 1, padding: '8px', cursor: 'pointer', backgroundColor: '#fff', border: '1px solid #ccc', borderRadius: '4px', color: '#333' }}>Undo</button>
                </div>

                <hr style={{ width: '100%', border: 'none', borderTop: '1px solid #eee' }} />

                {/* Main Action Buttons */}
                <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>

                    {/* Game Actions - Fixed Height Container to prevent layout shift */}
                    <div style={{ minHeight: '60px', display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
                        {!isGameOver && (
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

                    {isGameOver && (
                        <div style={{ textAlign: 'center', padding: '10px', backgroundColor: '#dff0d8', color: '#3c763d', borderRadius: '4px', fontWeight: 'bold' }}>
                            Game Over! {gameState.score[0] > gameState.score[1] ? "You Won!" : "CPU Won!"}
                        </div>
                    )}

                    {/* Spacer to prevent accidental clicks */}
                    <div style={{ height: '40px', borderBottom: '1px dashed #eee', marginBottom: '10px' }}></div>
                    <div style={{ fontSize: '0.85em', color: '#999', textTransform: 'uppercase', letterSpacing: '1px', fontWeight: 'bold' }}>Match Controls</div>

                    {/* Game Flow Controls */}
                    <button onClick={() => handleStart(startOption, false)} className="btn-primary" disabled={isLoading} title="Starts a new game but keeps the current score">New Game (Next Round)</button>
                    <button onClick={() => handleStart(startOption, true)} className="btn-secondary" disabled={isLoading} title="Totally resets the match and score to 0-0">Reset Match (0-0)</button>
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
                <div style={{ width: '800px', position: 'relative', backgroundColor: '#f5deb3', border: '15px solid #6d4c41', borderRadius: '15px', minHeight: '660px', boxShadow: '0 15px 35px rgba(0,0,0,0.15)', marginRight: '100px' }}>

                    {/* Top Row */}
                    <div style={{ display: 'flex', height: '300px', borderBottom: '6px solid #8d6e63' }}>
                        {Array.from({ length: 12 }, (_, i) => 12 + i).map(i => (
                            <Point key={i} index={i} checkers={gameState.board[i]} onDropChecker={handleMove} legalMoves={legalMoves} />
                        ))}
                    </div>

                    {/* Bar Area */}
                    <div style={{ height: '60px', backgroundColor: '#6d4c41', color: 'white', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '1.3em', fontWeight: 'bold', textShadow: '1px 1px 2px rgba(0,0,0,0.3)' }}>
                        <div style={{ marginRight: 20 }}>BAR</div>

                        {/* Player Bar Checker */}
                        {gameState.bar[0] > 0 && (
                            <div style={{ marginRight: 10 }}>
                                <Checker color={1} count={gameState.bar[0]} pointIndex="bar" canDrag={gameState.turn === 0} />
                            </div>
                        )}
                        <span style={{ fontSize: '0.8em', opacity: 0.9 }}>You: {gameState.bar[0]}</span>

                        <div style={{ width: 20 }} />

                        <span style={{ fontSize: '0.8em', opacity: 0.9 }}>CPU: {gameState.bar[1]}</span>
                        {/* CPU Bar Checker (Static visual only since CPU doesn't drag) */}
                        {gameState.bar[1] > 0 && (
                            <div style={{ marginLeft: 10 }}>
                                <Checker color={-1} count={gameState.bar[1]} pointIndex="bar_cpu" canDrag={false} />
                            </div>
                        )}
                    </div>

                    {/* Bottom Row */}
                    <div style={{ display: 'flex', height: '300px', borderTop: '6px solid #8d6e63' }}>
                        {Array.from({ length: 12 }, (_, i) => 11 - i).map(i => (
                            <Point key={i} index={i} checkers={gameState.board[i]} onDropChecker={handleMove} legalMoves={legalMoves} />
                        ))}
                    </div>

                    {/* Bear Off Zone */}
                    <div style={{ position: 'absolute', right: -90, top: 0, bottom: 0, width: 80, display: 'flex', flexDirection: 'column', height: '100%' }}>

                        {/* Player Bear Off (Top or Bottom? Standard is usually same side as home board direction) */}
                        {/* If White moves 12->24(Bottom Right), off is past 24. */}
                        {/* Actually, visually, White(0) moves to 24 (bottom right). So Off is Bottom Right. */}
                        {/* We put Player Off Zone at Bottom? */}
                        {/* Current layout: Top=12-23? No. */}
                        {/* Let's check Points: Top Row: 12..23 (Left->Right). Bottom Row: 11..0 (Right->Left)? */}
                        {/* Logic: Array(12).map((_,i)=>12+i) -> 12,13...23. */}
                        {/* Bottom: 11-i -> 11,10...0. */}
                        {/* White(0) starts at 24? (Usually 2 checkers at 24).  */}
                        {/* If game.board follows standard: 0 is White pos? 23 is Black pos? */}
                        {/* In my logic: White moves 0->23? or 24->1? */}
                        {/* Visuals: Bottom Left = 0. Bottom Right = 11. Top Right = 12. Top Left = 23. */}
                        {/* White moves Positive (0->23). So White bears off at 24 (Top Left?) or 24 (off board). */}
                        {/* Wait, if 12+i is Top Row... that's 12..23. */}
                        {/* If 11-i is Bottom Row... that's 11..0. */}
                        {/* White moves 0->...->23. */}
                        {/* So White destination is > 23. "Top Left"? */}
                        {/* If Top Left is 23, then Off is Left of Top Row. */}
                        {/* But previous BearOff zone was on Right. */}
                        {/* If visual board is rotated, standard is White Home = Inner Board. */}
                        {/* Let's just enable Bear Off for both sides in the container and rely on user to find it. */}
                        {/* I'll stack them: Top (CPU?) Bottom (Player?) or vice versa. */}
                        {/* Previous static code: Top = You. Bottom = CPU. */}

                        <div style={{ flex: 1, display: 'flex', border: '2px solid #8d6e63', background: '#fff8e1', marginBottom: 5 }}>
                            <BearOffZone owner="CPU" count={gameState.off[1]} legalMoves={legalMoves} onDrop={(from) => handleMove(from, 'off')} />
                        </div>

                        <div style={{ flex: 1, display: 'flex', border: '2px solid #8d6e63', background: '#fff8e1' }}>
                            <BearOffZone owner="You" count={gameState.off[0]} legalMoves={legalMoves} onDrop={(from) => handleMove(from, 'off')} />
                        </div>

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

                .btn-secondary { padding: 12px; background: #95a5a6; color: white; border: none; borderRadius: 6px; cursor: pointer; font-weight: 500; width: 100%; text-align: left; transition: background 0.2s; font-size: 1rem; }
                .btn-secondary:hover { background: #7f8c8d; }

                button:disabled { opacity: 0.5; cursor: not-allowed; box-shadow: none !important; transform: none !important; }
            `}</style>
        </div>
    );
};

export default Board;
