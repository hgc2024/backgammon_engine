import { useState } from 'react'
import { DndProvider } from 'react-dnd'
import { HTML5Backend } from 'react-dnd-html5-backend'
import Board from './components/Board'
import { Sandbox } from './components/Sandbox'
import './App.css'

function App() {
  // Default to 'home' to let user choose every time. 
  // We can still use localStorage to *remember* preference but not auto-load it if they want the choice screen.
  // User asked to "Restore the first screen", implying they WANT to see it.
  const [view, setView] = useState<'home' | 'board' | 'sandbox'>('home');

  return (
    <DndProvider backend={HTML5Backend}>
      <div className="App" style={{ display: 'flex', flexDirection: 'column', height: '100vh' }}>

        {/* Navigation Bar (Always visible or only on sub-pages? Let's keep it consistent but simplified on Home) */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '15px 30px', background: '#2c3e50', color: 'white', borderBottom: '1px solid #34495e' }}>
          <h1 style={{ margin: 0, fontSize: '1.4em', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '10px' }}
            onClick={() => setView('home')}>
            <span>🎲</span> Backgammon Gen 5
          </h1>

          {view !== 'home' && (
            <div style={{ display: 'flex', gap: '10px' }}>
              <button onClick={() => setView('board')} style={{ padding: '8px 16px', borderRadius: '4px', border: 'none', background: view === 'board' ? '#3498db' : 'rgba(255,255,255,0.1)', color: 'white', cursor: 'pointer', fontWeight: view === 'board' ? 'bold' : 'normal', transition: 'all 0.2s' }}>Play Game</button>
              <button onClick={() => setView('sandbox')} style={{ padding: '8px 16px', borderRadius: '4px', border: 'none', background: view === 'sandbox' ? '#e67e22' : 'rgba(255,255,255,0.1)', color: 'white', cursor: 'pointer', fontWeight: view === 'sandbox' ? 'bold' : 'normal', transition: 'all 0.2s' }}>Sandbox Editor</button>
            </div>
          )}
        </div>

        {/* Content Area */}
        <div style={{ flex: 1, overflow: 'auto', display: 'flex', flexDirection: 'column' }}>

          {view === 'home' && (
            <div style={{ flex: 1, display: 'flex', justifyContent: 'center', alignItems: 'center', background: '#ecf0f1', gap: '40px' }}>

              {/* Play Card */}
              <div onClick={() => setView('board')}
                style={{ width: '300px', height: '350px', background: 'white', borderRadius: '12px', boxShadow: '0 10px 25px rgba(0,0,0,0.1)', cursor: 'pointer', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', transition: 'transform 0.2s', border: '2px solid transparent' }}
                onMouseEnter={e => e.currentTarget.style.transform = 'translateY(-5px)'}
                onMouseLeave={e => e.currentTarget.style.transform = 'translateY(0)'}
              >
                <div style={{ fontSize: '4em', marginBottom: '20px' }}>🎮</div>
                <h2 style={{ color: '#2c3e50', marginBottom: '10px' }}>Play Game</h2>
                <p style={{ color: '#7f8c8d', textAlign: 'center', padding: '0 20px' }}>Challenge the AI in a standard game of Backgammon.</p>
                <button style={{ marginTop: '30px', padding: '10px 25px', background: '#3498db', color: 'white', border: 'none', borderRadius: '25px', fontSize: '1.1em', cursor: 'pointer' }}>Start Playing</button>
              </div>

              {/* Sandbox Card */}
              <div onClick={() => setView('sandbox')}
                style={{ width: '300px', height: '350px', background: 'white', borderRadius: '12px', boxShadow: '0 10px 25px rgba(0,0,0,0.1)', cursor: 'pointer', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', transition: 'transform 0.2s', border: '2px solid transparent' }}
                onMouseEnter={e => e.currentTarget.style.transform = 'translateY(-5px)'}
                onMouseLeave={e => e.currentTarget.style.transform = 'translateY(0)'}
              >
                <div style={{ fontSize: '4em', marginBottom: '20px' }}>🧪</div>
                <h2 style={{ color: '#2c3e50', marginBottom: '10px' }}>Sandbox Editor</h2>
                <p style={{ color: '#7f8c8d', textAlign: 'center', padding: '0 20px' }}>Setup custom scenarios, edit the board, and test AI moves.</p>
                <button style={{ marginTop: '30px', padding: '10px 25px', background: '#e67e22', color: 'white', border: 'none', borderRadius: '25px', fontSize: '1.1em', cursor: 'pointer' }}>Enter Sandbox</button>
              </div>

            </div>
          )}

          {view === 'board' && <Board />}
          {view === 'sandbox' && <Sandbox />}

        </div>
      </div>
    </DndProvider>
  )
}

export default App
