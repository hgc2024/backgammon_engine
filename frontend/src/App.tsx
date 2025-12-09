import { DndProvider } from 'react-dnd'
import { HTML5Backend } from 'react-dnd-html5-backend'
import Board from './components/Board'
import './App.css'

function App() {
  return (
    <DndProvider backend={HTML5Backend}>
      <div className="App">
        <h1>Backgammon AI (Gen 3)</h1>
        <Board />
      </div>
    </DndProvider>
  )
}

export default App
