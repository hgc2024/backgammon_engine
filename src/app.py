import streamlit as st
import numpy as np
import time
import sys
import os

# Path hack to allow importing src
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.game import BackgammonGame, GamePhase
from src.env import BackgammonEnv
from sb3_contrib import MaskablePPO

st.set_page_config(layout="wide", page_title="Backgammon RL")

def init_state():
    if 'env' not in st.session_state:
        st.session_state.env = BackgammonEnv()
        st.session_state.game = st.session_state.env.game
        st.session_state.obs, _ = st.session_state.env.reset()
        st.session_state.done = False
        
        # Load Model
        model_path = "backgammon_final"
        if os.path.exists(model_path + ".zip"):
            st.session_state.model = MaskablePPO.load(model_path)
            st.toast("Model Loaded!", icon="🤖")
        else:
            st.session_state.model = None
            st.toast("No Model Found. Playing vs Random.", icon="🎲")

init_state()

game = st.session_state.game
env = st.session_state.env

st.title("Backgammon RL Engine")

# --- Sidebar ---
with st.sidebar:
    st.header("Game Status")
    st.metric("Score (You vs Cpu)", f"{game.score[0]} - {game.score[1]}")
    st.metric("Cube Value", f"{game.cube_value}")
import streamlit as st
import numpy as np
import torch
import time
import sys
import os

# Path hack to allow importing src
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.game import BackgammonGame, GamePhase
from src.env import BackgammonEnv
from src.search import ExpectiminimaxAgent

st.set_page_config(layout="wide", page_title="Backgammon RL")

def init_state():
    if 'env' not in st.session_state:
        st.session_state.env = BackgammonEnv(match_target=1)
        st.session_state.game = st.session_state.env.game
        st.session_state.obs, _ = st.session_state.env.reset()
        st.session_state.done = False
        
        # Load Model
        # Load Model (TD-Gammon)
        model_path = "td_backgammon.pth"
        if os.path.exists(model_path):
            st.session_state.agent = ExpectiminimaxAgent(model_path, device="cuda" if torch.cuda.is_available() else "cpu")
            st.toast("TD-Gammon Engine Loaded!", icon="🧠")
        else:
            st.session_state.agent = None
            st.toast("No Model Found. Playing vs Random.", icon="🎲")
            
    if 'total_score' not in st.session_state:
        st.session_state.total_score = [0, 0]

init_state()

game = st.session_state.game
env = st.session_state.env

st.title("Backgammon RL Engine")

# --- Sidebar ---
with st.sidebar:
    st.header("Game Status")
    st.metric("Score (You vs Cpu)", f"{game.score[0]} - {game.score[1]}")
    st.metric("Cube Value", f"{game.cube_value}")
    st.metric("Cube Owner", "Cpu" if game.cube_owner == 1 else ("You" if game.cube_owner == 0 else "Centered"))
    
    if st.button("Reset Match"):
        env.reset()
        st.session_state.done = False
        st.session_state.logs = []
        st.rerun()
        
    st.divider()
    st.subheader("Action Log")
    if 'logs' not in st.session_state:
        st.session_state.logs = []
    
    for log in reversed(st.session_state.logs[-10:]):
        st.caption(log)

# --- Board Visualization ---
# --- Board Visualization ---
import streamlit.components.v1 as components

def draw_board_canvas():
    """renders a simple visual representation using HTML/CSS/SVG."""
    board = game.board
    bar = game.bar
    off = game.off
    
    # SVG Config
    width=600
    height=400
    p_w = width / 15 # Point width
    
    # Helper to draw triangle
    def tri(idx, x, is_top, color):
        y1 = 0 if is_top else height
        y2 = height * 0.4 if is_top else height * 0.6
        y3 = 0 if is_top else height
        # Ensure distinct IDs or just basic shapes
        return f'<polygon points="{x},{y1} {x + p_w/2},{y2} {x + p_w},{y3}" fill="{color}" stroke="black" stroke-width="1"/>'
        
    # Helper to draw checker
    def checker(x, y, color, count):
        r = p_w / 2.5
        fill = "white" if color=="white" else "#d9534f" # Reddish for CPU
        stroke = "black" if color=="white" else "#800000"
        t_col = "black" if color=="white" else "white"
        
        return f'''
        <g>
            <circle cx="{x}" cy="{y}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="2"/>
            <text x="{x}" y="{y+5}" font-size="12" text-anchor="middle" fill="{t_col}" font-family="sans-serif" font-weight="bold">{count if count > 0 else ""}</text>
        </g>
        '''
        
    svg_elements = []
    
    # Draw Background
    svg_elements.append(f'<rect width="{width}" height="{height}" fill="#f0d9b5"/>') # Lighter wood
    svg_elements.append(f'<rect x="{width/2-2}" y="0" width="4" height="{height}" fill="#6b4c35"/>') # Bar line
    
    # Draw Points
    # Top: 12..23 (Left to Right)
    for i in range(12, 24):
        pos_idx = i - 12
        is_right = pos_idx >= 6
        
        x = pos_idx * p_w
        if is_right: x += 30 # Bar offset
        
        color = "#8b4513" if i % 2 == 1 else "#d2b48c" # Dark Brown / Tan
        svg_elements.append(tri(i, x, True, color))
        
        # Checkers
        count = board[i]
        if count != 0:
            c_color = "white" if count > 0 else "black"
            num = abs(count)
            # Stack downwards
            for c in range(min(num, 5)):
                # If more than 5, verify visual stacking? 
                # Just cap at 5 visually for now or stack tighter? 
                # Let's stack standard.
                cy = (p_w/2) + (c * p_w)
                txt = num if (c==4 and num>5) else 0
                svg_elements.append(checker(x + p_w/2, cy, c_color, txt))

    # Bottom Points: 11..0 (Left to Right)
    for i in range(11, -1, -1):
        pos_idx = 11 - i
        is_right = pos_idx >= 6
        
        x = pos_idx * p_w
        if is_right: x += 30
        
        color = "#8b4513" if i % 2 == 0 else "#d2b48c"
        svg_elements.append(tri(i, x, False, color))
        
        # Checkers
        count = board[i]
        if count != 0:
            c_color = "white" if count > 0 else "black"
            num = abs(count)
            # Stack upwards
            for c in range(min(num, 5)):
                cy = height - (p_w/2) - (c * p_w)
                txt = num if (c==4 and num>5) else 0
                svg_elements.append(checker(x + p_w/2, cy, c_color, txt))
                
    # Draw Bar Checkers
    if bar[0] > 0: # White on bar
         svg_elements.append(checker(width/2, height*0.66, "white", bar[0]))
    if bar[1] > 0: # Black on bar
         svg_elements.append(checker(width/2, height*0.33, "black", bar[1]))
         
    svg_content = "".join(svg_elements)
    full_html = f'''
        <div style="display: flex; justify-content: center;">
            <svg viewBox="0 0 {width} {height}" width="100%" height="400" xmlns="http://www.w3.org/2000/svg">
                {svg_content}
            </svg>
        </div>
    '''
    components.html(full_html, height=420)

draw_board_canvas()

import streamlit as st
import numpy as np
import time
import sys
import os

# Path hack to allow importing src
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.game import BackgammonGame, GamePhase
from src.env import BackgammonEnv
from sb3_contrib import MaskablePPO

st.set_page_config(layout="wide", page_title="Backgammon RL")

def init_state():
    if 'env' not in st.session_state:
        st.session_state.env = BackgammonEnv()
        st.session_state.game = st.session_state.env.game
        st.session_state.obs, _ = st.session_state.env.reset()
        st.session_state.done = False
        
        # Load Model
        model_path = "backgammon_final"
        if os.path.exists(model_path + ".zip"):
            st.session_state.model = MaskablePPO.load(model_path)
            st.toast("Model Loaded!", icon="🤖")
        else:
            st.session_state.model = None
            st.toast("No Model Found. Playing vs Random.", icon="🎲")

init_state()

game = st.session_state.game
env = st.session_state.env

st.title("Backgammon RL Engine")

# --- Sidebar ---
with st.sidebar:
    st.header("Game Status")
    st.metric("Score (You vs Cpu)", f"{game.score[0]} - {game.score[1]}")
    st.metric("Cube Value", f"{game.cube_value}")
    st.metric("Cube Owner", "Cpu" if game.cube_owner == 1 else ("You" if game.cube_owner == 0 else "Centered"))
    
    if st.button("Reset Match", key="reset_match_sidebar"):
        env.reset()
        st.session_state.done = False
        st.session_state.logs = []
        st.rerun()
        
    st.divider()
    st.subheader("Action Log")
    if 'logs' not in st.session_state:
        st.session_state.logs = []
    
    for log in reversed(st.session_state.logs[-10:]):
        st.caption(log)

# --- Board Visualization ---
import streamlit.components.v1 as components

def draw_board_canvas():
    """renders a simple visual representation using HTML/CSS/SVG."""
    board = game.board
    bar = game.bar
    off = game.off
    
    # SVG Config
    width=600
    height=400
    p_w = width / 15 # Point width
    
    # Helper to draw triangle
    def tri(idx, x, is_top, color):
        y1 = 0 if is_top else height
        y2 = height * 0.4 if is_top else height * 0.6
        y3 = 0 if is_top else height
        # Ensure distinct IDs or just basic shapes
        return f'<polygon points="{x},{y1} {x + p_w/2},{y2} {x + p_w},{y3}" fill="{color}" stroke="black" stroke-width="1"/>'
        
    # Helper to draw checker
    def checker(x, y, color, count):
        r = p_w / 2.5
        fill = "white" if color=="white" else "#d9534f" # Reddish for CPU
        stroke = "black" if color=="white" else "#800000"
        t_col = "black" if color=="white" else "white"
        
        return f'''
        <g>
            <circle cx="{x}" cy="{y}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="2"/>
            <text x="{x}" y="{y+5}" font-size="12" text-anchor="middle" fill="{t_col}" font-family="sans-serif" font-weight="bold">{count if count > 0 else ""}</text>
        </g>
        '''
        
    svg_elements = []
    
    # Draw Background
    svg_elements.append(f'<rect width="{width}" height="{height}" fill="#f0d9b5"/>') # Lighter wood
    svg_elements.append(f'<rect x="{width/2-2}" y="0" width="4" height="{height}" fill="#6b4c35"/>') # Bar line
    
    # Draw Points
    # Top: 12..23 (Left to Right)
    for i in range(12, 24):
        pos_idx = i - 12
        is_right = pos_idx >= 6
        
        x = pos_idx * p_w
        if is_right: x += 30 # Bar offset
        
        color = "#8b4513" if i % 2 == 1 else "#d2b48c" # Dark Brown / Tan
        svg_elements.append(tri(i, x, True, color))
        
        # Checkers
        count = board[i]
        if count != 0:
            c_color = "white" if count > 0 else "black"
            num = abs(count)
            # Stack downwards
            for c in range(min(num, 5)):
                # If more than 5, verify visual stacking? 
                # Just cap at 5 visually for now or stack tighter? 
                # Let's stack standard.
                cy = (p_w/2) + (c * p_w)
                txt = num if (c==4 and num>5) else 0
                svg_elements.append(checker(x + p_w/2, cy, c_color, txt))

    # Bottom Points: 11..0 (Left to Right)
    for i in range(11, -1, -1):
        pos_idx = 11 - i
        is_right = pos_idx >= 6
        
        x = pos_idx * p_w
        if is_right: x += 30
        
        color = "#8b4513" if i % 2 == 0 else "#d2b48c"
        svg_elements.append(tri(i, x, False, color))
        
        # Checkers
        count = board[i]
        if count != 0:
            c_color = "white" if count > 0 else "black"
            num = abs(count)
            # Stack upwards
            for c in range(min(num, 5)):
                cy = height - (p_w/2) - (c * p_w)
                txt = num if (c==4 and num>5) else 0
                svg_elements.append(checker(x + p_w/2, cy, c_color, txt))
                
    # Draw Bar Checkers
    if bar[0] > 0: # White on bar
         svg_elements.append(checker(width/2, height*0.66, "white", bar[0]))
    if bar[1] > 0: # Black on bar
         svg_elements.append(checker(width/2, height*0.33, "black", bar[1]))
         
    svg_content = "".join(svg_elements)
    full_html = f'''
        <div style="display: flex; justify-content: center;">
            <svg viewBox="0 0 {width} {height}" width="100%" height="400" xmlns="http://www.w3.org/2000/svg">
                {svg_content}
            </svg>
        </div>
    '''
    components.html(full_html, height=420)

draw_board_canvas()

with st.expander("Show Internal State (Debug)"):
    st.text(game.render_ascii())

# --- Action Area ---
if not st.session_state.done:    
    # Header for Current Turn
    turn_name = "YOUR TURN (White)" if game.turn == 0 else "CPU TURN (Red)"
    turn_color = "green" if game.turn == 0 else "red"
    st.markdown(f"<h2 style='text-align: center; color: {turn_color};'>{turn_name}</h2>", unsafe_allow_html=True)

    if game.turn == 0:
        if game.phase == GamePhase.DECIDE_CUBE_OR_ROLL:
            col1, col2 = st.columns(2)
            if col1.button("🎲 Roll Dice", use_container_width=True):
                obs, reward, done, _, _ = env.step(0) # 0 = Roll
                st.session_state.logs.append(f"You rolled: {game.current_roll}")
                st.session_state.done = done
                st.rerun()
            
            if game._can_double(0):
                if col2.button("Double Cube", use_container_width=True):
                     obs, reward, done, _, _ = env.step(1) # 1 = Double
                     st.session_state.logs.append("You doubled the cube.")
                     st.session_state.done = done
                     st.rerun()
                     
        elif game.phase == GamePhase.RESPOND_TO_DOUBLE:
            st.error("CPU Doubled! Do you accept?")
            col1, col2 = st.columns(2)
            if col1.button("Take", use_container_width=True):
                obs, reward, done, _, _ = env.step(0) # Take
                st.session_state.logs.append("You accepted the double.")
                st.session_state.done = done
                st.rerun()
            if col2.button("Drop", use_container_width=True):
                 obs, reward, done, _, _ = env.step(1) # Drop
                 st.session_state.logs.append("You dropped the game.")
                 st.session_state.done = done
                 st.rerun()
                 
        elif game.phase == GamePhase.DECIDE_MOVE:
            st.info(f"**Your Roll:** {game.current_roll}")
            moves = game.legal_moves
            
            if not moves:
                st.warning("No legal moves available.")
                if st.button("Pass Turn"):
                     # Auto-pass logic handled by engine or needs dummy step?
                     # The engine should have auto-switched if no moves. 
                     # If we are here, wait, `_roll_and_start_turn` switches if no moves.
                     # So user should NEVER see this state unless logic is weird.
                     # But purely defensive:
                     st.session_state.logs.append("You passed (No moves).")
                     # Force turn switch if engine stuck?
                     # Actually, `step` handles move selection.
                     pass 
            else:
                # Format options for readability
                options = {}
                for i, seq in enumerate(moves):
                    # seq is tuple of ((start, end), ...)
                    display_txt = " -> ".join([f"{s} to {e}" for s, e in seq])
                    options[i] = display_txt
                    
                selected_idx = st.selectbox("Select Move:", options.keys(), format_func=lambda x: options[x])
                
                if st.button("Make Move", type="primary"):
                    st.session_state.logs.append(f"You moved: {options[selected_idx]}")
                    obs, reward, done, _, _ = env.step(selected_idx)
                    st.session_state.done = done
                    st.rerun()
        
    else:
        # CPU Turn
        st.markdown("### Waiting for CPU...")
        
        # Show what phase CPU determines
        # If CPU needs to ROLL, it happens automatically in previous step transition?
        # No, `step` returns. 
        # `game.phase` tells us what CPU is facing.
        
        step_needed = True
        
        # Difficulty Selector (Only for CPU)
        difficulty = st.radio("Engine Strength", ["1-Ply (Fast)", "2-Ply (Grandmaster)"], index=0, horizontal=True)
        depth = 1 if "1-Ply" in difficulty else 2
        
        if st.button("▶ Run CPU Move", type="primary"):
             # AI Step
             current_phase = game.phase
             
             action = 0
             
             if st.session_state.agent:
                 # Use Search Agent
                 # Pass `game` object directly
                 # Note: get_action expects 'game'
                 st.caption(f"Thinking ({difficulty})...")
                 action = st.session_state.agent.get_action(game, depth=depth)
                 
                 # Handling None? get_action returns None if no moves, but game logic shouldn't be here if no moves.
                 if action is None: action = 0 
             else:
                 # Random
                 moves = game.legal_moves
                 if moves:
                     action = np.random.randint(0, len(moves))
                 else:
                     action = 0
             
             # Log Logic
             if current_phase == GamePhase.DECIDE_CUBE_OR_ROLL:
                 if action == 0: 
                     st.session_state.logs.append("CPU rolled dice.")
                 else:
                     st.session_state.logs.append("CPU Doubled!")
                     
             elif current_phase == GamePhase.DECIDE_MOVE:
                 st.session_state.logs.append(f"CPU is playing roll: {game.current_roll}")
                 if action < len(game.legal_moves):
                      seq = game.legal_moves[action]
                      display_txt = " -> ".join([f"{s} to {e}" for s, e in seq])
                      st.session_state.logs.append(f"CPU Moved: {display_txt}")
                 else:
                      st.session_state.logs.append(f"CPU Action Index: {action}")

             obs, reward, done_ep, _, _ = env.step(action)
             st.session_state.done = done_ep
             st.rerun()

else:
    st.balloons()
    st.balloons()
    
    # Calculate points from this game
    pts_0 = game.score[0]
    pts_1 = game.score[1]
    
    winner = "You" if pts_0 > pts_1 else "CPU"
    pts = max(pts_0, pts_1)
    
    st.success(f"Game Over! Winner: {winner} ({pts} points)")
    
    if st.button("Play Next Game"):
        # Update Session Score
        st.session_state.total_score[0] += pts_0
        st.session_state.total_score[1] += pts_1
        
        env.reset()
        st.session_state.done = False
        st.rerun()
