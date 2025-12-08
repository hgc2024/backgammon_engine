import streamlit as st
import numpy as np
import torch
import time
import sys
import os
import streamlit.components.v1 as components

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
        
        # Load Model (TD-Gammon)
        model_path = "td_backgammon_best.pth"
        if os.path.exists(model_path):
            st.session_state.agent = ExpectiminimaxAgent(model_path, device="cuda" if torch.cuda.is_available() else "cpu")
            st.toast("TD-Gammon Engine Loaded!", icon="🧠")
        else:
            st.session_state.agent = None
            st.toast("No Model Found. Playing vs Random.", icon="🎲")
            
    if 'total_score' not in st.session_state:
        st.session_state.total_score = [0, 0]
    
    if 'logs' not in st.session_state:
        st.session_state.logs = []

init_state()

game = st.session_state.game
env = st.session_state.env

st.title("Backgammon RL Engine")

# --- Sidebar ---
with st.sidebar:
    st.header("Game Status")
    
    # Show Session Score
    ts = st.session_state.total_score
    current = game.score
    st.metric("Session Score (You - CPU)", f"{ts[0] + current[0]} - {ts[1] + current[1]}")
    
    st.metric("Cube Value", f"{game.cube_value}")
    st.metric("Cube Owner", "Cpu" if game.cube_owner == 1 else ("You" if game.cube_owner == 0 else "Centered"))
    
    # Difficulty Selector
    difficulty = st.radio("Engine Strength", ["1-Ply (Fast)", "2-Ply (Grandmaster)"], index=0, horizontal=True)
    st.session_state.depth = 1 if "1-Ply" in difficulty else 2
    
    st.divider()
    
    # Starting Player
    starter = st.radio("Who Starts?", ["You (White)", "CPU (Red)", "Random"], index=2, horizontal=True)
    
    if "You" in starter:
        start_idx = 0
    elif "CPU" in starter:
        start_idx = 1
    else:
        start_idx = np.random.randint(0, 2)
    
    if st.button("Reset / Start Match", key="reset_match_sidebar"):
        env.reset(options={'starting_player': start_idx})
        st.session_state.done = False
        st.session_state.logs = []
        st.session_state.total_score = [0, 0]
        st.rerun()
    st.subheader("Action Log")
    
    for log in reversed(st.session_state.logs[-10:]):
        st.caption(log)

# --- Board Visualization ---
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
            st.info(f"**Your Roll:** {game.dice}") # Use .dice to show remaining
            
            # 1. Get ALL legal partial moves
            # List of (start, end)
            partial_moves = game.get_legal_partial_moves()
            
            if not partial_moves:
                st.warning("No legal moves available.")
                if st.button("Pass Turn"):
                     # Force turn switch if engine handled it or just refresh
                     # If get_legal_partial_moves is empty but dice exist, step_partial logic 
                     # should have auto-switched turn? 
                     # self.dice check in step_partial handles it.
                     # But if we just STARTED turn and have no moves?
                     # _roll_and_start_turn handles initial block.
                     # So we should only be here if we made 1 move and are now blocked.
                     
                     # Manually force turn switch
                     game.turn = 1 - game.turn
                     game.phase = GamePhase.DECIDE_CUBE_OR_ROLL
                     st.session_state.logs.append("You passed.")
                     st.rerun()
            else:
                # 2. Interactive Selection State
                if 'selected_source' not in st.session_state:
                    st.session_state.selected_source = None
                    
                # Group moves by Source
                sources = sorted(list(set([m[0] for m in partial_moves])), key=lambda x: (x if isinstance(x, int) else -1))
                
                # If source selected, show destinations
                if st.session_state.selected_source is not None:
                    src = st.session_state.selected_source
                    
                    # Verify source still valid (in case dice changed?)
                    valid_dests = [m[1] for m in partial_moves if m[0] == src]
                    
                    if not valid_dests:
                        st.session_state.selected_source = None
                        st.rerun()
                    
                    st.markdown(f"**Selected Checkers at: {src}**")
                    
                    cols = st.columns(len(valid_dests) + 1)
                    with cols[0]:
                        if st.button("❌ Cancel Selection"):
                            st.session_state.selected_source = None
                            st.rerun()
                            
                    for i, dst in enumerate(valid_dests):
                        with cols[i+1]:
                            lbl = "Bear Off" if dst == 'off' else f"To {dst}"
                            if st.button(lbl, key=f"move_{src}_{dst}"):
                                # Execute Partial Step
                                st.session_state.logs.append(f"You moved {src} -> {dst}")
                                pts, winner, done = game.step_partial((src, dst))
                                st.session_state.selected_source = None # Reset selection
                                st.session_state.done = done
                                st.rerun()
                else:
                    # Show valid Sources
                    st.markdown("**Select Checker to Move:**")
                    
                    # Create columns for buttons (wrap if many)
                    # Simple list for now
                    cols = st.columns(min(len(sources), 8))
                    for i, src in enumerate(sources):
                        col_idx = i % 8
                        with cols[col_idx]:
                            lbl = "Bar" if src == 'bar' else f"Pt {src}"
                            if st.button(lbl, key=f"src_{src}"):
                                st.session_state.selected_source = src
                                st.rerun()
        
    else:
        # CPU Turn
        st.markdown("### CPU is thinking...")
        
        # UX Delay so it's not instant
        time.sleep(0.8)
        
        # AI Step
        current_phase = game.phase
        depth = st.session_state.get('depth', 1)
        
        action = 0
             
        if st.session_state.agent:
             # Check Phase
             if current_phase == GamePhase.DECIDE_CUBE_OR_ROLL:
                 # Agent has no Cube Logic -> Always Roll
                 action = 0
             else:
                 # st.caption(f"Thinking...") # No need if auto
                 action = st.session_state.agent.get_action(game, depth=depth)
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
             st.session_state.logs.append(f"CPU is playing roll: {game.dice}")
             if action < len(game.legal_moves):
                  seq = game.legal_moves[action]
                  display_txt = " -> ".join([f"{s} to {e}" for s, e in seq])
                  st.session_state.logs.append(f"CPU Moved: {display_txt}")
                  
                  # Debug: Show Win Prob
                  if hasattr(st.session_state.agent, 'last_value'):
                      val = st.session_state.agent.last_value
                      # Tanh (-1 to 1) -> Prob (0 to 1 scale roughly) or just value
                      # Or if trained with Win (+1) / Loss (-1), 
                      # Win Prob approx (val + 1) / 2
                      prob = (val + 1) / 2
                      st.session_state.logs.append(f"CPU Confidence: {prob*100:.1f}% Win")
             else:
                  st.session_state.logs.append(f"CPU Action Index: {action}")

        obs, reward, done_ep, _, _ = env.step(action)
        st.session_state.done = done_ep
        st.rerun()

else:
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
        
        # Respect the "Who Starts?" selection from sidebar
        env.reset(options={'starting_player': start_idx})
        st.session_state.done = False
        st.rerun()
