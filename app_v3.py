import streamlit as st
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from univ_env3 import UniversitySchedulerEnv

st.set_page_config(page_title="AI Capstone Scheduler", layout="wide")

# --- Session State Init ---
if 'training_history' not in st.session_state:
    st.session_state['training_history'] = {'steps': [], 'rewards': []}
if 'last_schedule' not in st.session_state:
    st.session_state['last_schedule'] = None

# --- Custom Callback ---
class StreamlitCallback(BaseCallback):
    def __init__(self, chart_placeholder, info_placeholder, verbose=0):
        super(StreamlitCallback, self).__init__(verbose)
        self.chart_placeholder = chart_placeholder
        self.info_placeholder = info_placeholder

    def _on_step(self) -> bool:
        if self.num_timesteps % 1000 == 0:
            if len(self.model.ep_info_buffer) > 0:
                mean_reward = np.mean([ep['r'] for ep in self.model.ep_info_buffer])
                
                # Save to Session State
                st.session_state['training_history']['steps'].append(self.num_timesteps)
                st.session_state['training_history']['rewards'].append(mean_reward)
                
                # Live Plotting
                self.plot_graph(st.session_state['training_history']['steps'], st.session_state['training_history']['rewards'])
                self.info_placeholder.metric("Mean Reward", f"{mean_reward:.2f}")
        return True
    
    def plot_graph(self, steps, rewards):
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(steps, rewards, color='#00ff41', linewidth=2)
        ax.set_xlabel('Steps', color='white')
        ax.set_ylabel('Reward', color='white')
        ax.set_title('Agent Learning Curve', color='white')
        ax.tick_params(colors='white')
        ax.set_facecolor('#0e1117')
        fig.patch.set_facecolor('#0e1117')
        ax.grid(color='#444444', linestyle='--')
        self.chart_placeholder.pyplot(fig)
        plt.close(fig)

def get_time_label(slot_index, start_time, duration_mins):
    start_dt = datetime.combine(datetime.today(), start_time)
    slot_start = start_dt + timedelta(minutes=slot_index * duration_mins)
    slot_end = slot_start + timedelta(minutes=duration_mins)
    return f"{slot_start.strftime('%H:%M')} - {slot_end.strftime('%H:%M')}"

# --- Main UI ---
st.title("🎓 AI University Scheduler (Capstone Edition)")
st.markdown("Advanced Constraint Satisfaction using Deep Reinforcement Learning.")

# --- SIDEBAR ---
st.sidebar.header("1. Setup & Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if not uploaded_file:
    st.info("Please upload 'university_courses.csv'.")
    st.stop()

df = pd.read_csv(uploaded_file)
if 'Size' not in df.columns:
    df['Size'] = 40 # Default
st.sidebar.success(f"Loaded {len(df)} classes.")

st.sidebar.header("2. Time Settings")
n_days = st.sidebar.slider("Days/Week", 1, 7, 5)
n_slots = st.sidebar.slider("Slots/Day", 1, 10, 7)
start_time_str = st.sidebar.time_input("Start Time", value=datetime.strptime("09:30", "%H:%M").time())
slot_duration_mins = st.sidebar.number_input("Slot Mins", value=60)
day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
selected_days = day_names[:n_days]

# --- TABS ---
tab1, tab2, tab3 = st.tabs(["⚙️ Configuration", "🧠 Train & Generate", "📊 Visualization"])

# --- TAB 1: CONFIG ---
with tab1:
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("🏗️ Room Configuration")
        n_rooms = st.number_input("Total Rooms", 1, 10, 4)
        room_capacities = []
        for i in range(n_rooms):
            cap = st.slider(f"Capacity Room {i+1}", 10, 100, 60, key=f"room_{i}")
            room_capacities.append(cap)

    with col2:
        st.subheader("⛔ Instructor Constraints")
        st.info("Block professors from teaching at specific times.")
        unique_instructors = df['Instructor'].unique()
        instructor_map = {name: i for i, name in enumerate(unique_instructors)}
        
        blocked_prof = st.selectbox("Select Professor to Block", unique_instructors)
        blocked_day = st.selectbox("Select Day", selected_days)
        blocked_time_ranges = [f"Slot {i+1} ({get_time_label(i, start_time_str, slot_duration_mins)})" for i in range(n_slots)]
        blocked_times = st.multiselect("Select Busy Slots", blocked_time_ranges)
        
        blocked_slots_indices = []
        if blocked_prof and blocked_day and blocked_times:
            day_idx = selected_days.index(blocked_day)
            inst_id = instructor_map[blocked_prof]
            for t_str in blocked_times:
                slot_idx_in_day = int(t_str.split()[1]) - 1
                global_slot = day_idx * n_slots + slot_idx_in_day
                blocked_slots_indices.append((inst_id, global_slot))
                
        if blocked_slots_indices:
            st.warning(f"Blocking {len(blocked_slots_indices)} slots for {blocked_prof}")

# --- PREPARE DATA ---
n_instructors = len(unique_instructors)
course_data = []
for _, row in df.iterrows():
    course_data.append({
        'name': row['Course'],
        'instructor_name': row['Instructor'],
        'instructor_id': instructor_map[row['Instructor']],
        'size': row['Size']
    })

config = {'n_days': n_days, 'n_slots': n_slots, 'n_rooms': n_rooms}

# --- TAB 2: TRAINING & GENERATION ---
with tab2:
    c1, c2 = st.columns(2)
    
    with c1:
        st.header("1. Train Agent")
        training_steps = st.slider("Steps", 10000, 100000, 30000)
        if st.button("Start Training"):
            # Reset History
            st.session_state['training_history'] = {'steps': [], 'rewards': []}
            
            chart = st.empty()
            info = st.empty()
            
            env = UniversitySchedulerEnv(config, course_data, n_instructors, room_capacities, blocked_slots_indices)
            st_cb = StreamlitCallback(chart, info)
            
            model = PPO("MultiInputPolicy", env, verbose=0, seed=42)
            model.learn(total_timesteps=training_steps, callback=st_cb)
            
            st.session_state['model'] = model
            st.success("Training Complete! Go to 'Visualization' tab to see results.")

    with c2:
        st.header("2. Generate")
        if st.button("Generate Timetable"):
            if 'model' not in st.session_state:
                st.error("Train first!")
            else:
                env = UniversitySchedulerEnv(config, course_data, n_instructors, room_capacities, blocked_slots_indices)
                model = st.session_state['model']
                
                success = False
                best_schedule = None
                status = st.empty()
                progress = st.progress(0)
                
                for attempt in range(25):
                    status.text(f"Optimization Attempt {attempt+1}...")
                    obs, _ = env.reset()
                    done = False
                    while not done:
                        action, _ = model.predict(obs, deterministic=False)
                        obs, _, terminated, truncated, info = env.step(action)
                        done = terminated or truncated
                        if 'success' in info and info['success']:
                            success = True
                            best_schedule = env.assignments
                            break
                    progress.progress((attempt + 1) * 4)
                    if success: break
                
                status.empty()
                progress.empty()
                
                if success:
                    st.balloons()
                    st.success(f"Solved Constraints on Attempt {attempt+1}!")
                    st.session_state['last_schedule'] = best_schedule
                else:
                    st.error("Could not satisfy all constraints. Try increasing training steps.")

# --- TAB 3: VISUALIZATION ---
with tab3:
    st.header("📊 Analysis & Results")
    
    v1, v2 = st.columns(2)
    
    with v1:
        st.subheader("Learning Curve")
        if st.session_state['training_history']['steps']:
            # Re-plot from session state
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(st.session_state['training_history']['steps'], st.session_state['training_history']['rewards'], color='#00ff41', linewidth=2)
            ax.set_xlabel('Training Steps')
            ax.set_ylabel('Mean Reward')
            ax.grid(True, linestyle='--', alpha=0.6)
            st.pyplot(fig)
        else:
            st.info("Train the model to see the learning curve.")
            
    with v2:
        st.subheader("Final Timetable")
        if st.session_state['last_schedule']:
            res = pd.DataFrame(st.session_state['last_schedule'])
            
            # Formatting
            res['Day Name'] = res['Day'].apply(lambda d: selected_days[d-1])
            res['Time'] = res['Slot'].apply(lambda s: get_time_label(s-1, start_time_str, slot_duration_mins))
            
            # Sort Order
            res['Day Name'] = pd.Categorical(res['Day Name'], categories=selected_days, ordered=True)
            
            # Pivot
            pivot = res.pivot_table(
                index=['Day Name', 'Time'], columns='Room', values='Course', aggfunc=lambda x: ' | '.join(x)
            ).fillna("-")
            
            st.dataframe(pivot, use_container_width=True)
            st.download_button("Download CSV", res.to_csv(), "schedule.csv")
        else:
            st.info("Generate a schedule to see the timetable.")