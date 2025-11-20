# RL-Powered-Smart-Timetable-Generator

## Access Project Link
https://rl-time-table-generator-44.streamlit.app/

---
## 📚 Project Overview

An **intelligent university timetable scheduler** powered by **Deep Reinforcement Learning (PPO)** that automatically assigns courses to time slots and rooms while satisfying hard constraints like instructor availability and room capacity.

- **Language**: Python 3.10+
- **Framework**: Streamlit (UI) + Stable Baselines 3 (RL) + Gymnasium (Environment)
- **Status**: Production-ready Capstone Edition

---

## 🎯 Key Features

✅ **Advanced Constraint Satisfaction**
- Instructor availability constraints
- Room capacity matching
- Automatic conflict resolution

✅ **Interactive Streamlit Interface**
- Upload course data via CSV
- Real-time training visualization
- Configure schedules dynamically

✅ **Deep Reinforcement Learning**
- PPO (Proximal Policy Optimization) agent training
- Multi-input neural network policies
- Live learning curve monitoring

✅ **Smart Room Assignment**
- Capacity-aware room allocation
- Best-fit room selection logic
- Prevents over-booking

---

## 📋 Core Components

### **1. `app_v3.py` - Streamlit Web Application**

**Purpose**: Interactive web interface for timetable generation and constraint configuration.

**Key Features**:
- **Tab 1 - Configuration**: 
  - Upload course CSV file
  - Define number of days/week and slots/day
  - Configure room capacities
  - Set instructor availability constraints (block specific professors from specific time slots)
  
- **Tab 2 - Training & Generation**:
  - Train PPO agent on custom environment with specified constraints
  - Auto-generate timetables via trained model
  - Iterative optimization with 25 max attempts
  
- **Tab 3 - Visualization**:
  - Plot learning curves (training rewards over time)
  - Display final timetable as pivot table (rows: day+time, cols: room)
  - Download schedule as CSV

**Main Classes**:
```python
StreamlitCallback(BaseCallback)
  - _on_step(): Log mean rewards every 1000 timesteps
  - plot_graph(): Real-time matplotlib visualization on Streamlit

Helper Functions:
  - get_time_label(slot_index, start_time, duration_mins): 
    Converts slot indices to readable time ranges (e.g., "09:30 - 10:30")
```

**Workflow**:
1. User uploads CSV with courses and instructors
2. Configures rooms, days, time slots, and instructor constraints
3. Clicks "Start Training" → PPO learns for N steps
4. Clicks "Generate" → Model predicts optimal schedule (up to 25 attempts)
5. View/download final timetable

**Code Highlights**:
- Uses `gymnasium` MultiInputPolicy for multi-part state observation
- Implements custom callback to update Streamlit UI during training
- Session state manages training history and generated schedules
- Time slot conversion via `get_time_label()` for user-friendly display

---

### **2. `univ_env3.py` - Custom Gymnasium Environment**

**Purpose**: RL environment that models university scheduling as a Markov Decision Process (MDP).

**Environment Design**:
- **State (Observation)**:
  - `schedule_grid`: 1D array tracking room availability (size: n_days × n_slots × n_rooms)
  - `instructor_grid`: 1D array tracking instructor availability (size: n_days × n_slots × n_instructors)
  - `current_course_idx`: Index of course being scheduled
  
- **Action Space**: Discrete(total_time_slots)
  - Agent picks a time slot for the current course
  - Room selection is automatic (best-fit algorithm)
  
- **Reward Shaping**:
  - **+10**: Successfully assign course to valid room (capacity & availability met)
  - **-15**: Instructor unavailable (ignoring constraints)
  - **-10**: No suitable room (all full OR too small)
  - **+50**: Complete schedule (all courses assigned)
  - **Early termination**: If cumulative reward < -100

**Constraint Handling**:
```python
Hard Constraints (Pre-applied at reset):
  - blocked_slots: List of (instructor_id, time_slot) tuples
  - Sets instructor_grid[slot * n_instructors + inst_id] = 1

Soft Constraints (Via Rewards):
  - Room capacity: required_size <= room_capacities[room]
  - Instructor conflict: Checks instructor_grid[slot * n_instructors + inst_id] == 0
  - Room occupancy: Checks schedule_grid[slot * n_rooms + room] == 0
```

**Key Methods**:
```python
reset(seed=None, options=None):
  - Initialize schedule_grid & instructor_grid with zeros
  - Apply hard constraints (blocked_slots)
  - Return initial observation & empty info dict

step(action_time_slot):
  - Get current course & instructor
  - Check instructor availability
  - Find best-fit room via capacity matching
  - Update grids & record assignment
  - Return: obs, reward, terminated, truncated, info

_get_obs():
  - Returns dict with schedule_grid, instructor_grid, current_course_idx
```

**Best-Fit Room Selection Logic**:
```python
for r in range(self.n_rooms):
  if room_capacities[r] < required_size:
    continue  # Skip undersized rooms
  if schedule_grid[slot * n_rooms + r] == 0:
    found_room = r  # Found first available room with capacity
    break
```

---

### **3. `requirements.txt` - Dependencies**

```
gymnasium==0.29.1              # RL environment API standard
stable-baselines3==2.3.2       # PPO & other RL algorithms
numpy>=1.26.0                  # Numerical computing
shimmy>=1.3.0                  # Gymnasium compatibility layer
streamlit                      # Web framework for UI
pandas                         # Data manipulation & CSV handling
matplotlib                     # Plotting & visualization
```

**Why Each Package**:
- `gymnasium`: Industry-standard RL environment interface (Gym replacement)
- `stable-baselines3`: Pre-optimized PPO implementation with proven convergence
- `numpy/pandas`: Data processing for schedules, constraints & CSV I/O
- `streamlit`: Rapid deployment of interactive dashboard with minimal code
- `matplotlib`: Real-time training visualization & schedule plots
- `shimmy`: Ensures compatibility between newer Gymnasium & SB3 versions

---

## 📊 Data Format

### Input CSV Structure (`data.csv`)

```csv
Course,Instructor,No. of Students

```

**Column Details**:
| Column | Type | Required | Description |
|--------|------|----------|-------------|
| Course | string | ✓ | Course code + name (e.g., "CSE2701 - Big Data") |
| Instructor | string | ✓ | Professor name (mapped to integer ID automatically) |
| Size | integer | ✓ | Expected class enrollment (used for room capacity matching) |

**Notes**:
- Instructor names must be consistent (spelling/capitalization)
- Size should reflect actual class enrollment for realistic room allocation
- CSV header is required

### Output Schedule Format

Generated timetable rows:
```json
{
  "Course": "Big Data Analytics (CSE2701)",
  "Instructor": "Name of Instructor",
  "Day": 1,              # 1=Monday, 2=Tuesday, ..., 5=Friday
  "Slot": 2,            # 1-based slot index (1 = first slot of day)
  "Room": "Room 1 (Cap: 60)"  # Assigned room + capacity
}
```

**Pivot Table Format** (displayed in app):
```
              Room 1  Room 2  Room 3  Room 4
Mon 09:30     CSE2701   -     -      CSE2741
Mon 10:30     CSE2702   -     -        -
Tue 09:30       -     CSE2732  -      CSE2731
...
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Virtual environment (recommended)

### Installation

```powershell
# Create & activate venv
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### Running the Application

```powershell
# Activate venv (if not already active)
.\venv\Scripts\Activate.ps1

# Launch Streamlit app
streamlit run app_v3.py
```

Application opens at: `http://localhost:8501`

---

## 🧠 How It Works

### 1. **Environment Setup**
   - User provides course data (CSV) & constraint configuration
   - `UniversitySchedulerEnv` initializes with dynamic configuration
   - Hard constraints (blocked instructor slots) pre-applied to instructor_grid

### 2. **Agent Training**
   - PPO agent learns to map multi-part observations → optimal time slot actions
   - Reward function guides towards constraint satisfaction & complete scheduling
   - Live callback plots training progress every 1000 steps
   - Convergence typically after 20K-50K timesteps

### 3. **Schedule Generation**
   - Trained model predicts actions sequentially for each course
   - Best-fit room selection ensures capacity constraints met
   - Up to 25 optimization attempts for best solution
   - Terminates early if feasible schedule found

### 4. **Visualization**
   - Learning curve shows convergence of mean reward over training steps
   - Final schedule displayed as human-readable pivot table
   - Downloadable CSV export for import into other systems

---

## 📈 Example Scenario

**Setup**:
- 5 days/week, 7 slots/day = 35 time slots
- 4 rooms with capacities: [30, 60, 40, 50]
- 7 courses (some with multiple sessions), 6 instructors
- Block Prof. XYZ from Friday slots 2-4

**Process**:
1. Upload `data.csv` → App loads 7 courses
2. Configure: 5 days, 7 slots, 4 rooms
3. Block XYZ on Friday for slots 2-4
4. Click "Train" → PPO runs 30,000 steps
5. Learning curve shows reward increasing from ~-50 to +100
6. Click "Generate" → Attempts to schedule all courses
7. Success on attempt 3 → All courses assigned without conflicts

**Output**:
```
Timetable:
         Room 1 (30)    Room 2 (60)         Room 3 (40)    Room 4 (50)
Mon 9:30    CSE2701    Big Data (40)       CSE2741        -
Mon 10:30   CSE2702    Cloud Comp(50)      -              -
Tue 9:30    -          CSE2732 (60 max)    -              CSE2731
Fri 9:30    -          -                   DevOps(40)     CSE2747
Fri 10:30   -          -                   (XYZ blocked)  -
Fri 1:30    CSE2701    -                   -              CI/CD
```

All constraints satisfied! Schedule downloadable as CSV.

---

## 🔧 Customization

### Modify Training Parameters

In `app_v3.py`, line ~180:
```python
training_steps = st.slider("Steps", 10000, 100000, 30000)  # Default: 30K
```
Increase for harder problems with more courses/constraints.

### Adjust Reward Shaping

In `univ_env3.py`, modify `step()` method:
```python
reward += 10   # Successfully assigned → increase for harder prioritization
reward -= 15   # Instructor conflict → increase to heavily penalize violations
reward += 50   # Complete schedule → increase to prioritize full solutions
```

### Add New Constraints

Example - prevent back-to-back assignments:
```python
# In step() method:
if self.current_course_index > 0:
    last_assignment = self.assignments[-1]
    if last_assignment['Slot'] == slot and last_assignment['Day'] == day:
        reward -= 5  # Penalize back-to-back in same slot
```

---

## ⚙️ Architecture Details

### State Representation
- **schedule_grid** (flattened 3D):
  - Dimension: [n_days × n_slots × n_rooms]
  - Value: 0 (available) or 1 (occupied)
  - Used for room conflict detection

- **instructor_grid** (flattened 2D):
  - Dimension: [n_days × n_slots × n_instructors]
  - Value: 0 (available) or 1 (teaching/blocked)
  - Used for instructor conflict detection

- **current_course_idx**:
  - Scalar: which course is being scheduled (0 to n_courses-1)

### Action-to-Room Mapping
```
User Action: Select time_slot (0 to total_time_slots-1)
  ↓
Environment finds best-fit room:
  - Filter by capacity: room_cap >= class_size
  - Check availability: schedule_grid[time_slot, room] == 0
  - Allocate: Mark both room AND instructor as busy
```

### Episode Termination
- **Success**: All courses assigned (current_course_index >= n_courses)
- **Failure**: Cumulative reward < -100 (agent giving up)
- **Natural end**: Max timesteps hit (Streamlit timeout)

---

## 📝 License

This project is provided as-is for educational & research purposes.

---

## 👤 Author & Credits

**Capstone Project** - Advanced Constraint Satisfaction using Deep Reinforcement Learning (DRL)

- Framework: Stable Baselines 3 (PPO)
- Environment API: Gymnasium
- UI: Streamlit
- Optimization: Multi-input neural network policies

---

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| "ModuleNotFoundError: gymnasium" | Run `pip install -r requirements.txt` in active venv |
| Streamlit shows blank screen | Check CSV format has "Course" & "Instructor" columns |
| Training very slow | Reduce training_steps slider or simplify problem (fewer courses) |
| "No suitable rooms found" warning | Add more rooms or increase room capacities in config |
| Shape mismatch errors | Ensure Python 3.10+, gymnasium 0.29.1, stable-baselines3 2.3.2 |

---

## 🚀 Next Steps & Improvements

- [ ] Add student preferences (preferred time slots)
- [ ] Implement room features (lab equipment, whiteboard, etc.)
- [ ] Add exam scheduling mode
- [ ] Support load balancing (even distribution of classes)
- [ ] Multi-objective optimization (minimize room fragmentation)
- [ ] Unit tests for environment constraints
- [ ] Performance benchmarking across RL algorithms
