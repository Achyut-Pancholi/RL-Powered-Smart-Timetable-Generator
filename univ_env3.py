import gymnasium as gym
from gymnasium import spaces
import numpy as np

class UniversitySchedulerEnv(gym.Env):
    """
    Advanced Environment with:
    1. Instructor Availability Constraints
    2. Room Capacity Logic
    """
    def __init__(self, config, courses, n_instructors, room_capacities, blocked_slots):
        super(UniversitySchedulerEnv, self).__init__()

        # --- Configuration ---
        self.n_days = config['n_days']
        self.n_slots_per_day = config['n_slots']
        self.n_rooms = config['n_rooms']
        self.total_time_slots = self.n_days * self.n_slots_per_day
        
        self.courses = courses
        self.n_courses = len(self.courses)
        self.n_instructors = n_instructors
        
        # New Features
        self.room_capacities = room_capacities # List of ints, e.g. [30, 60, 40]
        self.blocked_slots = blocked_slots # List of (instructor_id, time_slot_idx)

        # --- OPTIMIZED ACTION SPACE ---
        # Action = Pick a TimeSlot index
        self.action_space = spaces.Discrete(self.total_time_slots)

        # --- Observation Space ---
        grid_size = self.total_time_slots * self.n_rooms
        instructor_grid_size = self.total_time_slots * self.n_instructors
        
        self.observation_space = spaces.Dict({
            "schedule_grid": spaces.Box(low=0, high=1, shape=(grid_size,), dtype=np.int8),
            "instructor_grid": spaces.Box(low=0, high=1, shape=(instructor_grid_size,), dtype=np.int8),
            "current_course_idx": spaces.Box(low=0, high=self.n_courses, shape=(1,), dtype=np.int32)
        })

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.schedule_grid = np.zeros(self.total_time_slots * self.n_rooms, dtype=np.int8)
        self.instructor_grid = np.zeros(self.total_time_slots * self.n_instructors, dtype=np.int8)
        
        # --- APPLY HARD CONSTRAINTS (BLOCKED SLOTS) ---
        # Pre-fill the instructor grid with 1s where they are unavailable
        for (inst_id, slot_idx) in self.blocked_slots:
            if 0 <= slot_idx < self.total_time_slots:
                idx = slot_idx * self.n_instructors + inst_id
                self.instructor_grid[idx] = 1

        self.current_course_index = 0
        self.assignments = [] 
        return self._get_obs(), {}

    def _get_obs(self):
        return {
            "schedule_grid": self.schedule_grid.copy(),
            "instructor_grid": self.instructor_grid.copy(),
            "current_course_idx": np.array([self.current_course_index], dtype=np.int32)
        }

    def step(self, action_time_slot):
        current_course = self.courses[self.current_course_index]
        instructor_id = current_course['instructor_id']
        required_size = current_course.get('size', 0) # Get class size

        reward = 0
        terminated = False
        truncated = False
        info = {}

        # --- Logic 1: Instructor Conflict Check ---
        # Checks if instructor is teaching OR BLOCKED by user
        if self.instructor_grid[action_time_slot * self.n_instructors + instructor_id] == 1:
            reward -= 15 # Higher penalty for ignoring constraints
            info['error'] = "Instructor Unavailable"
        
        else:
            # --- Logic 2: Find a Suitable Room (Capacity Check) ---
            found_room = -1
            
            # Try to find a room that is empty AND big enough
            # We iterate through rooms sorted by capacity to pick the "best fit" if possible, 
            # but for simplicity, we just take the first valid one.
            for r in range(self.n_rooms):
                # Check Capacity First
                if self.room_capacities[r] < required_size:
                    continue # Room too small!
                
                # Check Availability
                idx = action_time_slot * self.n_rooms + r
                if self.schedule_grid[idx] == 0:
                    found_room = r
                    break
            
            if found_room == -1:
                # No suitable rooms (either full or too small)
                reward -= 10
                info['error'] = "No Room (Capacity/Space)"
            else:
                # --- SUCCESS ---
                reward += 10 # Higher reward for satisfying capacity
                
                # Mark Instructor Busy
                self.instructor_grid[action_time_slot * self.n_instructors + instructor_id] = 1
                
                # Mark Room Busy
                grid_idx = action_time_slot * self.n_rooms + found_room
                self.schedule_grid[grid_idx] = 1
                
                # Record Data
                day = action_time_slot // self.n_slots_per_day
                slot = action_time_slot % self.n_slots_per_day
                
                self.assignments.append({
                    "Course": current_course['name'],
                    "Instructor": current_course['instructor_name'],
                    "Day": day + 1,
                    "Slot": slot + 1,
                    "Room": f"Room {found_room + 1} (Cap: {self.room_capacities[found_room]})"
                })
                
                self.current_course_index += 1

        if self.current_course_index >= self.n_courses:
            reward += 50
            terminated = True
            info['success'] = True
        elif reward < -100: # Early stopping if very bad
            terminated = True

        return self._get_obs(), reward, terminated, truncated, info