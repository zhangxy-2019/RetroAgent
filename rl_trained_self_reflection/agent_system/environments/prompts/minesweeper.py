# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

MINESWEEPER_TEMPLATE_NO_HIS = """
You are an expert agent operating in the Minesweeper game.
You will be given a two dimensional {board_size} by {board_size} board, with {n_mines} hidden mines. 
The rows and columns are indexed from 1 to {board_size}.

# Cell States
- Unopened cells (?): cells that are yet to be revealed and may contain a mine.
- Blank cells (.): opened and non-mine cells, and they have no neighboring mines
- Numbered cells (1-8): opened and non-mine cells, and the number indicates how many mines are in the eight neighboring cells, including those diagonally adjacent. For example, a cell with a `8' means all its neighboring cells contain mines.
- Mine cells (*): opened cells that contain a mine.

# Your Goal
Your goal is to clear the board by revealing all the cells that don't contain mines, without detonating any of the hidden mines scattered throughout the board.
Use clues about the number of neighboring mines in each field to reason about the position of mines and non-mine cells.

# Reveal Rules
Your admissible action is to choose ONE unopened cell (?) to reveal per turn. The outcome depends on the content of that cell:
- Blank cell (.): That cell is revealed, and all contiguous blank cells plus their bordering numbered cells are automatically revealed (auto-cascade).
- Numbered cell (1–8): Only that single cell is revealed, showing the count of neighboring mines.
- Mine (*): The game ends immediately in a loss.

{reflections}

# Current Step
Your current observation is:
{current_observation}
Now it's your turn to make a move.
- Your should first reason step-by-step about the current situation — observe the status of the board, inferring the states of unopened cells (?).  This reasoning process MUST be enclosed within <think> </think> tags.
- Once you've finished your reasoning, you should choose ONE unopened cell (?) to reveal. Put the index of cell in the format of "(row, col)" within the <action> </action> tag.
"""


MINESWEEPER_TEMPLATE = """
You are an expert agent operating in the Minesweeper game.
You will be given a two dimensional {board_size} by {board_size} board, with {n_mines} hidden mines. 
The rows and columns are indexed from 1 to {board_size}.

# Cell States
- Unopened cells (?): cells that are yet to be revealed and may contain a mine.
- Blank cells (.): opened and non-mine cells, and they have no neighboring mines
- Numbered cells (1-8): opened and non-mine cells, and the number indicates how many mines are in the eight neighboring cells, including those diagonally adjacent. For example, a cell with a `8' means all its neighboring cells contain mines.
- Mine cells (*): opened cells that contain a mine.

# Your Goal
Your goal is to clear the board by revealing all the cells that don't contain mines, without detonating any of the hidden mines scattered throughout the board.
Use clues about the number of neighboring mines in each field to reason about the position of mines and non-mine cells.

# Reveal Rules
Your admissible action is to choose ONE unopened cell (?) to reveal per turn. The outcome depends on the content of that cell:
- Blank cell (.): That cell is revealed, and all contiguous blank cells plus their bordering numbered cells are automatically revealed (auto-cascade).
- Numbered cell (1–8): Only that single cell is revealed, showing the count of neighboring mines.
- Mine (*): The game ends immediately in a loss.

{reflections}

# Current Step
Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took: {action_history}
You are now at step {current_step} and your current observation is:
{current_observation}
Now it's your turn to make a move.
- Your should first reason step-by-step about the current situation — observe the status of the board, inferring the states of unopened cells (?).  This reasoning process MUST be enclosed within <think> </think> tags.
- Once you've finished your reasoning, you should choose ONE unopened cell (?) to reveal. Put the index of cell in the format of "(row, col)" within the <action> </action> tag.
"""



# MINESWEEPER_REFLECT_TEMPLATE = """
# You are an expert Minesweeper analyst reviewing game trajectories.
# You will analyze a past Minesweeper game attempt on a two dimensional {board_size} by {board_size} board, with {n_mines} hidden mines. The rows and columns are indexed from 1 to {board_size}.

# # Game Context
# ## Cell States
# - Unopened cells (?): cells that are yet to be revealed and may contain a mine
# - Blank cells (.): opened non-mine cells with no neighboring mines
# - Numbered cells (1-8): opened non-mine cells showing count of neighboring mines (including diagonals)
# - Mine cells (*): opened cells containing a mine

# ## Game Goal
# Clear the board by revealing all cells that don't contain mines, without detonating any hidden mines.

# ## Game Rules
# - One action per turn: choose ONE unopened cell (?) to reveal
# - Revealing a blank cell (.): auto-cascades to reveal all contiguous blank cells plus bordering numbered cells
# - Revealing a numbered cell (1-8): only that cell is revealed
# - Revealing a mine (*): game ends in loss

# The game was {success} completed.

# ## TRAJECTORY OF ACTIONS AND OBSERVATIONS:
# {current_trajectory}

# ## YOUR REFLECTION TASK:

# You will now reflect on this experience following a two-step process:

# ### Step 1: Analysis (Required)
# Provide a detailed analysis within <think> </think> tags that includes:
# 1. Break down the game into atomic subtasks (e.g., "Clear top-left corner", "Resolve 1-2-1 pattern at row 3")
# 2. Check if each subtask was completed based on the observations
# 3. Identify critical deductions made or missed
# 4. Extract the most important lesson from this specific attempt

# ### Step 2: Structured Output (Required)
# After your analysis, provide a JSON reflection within <reflection> </reflection> tags using this EXACT structure:

# {{
#   "subtasks": [
#     {{"name": "subtask_1", "description": "Reveal cell at (1,3) based on '2' constraint", "status": "completed"}},
#     {{"name": "subtask_2", "description": "Deduce mine at (1,2) from '1' at (3,1)", "status": "incomplete"}},
#     {{"name": "subtask_3", "description": "Clear remaining cell at (6,4)", "status": "incomplete"}}
#   ],
#   "task_success": false,
#   "lesson": "Failed to apply constraint from '1' at (6,3) - since (6,4) was the only unopened neighbor, it was guaranteed safe but never revealed"
# }}

# ## FORMAT SPECIFICATIONS:

# **Subtask fields:**
# - name: Sequential identifier (subtask_1, subtask_2, etc.)
# - description: Specific action with grid coordinates (row, column)
# - status: "completed" or "incomplete" (lowercase only)

# **Task_success:** boolean (true/false, not string)

# **Lesson:** Single most important insight containing:
# - Specific grid positions from trajectory
# - Either the critical logic error OR key deduction pattern
# - Actionable information for future attempts
# """


MINESWEEPER_REFLECT_TEMPLATE = """
You are an expert evaluating a Minesweeper game attempt.
Task Requirements: Reveal all non-mine cells on a {board_size}x{board_size} board with {n_mines} mines without detonating any mine.

You have just completed an attempt at this Minesweeper game.
Current Trajectory of the attempt:
{current_trajectory}

<think>
If a reference trajectory exists, compare it with the current trajectory.
Analyze the trajectory to determine if the task was successful:
1. Identify the board constraints (size, mine count) in the 'Target Task'.
2. Examine the final action in the trajectory. Did it result in a mine detonation (loss) or a cleared board (win)?
3. If the game ended without a mine detonation, check if ALL safe cells were revealed.
4. Did the player successfully flag mines (optional but helpful) and reveal all safe spots? (If a mine was hit or safe cells remain hidden, it is a failure).
5. What specific logic or guessing behavior led to this outcome?
6. What are the 1-2 most valuable lessons from this attempt?
7. Devise a concise, new plan of action that accounts for any mistakes with reference to specific actions that should be taken in the next trial

Game notation for reference:
• Cell states: ? (unopened), . (blank/no neighbors), 1-8 (mine count), * (mine)
• Coordinates: rows/columns indexed 1 to {board_size}
• Valid actions: (row, col) where 1 ≤ row,col ≤ {board_size}
• Blank cells auto-cascade to reveal connected blanks + borders

Subtask Completion Criteria (binary evaluation for failed trajectories too):
• valid_moves: COMPLETED if made at least 2 valid format moves; INCOMPLETE if mostly invalid actions
• exploration_progress: COMPLETED if revealed >10% of board; INCOMPLETE if revealed <10%  
• logical_attempt: COMPLETED if attempted any deduction (even if wrong); INCOMPLETE if only random/invalid moves
• error_recovery: COMPLETED if corrected any error within 3 attempts; INCOMPLETE if repeated same errors
• cascade_usage: COMPLETED if triggered or attempted any cascade; INCOMPLETE if only single cell reveals
• systematic_approach: COMPLETED if showed any pattern in move selection; INCOMPLETE if purely random
</think>

**Required JSON Output:**
{{
  "subtasks": [
    {{"name": "valid_moves", "description": "[e.g., 'Made 5 valid moves like (1,1), (2,3)' or 'Only invalid formats like (-1,-1)']", "status": "[completed/incomplete]"}},
    {{"name": "exploration_progress", "description": "[e.g., 'Revealed 15 cells (25% of board)' or 'Only revealed 2 cells']", "status": "[completed/incomplete]"}},
    {{"name": "logical_attempt", "description": "[e.g., 'Tried to use (3,3)=1 constraint' or 'No deduction attempts']", "status": "[completed/incomplete]"}},
    {{"name": "error_recovery", "description": "[e.g., 'Fixed format after 2 attempts' or 'Repeated invalid action 10 times']", "status": "[completed/incomplete]"}},
    {{"name": "cascade_usage", "description": "[e.g., '(1,1) triggered 8-cell cascade' or 'No cascade attempts']", "status": "[completed/incomplete]"}},
    {{"name": "systematic_approach", "description": "[e.g., 'Checked corners first' or 'Random clicking']", "status": "[completed/incomplete]"}}
  ],
  "trajectory_value": [count of completed subtasks out of 6],
  "task_success": [true if successfully cleared all safe cells, false if detonated mine or incomplete],
  "next_priority": "[Most important fix, e.g., 'Use valid (row,col) format' or 'When cell shows 1, count unopened neighbors']"
}}

**Evaluation Rules:**
- **Determine Success Yourself:** You must judge 'task_success' by comparing the final board state in the trajectory to the Target Task.
- **Criteria for Success:** The task is ONLY true if the agent successfully revealed ALL safe cells without detonating a mine.
- **Criteria for Failure:** If the trajectory ends with a mine detonation, or if the agent stopped before revealing all safe cells, 'task_success' is false.
- Each subtask status must reflect actual trajectory events.
- Lessons should explain factors that led to the outcome (logic vs. guessing).
- Reference specific elements from trajectory (coordinates, cell values).
- Use null for lessons only if truly not applicable.

Output ONLY the JSON evaluation.
"""


# MINESWEEPER_REFLECT_TEMPLATE = """
# You are an expert evaluating a Minesweeper game attempt.
# Task Requirements: Reveal all non-mine cells on a {board_size}x{board_size} board with {n_mines} mines without detonating any mine.

# You have just completed an attempt at this Minesweeper game.

# {reference_trajectory}

# Current Trajectory of the attempt:
# {current_trajectory}

# <think>
# If a reference trajectory exists, compare it with the current trajectory.
# Analyze the trajectory to determine if the task was successful:
# 1. Identify the board constraints (size, mine count) in the 'Target Task'.
# 2. Examine the final action in the trajectory. Did it result in a mine detonation (loss) or a cleared board (win)?
# 3. If the game ended without a mine detonation, check if ALL safe cells were revealed.
# 4. Did the player successfully flag mines (optional but helpful) and reveal all safe spots? (If a mine was hit or safe cells remain hidden, it is a failure).
# 5. What specific logic or guessing behavior led to this outcome?
# 6. What are the 1-2 most valuable lessons from this attempt?
# 7. Devise a concise, new plan of action that accounts for any mistakes with reference to specific actions that should be taken in the next trial

# Game notation for reference:
# • Cell states: ? (unopened), . (blank/no neighbors), 1-8 (mine count), * (mine)
# • Coordinates: rows/columns indexed 1 to {board_size}
# • Valid actions: (row, col) where 1 ≤ row,col ≤ {board_size}
# • Blank cells auto-cascade to reveal connected blanks + borders

# Subtask Completion Criteria (binary evaluation for failed trajectories too):
# • valid_moves: COMPLETED if made at least 2 valid format moves; INCOMPLETE if mostly invalid actions
# • exploration_progress: COMPLETED if revealed >10% of board; INCOMPLETE if revealed <10%  
# • logical_attempt: COMPLETED if attempted any deduction (even if wrong); INCOMPLETE if only random/invalid moves
# • error_recovery: COMPLETED if corrected any error within 3 attempts; INCOMPLETE if repeated same errors
# • cascade_usage: COMPLETED if triggered or attempted any cascade; INCOMPLETE if only single cell reveals
# • systematic_approach: COMPLETED if showed any pattern in move selection; INCOMPLETE if purely random
# </think>

# **Required JSON Output:**
# {{
#   "subtasks": [
#     {{"name": "valid_moves", "description": "[e.g., 'Made 5 valid moves like (1,1), (2,3)' or 'Only invalid formats like (-1,-1)']", "status": "[completed/incomplete]"}},
#     {{"name": "exploration_progress", "description": "[e.g., 'Revealed 15 cells (25% of board)' or 'Only revealed 2 cells']", "status": "[completed/incomplete]"}},
#     {{"name": "logical_attempt", "description": "[e.g., 'Tried to use (3,3)=1 constraint' or 'No deduction attempts']", "status": "[completed/incomplete]"}},
#     {{"name": "error_recovery", "description": "[e.g., 'Fixed format after 2 attempts' or 'Repeated invalid action 10 times']", "status": "[completed/incomplete]"}},
#     {{"name": "cascade_usage", "description": "[e.g., '(1,1) triggered 8-cell cascade' or 'No cascade attempts']", "status": "[completed/incomplete]"}},
#     {{"name": "systematic_approach", "description": "[e.g., 'Checked corners first' or 'Random clicking']", "status": "[completed/incomplete]"}}
#   ],
#   "trajectory_value": [count of completed subtasks out of 6],
#   "task_success": [true if successfully cleared all safe cells, false if detonated mine or incomplete],
#   "next_priority": "[Most important fix, e.g., 'Use valid (row,col) format' or 'When cell shows 1, count unopened neighbors']"
# }}

# **Evaluation Rules:**
# - **Determine Success Yourself:** You must judge 'task_success' by comparing the final board state in the trajectory to the Target Task.
# - **Criteria for Success:** The task is ONLY true if the agent successfully revealed ALL safe cells without detonating a mine.
# - **Criteria for Failure:** If the trajectory ends with a mine detonation, or if the agent stopped before revealing all safe cells, 'task_success' is false.
# - Each subtask status must reflect actual trajectory events.
# - Lessons should explain factors that led to the outcome (logic vs. guessing).
# - Reference specific elements from trajectory (coordinates, cell values).
# - Use null for lessons only if truly not applicable.

# Output ONLY the JSON evaluation.
# """