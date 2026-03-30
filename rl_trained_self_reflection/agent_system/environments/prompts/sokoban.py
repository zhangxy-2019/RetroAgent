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

# --------------------- Sokoban --------------------- #
SOKOBAN_TEMPLATE_NO_HIS = """
You are an expert agent operating in the Sokoban environment.

# Symbols and Their Meaning
- Walls (`#`): These block movement. You can't move through or push anything into walls.
- Floor (`_`): Open spaces where you can walk and move boxes.
- Targets (`O`): The spots where boxes need to go.
- Boxes (`X`): These are what you need to push onto the targets.
- Player (`P`): That's you! You'll move around the grid to push boxes.
- Box on Target (`√`): A box successfully placed on a target.
- Player on Target (`S`): You standing on a target.

# Your Goal
Your goal is to push all the boxes (`X`) onto the target spots (`O`). Once all boxes are on the targets, you win!

# Rules
You can only push boxes. You can't pull them, so plan ahead to avoid getting stuck.
You can't walk through or push boxes into walls (`#`).
To avoid traps, do not push boxes into corners or against walls where they can't be moved again.

{reflections}

# Current Step
Your current observation is:
{current_observation}
Your admissible actions are ["up", "down", "left", "right"].

Now it's your turn to make a move (choose ONE action only for the current step).
You should first reason step-by-step about the current situation — observe the positions of boxes and targets, plan a path to push a box toward a target, and avoid traps like corners or walls. This reasoning process MUST be enclosed within <think> </think> tags. 
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
"""

SOKOBAN_TEMPLATE = """
You are an expert agent operating in the Sokoban environment.

# Symbols and Their Meaning
- Walls (`#`): These block movement. You can't move through or push anything into walls.
- Floor (`_`): Open spaces where you can walk and move boxes.
- Targets (`O`): The spots where boxes need to go.
- Boxes (`X`): These are what you need to push onto the targets.
- Player (`P`): That's you! You'll move around the grid to push boxes.
- Box on Target (`√`): A box successfully placed on a target.
- Player on Target (`S`): You standing on a target.

# Your Goal
Your goal is to push all the boxes (`X`) onto the target spots (`O`). Once all boxes are on the targets, you win!

# Rules
You can only push boxes. You can't pull them, so plan ahead to avoid getting stuck.
You can't walk through or push boxes into walls (`#`).
To avoid traps, do not push boxes into corners or against walls where they can't be moved again.

{reflections}

# Current Step
Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took: {action_history}
You are now at step {current_step} and your current observation is:
{current_observation}
Your admissible actions are ["up", "down", "left", "right"].

Now it's your turn to make a move (choose ONE action only for the current step).
You should first reason step-by-step about the current situation — observe the positions of boxes and targets, plan a path to push a box toward a target, and avoid traps like corners or walls. This reasoning process MUST be enclosed within <think> </think> tags. 
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
"""

SOKOBAN_VISUAL_TEMPLATE = """
You are an expert agent operating in the Sokoban environment. Your goal is to push all the boxes onto the target spots. Once all boxes are on the targets, you win!

# Rules
You can only push boxes. You can't pull them, so plan ahead to avoid getting stuck.
You can't walk through or push boxes into walls.
To avoid traps, do not push boxes into corners or against walls where they can't be moved again.

# Visual Elements in the Image:
Character: A small, green alien-like figure with two antennae and black eyes. It represents you.
Box: A yellow crate marked with an orange "X" across its front. It is the box you need to push.
Target: A black tile outlined in red, with a small red diamond shape in the center. It marks the destination where a box should be pushed.

# Current Step
Your current observation is shown in the image: <image>
Your admissible actions are ["up", "down", "left", "right"].

Now it's your turn to make a move (choose ONE action only for the current step).
You should first reason step-by-step about the current situation — observe the positions of boxes and targets, plan a path to push a box toward a target, and avoid traps like corners or walls. This reasoning process MUST be enclosed within <think> </think> tags. 
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
"""

SOKOBAN_REFLECT_TEMPLATE = """
You are an expert evaluating a Sokoban game attempt.
Task Requirements: Push all boxes ('X') onto target spots ('O') in the grid without getting them stuck against walls ('#') or in corners.

You have just completed an attempt at this Sokoban level.
Current Trajectory of the attempt:
{current_trajectory}

<think>
If a reference trajectory exists, compare it with the current trajectory.
Analyze the trajectory to determine if the task was successful:
1. Identify the grid layout and target locations in the 'Target Task'.
2. Examine the final board state in the trajectory. Are ALL boxes ('X') placed on targets ('O') resulting in '√'?
3. If the game ended without success, check for deadlocks (boxes stuck in corners or against walls).
4. Did the player successfully navigate the player ('P') to push positions without hitting walls repeatedly?
5. What specific logic or movement behavior led to this outcome?
6. What are the 1-2 most valuable lessons from this attempt?
7. Devise a concise, new plan of action that accounts for any mistakes with reference to specific actions that should be taken in the next trial

Game notation for reference:
• Symbols: # (wall), _ (floor), O (target), X (box), P (player), √ (box on target)
• Coordinates: (row, col)
• Valid actions: ["up", "down", "left", "right"]
• Rules: Push only (no pull), one box at a time, walls block movement.

Subtask Completion Criteria (binary evaluation for failed trajectories too):
• valid_moves: COMPLETED if made at least 2 valid directional moves; INCOMPLETE if mostly invalid formats/hallucinations
• navigation_logic: COMPLETED if player successfully navigated to a box; INCOMPLETE if stuck hitting walls/looping
• box_interaction: COMPLETED if at least one box was pushed to a new coordinate; INCOMPLETE if no boxes moved
• deadlock_avoidance: COMPLETED if avoided pushing boxes into unrecoverable corners/walls; INCOMPLETE if immediate deadlock created
• goal_progress: COMPLETED if at least one box was placed on a target; INCOMPLETE if 0 boxes on targets
• systematic_approach: COMPLETED if moves showed clear intent (e.g., moving behind a box to push); INCOMPLETE if random walking
</think>

**Required JSON Output:**
{{
  "subtasks": [
    {{"name": "valid_moves", "description": "[e.g., 'Outputted valid directions like up, down' or 'Used invalid commands']", "status": "[completed/incomplete]"}},
    {{"name": "navigation_logic", "description": "[e.g., 'Reached box at (3,2)' or 'Walked into wall at (1,1) repeatedly']", "status": "[completed/incomplete]"}},
    {{"name": "box_interaction", "description": "[e.g., 'Pushed box from (2,2) to (2,3)' or 'No boxes moved']", "status": "[completed/incomplete]"}},
    {{"name": "deadlock_avoidance", "description": "[e.g., 'Kept boxes away from corners' or 'Pushed box into corner (1,1)']", "status": "[completed/incomplete]"}},
    {{"name": "goal_progress", "description": "[e.g., '1/3 boxes placed on target' or 'No boxes on targets']", "status": "[completed/incomplete]"}},
    {{"name": "systematic_approach", "description": "[e.g., 'Cleared path for second box' or 'Random movement']", "status": "[completed/incomplete]"}}
  ],
  "trajectory_value": [count of completed subtasks out of 6],
  "task_success": [true if successfully placed all boxes on targets, false if deadlock or incomplete],
  "next_priority": "[Most important fix, e.g., 'Don't push box into corner at (1,1)' or 'Move to (2,3) to push down']"
}}

**Evaluation Rules:**
- **Determine Success Yourself:** You must judge 'task_success' by comparing the final board state in the trajectory to the Target Task.
- **Criteria for Success:** The task is ONLY true if ALL boxes are on target spots ('√').
- **Criteria for Failure:** If the trajectory ends with a deadlock, or if the agent stopped before placing all boxes, 'task_success' is false.
- Each subtask status must reflect actual trajectory events.
- Lessons should explain factors that led to the outcome (planning vs. random).
- Reference specific elements from trajectory (coordinates, symbols).
- Use null for lessons only if truly not applicable.

Output ONLY the JSON evaluation.
"""



# SOKOBAN_REFLECT_TEMPLATE = """
# You are an expert evaluating a Sokoban game attempt.
# Task Requirements: Push all boxes ('X') onto target spots ('O') in the grid without getting them stuck against walls ('#') or in corners.

# You have just completed an attempt at this Sokoban level.

# {reference_trajectory}

# Current Trajectory of the attempt:
# {current_trajectory}

# <think>
# If a reference trajectory exists, compare it with the current trajectory.
# Analyze the trajectory to determine if the task was successful:
# 1. Identify the grid layout and target locations in the 'Target Task'.
# 2. Examine the final board state in the trajectory. Are ALL boxes ('X') placed on targets ('O') resulting in '√'?
# 3. If the game ended without success, check for deadlocks (boxes stuck in corners or against walls).
# 4. Did the player successfully navigate the player ('P') to push positions without hitting walls repeatedly?
# 5. What specific logic or movement behavior led to this outcome?
# 6. What are the 1-2 most valuable lessons from this attempt?
# 7. Devise a concise, new plan of action that accounts for any mistakes with reference to specific actions that should be taken in the next trial

# Game notation for reference:
# • Symbols: # (wall), _ (floor), O (target), X (box), P (player), √ (box on target)
# • Coordinates: (row, col)
# • Valid actions: ["up", "down", "left", "right"]
# • Rules: Push only (no pull), one box at a time, walls block movement.

# Subtask Completion Criteria (binary evaluation for failed trajectories too):
# • valid_moves: COMPLETED if made at least 2 valid directional moves; INCOMPLETE if mostly invalid formats/hallucinations
# • navigation_logic: COMPLETED if player successfully navigated to a box; INCOMPLETE if stuck hitting walls/looping
# • box_interaction: COMPLETED if at least one box was pushed to a new coordinate; INCOMPLETE if no boxes moved
# • deadlock_avoidance: COMPLETED if avoided pushing boxes into unrecoverable corners/walls; INCOMPLETE if immediate deadlock created
# • goal_progress: COMPLETED if at least one box was placed on a target; INCOMPLETE if 0 boxes on targets
# • systematic_approach: COMPLETED if moves showed clear intent (e.g., moving behind a box to push); INCOMPLETE if random walking
# </think>

# **Required JSON Output:**
# {{
#   "subtasks": [
#     {{"name": "valid_moves", "description": "[e.g., 'Outputted valid directions like up, down' or 'Used invalid commands']", "status": "[completed/incomplete]"}},
#     {{"name": "navigation_logic", "description": "[e.g., 'Reached box at (3,2)' or 'Walked into wall at (1,1) repeatedly']", "status": "[completed/incomplete]"}},
#     {{"name": "box_interaction", "description": "[e.g., 'Pushed box from (2,2) to (2,3)' or 'No boxes moved']", "status": "[completed/incomplete]"}},
#     {{"name": "deadlock_avoidance", "description": "[e.g., 'Kept boxes away from corners' or 'Pushed box into corner (1,1)']", "status": "[completed/incomplete]"}},
#     {{"name": "goal_progress", "description": "[e.g., '1/3 boxes placed on target' or 'No boxes on targets']", "status": "[completed/incomplete]"}},
#     {{"name": "systematic_approach", "description": "[e.g., 'Cleared path for second box' or 'Random movement']", "status": "[completed/incomplete]"}}
#   ],
#   "trajectory_value": [count of completed subtasks out of 6],
#   "task_success": [true if successfully placed all boxes on targets, false if deadlock or incomplete],
#   "next_priority": "[Most important fix, e.g., 'Don't push box into corner at (1,1)' or 'Move to (2,3) to push down']"
# }}

# **Evaluation Rules:**
# - **Determine Success Yourself:** You must judge 'task_success' by comparing the final board state in the trajectory to the Target Task.
# - **Criteria for Success:** The task is ONLY true if ALL boxes are on target spots ('√').
# - **Criteria for Failure:** If the trajectory ends with a deadlock, or if the agent stopped before placing all boxes, 'task_success' is false.
# - Each subtask status must reflect actual trajectory events.
# - Lessons should explain factors that led to the outcome (planning vs. random).
# - Reference specific elements from trajectory (coordinates, symbols).
# - Use null for lessons only if truly not applicable.

# Output ONLY the JSON evaluation.
# """