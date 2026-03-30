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

# --------------------- ALFWorld --------------------- #
ALFWORLD_TEMPLATE_NO_HIS = """
You are an expert agent operating in the ALFRED Embodied Environment.
{reflections}
Your current observation is: {current_observation}
Your admissible actions of the current situation are: [{admissible_actions}].

Now it's your turn to take an action.
You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags. 
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
"""

ALFWORLD_TEMPLATE = """
You are an expert agent operating in the ALFRED Embodied Environment. Your task is to: {task_description}
{reflections}
Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took: {action_history}
You are now at step {current_step} and your current observation is: {current_observation}
Your admissible actions of the current situation are: [{admissible_actions}].

Now it's your turn to take an action.
You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags. 
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
"""


# ALFWORLD_REFLECT_TEMPLATE = """
# You are an expert evaluating an ALFRED Embodied Environment task attempt.
# Task Requirements: {task_description}

# You have just completed an attempt at this task. The task was {success} completed.

# {reference_trajectory}

# Current Trajectory of the attempt:
# {current_trajectory}

# <think>
# If a reference trajectory exists, compare it with the current trajectory.
# Analyze the current trajectory to determine:
# 1. Which subtasks were attempted (pick up, navigate, use appliance, place object)
# 2. Success or failure of each subtask based on observations
# 3. Specific actions/decisions that caused this outcome
# 4. 1-2 most valuable lessons from this attempt
# </think>

# **Required JSON Output:**
# {{
#   "subtasks": [
#     {{"name": "pick_up_object", "description": "[actual pickup performed]", "status": "[completed/incomplete]"}},
#     {{"name": "navigate_to_location", "description": "[navigation attempted]", "status": "[completed/incomplete]"}},
#     {{"name": "use_appliance", "description": "[appliance interaction if any]", "status": "[completed/incomplete]"}},
#     {{"name": "place_object", "description": "[placement attempt details]", "status": "[completed/incomplete]"}}
#   ],
#   "task_success": [true if successfully completed task goal, false if failed],
#   "action_lesson": "[Key insight about actions taken, with specific object IDs and outcomes]",
#   "navigation_lesson": "[Key insight about spatial navigation, with specific locations]"
# }}

# **Evaluation Rules:**
# • Set task_success to match the provided outcome
# • Analyze causation: What action sequence led to success OR what specific step/logic failure caused the unsuccessful outcome
# • Each subtask status must match actual trajectory events
# • Include specific references (object IDs like 'mug 1', locations like 'cabinet 2', appliances like 'microwave 1') in lessons
# • Use null for lessons only when genuinely not applicable

# Output JSON only.
# """

# ALFWORLD_REFLECT_TEMPLATE = """
# You are an expert evaluating an ALFWorld embodied agent attempt.
# Target Task: {task_description}

# {reference_trajectory}

# You have just completed an attempt at this household task.

# Trajectory of the attempt:
# {current_trajectory}

# <think>
# If a reference trajectory exists, compare it with the current trajectory.
# Analyze the trajectory to determine if the task was successful:
# 1. Identify the specific requirements in the 'Target Task' (target object, required state change, final destination).
# 2. Examine the sequence of actions. Did the agent successfully locate the correct object?
# 3. If a state change was required (clean, heat, cool, slice), was the correct appliance or tool used?
# 4. Did the agent place the object in the correct final receptacle?
# 5. Did the trajectory end with the 'stop' action after achieving the goal state? (If the agent stopped prematurely or failed to stop, it is a failure).
# 6. What specific actions or decisions led to this outcome?
# 7. What are the 1-2 most valuable lessons from this attempt?
# </think>

# Output your evaluation as JSON:

# {{
# "subtasks": [
# {{"name": "locate_object", "description": "[describe search for target object]", "status": "[completed or incomplete]"}},
# {{"name": "acquire_object", "description": "[describe picking up target]", "status": "[completed or incomplete]"}},
# {{"name": "modify_state", "description": "[describe heating/cleaning/cooling/slicing if applicable, else 'N/A']", "status": "[completed, incomplete, or N/A]"}},
# {{"name": "place_object", "description": "[describe final placement]", "status": "[completed or incomplete]"}}
# ],
# "task_success": [true if the goal state was achieved and 'stop' was called, false otherwise],
# "action_lesson": "[key action insight, e.g., 'Used microwave to heat apple instead of fridge' OR 'Failed to slice bread before plating']",
# "navigation_lesson": "[spatial/search insight, e.g., 'systematically checked all cabinet receptacles' OR 'wasted steps revisiting empty drawers']"
# }}

# EVALUATION GUIDELINES:
# - **Determine Success Yourself:** You must judge 'task_success' by comparing the final state in the trajectory to the Target Task.
# - **Criteria for Success:** The task is ONLY true if the agent manipulated the correct object, achieved the correct state (e.g., hot, clean), placed it in the correct target, and issued the 'stop' command.
# - **Criteria for Failure:** If the trajectory ends without the 'stop' command, or if the agent stopped without completing the goal (e.g., holding the object instead of placing it), 'task_success' is false.
# - Each subtask status must reflect actual trajectory events.
# - Lessons should explain factors that led to the outcome.
# - Reference specific elements from trajectory (object IDs like 'apple 1', receptacle IDs like 'countertop 2').
# - Use null for lessons only if truly not applicable.

# Output ONLY the JSON evaluation.
# """



ALFWORLD_REFLECT_TEMPLATE = """
You are an expert evaluating an ALFWorld embodied agent attempt.
Target Task: {task_description}

You have just completed an attempt at this household task.
Trajectory of the attempt:
{current_trajectory}

<think>
If a reference trajectory exists, compare it with the current trajectory.
Analyze the trajectory to determine if the task was successful:
1. Identify the specific requirements in the 'Target Task' (target object, required state change, final destination).
2. Examine the sequence of actions. Did the agent successfully locate the correct object?
3. If a state change was required (clean, heat, cool, slice), was the correct appliance or tool used?
4. Did the agent place the object in the correct final receptacle?
5. Did the trajectory end with the 'stop' action after achieving the goal state? (If the agent stopped prematurely or failed to stop, it is a failure).
6. What specific actions or decisions led to this outcome?
7. What are the 1-2 most valuable lessons from this attempt?
</think>

Output your evaluation as JSON:

{{
"subtasks": [
{{"name": "locate_object", "description": "[describe search for target object]", "status": "[completed or incomplete]"}},
{{"name": "acquire_object", "description": "[describe picking up target]", "status": "[completed or incomplete]"}},
{{"name": "modify_state", "description": "[describe heating/cleaning/cooling/slicing if applicable, else 'N/A']", "status": "[completed, incomplete, or N/A]"}},
{{"name": "place_object", "description": "[describe final placement]", "status": "[completed or incomplete]"}}
],
"task_success": [true if the goal state was achieved and 'stop' was called, false otherwise],
"action_lesson": "[key action insight, e.g., 'Used microwave to heat apple instead of fridge' OR 'Failed to slice bread before plating']",
"navigation_lesson": "[spatial/search insight, e.g., 'systematically checked all cabinet receptacles' OR 'wasted steps revisiting empty drawers']"
}}

EVALUATION GUIDELINES:
- **Determine Success Yourself:** You must judge 'task_success' by comparing the final state in the trajectory to the Target Task.
- **Criteria for Success:** The task is ONLY true if the agent manipulated the correct object, achieved the correct state (e.g., hot, clean), placed it in the correct target, and issued the 'stop' command.
- **Criteria for Failure:** If the trajectory ends without the 'stop' command, or if the agent stopped without completing the goal (e.g., holding the object instead of placing it), 'task_success' is false.
- Each subtask status must reflect actual trajectory events.
- Lessons should explain factors that led to the outcome.
- Reference specific elements from trajectory (object IDs like 'apple 1', receptacle IDs like 'countertop 2').
- Use null for lessons only if truly not applicable.

Output ONLY the JSON evaluation.
"""