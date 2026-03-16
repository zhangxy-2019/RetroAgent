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

ALFWORLD_REFLECT_TEMPLATE = """
You are an expert evaluating an ALFRED Embodied Environment task attempt.
Your task is to: {task_description}

You have just completed an attempt at this task. The task was {success} completed.

{reference_trajectory}

Trajectory of the attempt:
{current_trajectory}

<think>
If a reference trajectory exists, compare it with the current trajectory.
Given the task outcome, analyze the trajectory to understand:
1. What subtasks were attempted? (pick up, navigate, use appliance, place object)
2. Which subtasks succeeded vs failed based on the observations?
3. What specific actions or decisions led to this outcome?
4. What is the most valuable lesson from this attempt?
</think>

Output your evaluation as JSON:

{{
"subtasks": [
{{"name": "pick_up_object", "description": "[describe pickup action, e.g., 'Pick up mug from countertop']", "status": "[completed or incomplete]"}},
{{"name": "navigate_to_location", "description": "[describe navigation, e.g., 'Go to microwave 1']", "status": "[completed or incomplete]"}},
{{"name": "use_appliance", "description": "[describe appliance use, e.g., 'Heat mug in microwave']", "status": "[completed or incomplete]"}},
{{"name": "place_object", "description": "[describe placement, e.g., 'Place heated mug in cabinet']", "status": "[completed or incomplete]"}}
],
"task_success": [true if successfully completed task goal, false if failed],
"action_lesson": "[key action insight, e.g., 'Attempted to place mug 1 directly in cabinet 2 without heating - must use microwave 1 first' OR 'Successfully found knife in drawer 3 after checking wrong locations']",
"navigation_lesson": "[spatial insight, e.g., 'Microwave 1 located in kitchen area, not near cabinets' OR 'Multiple sinkbasins exist - must check all for target object']"
}}

EVALUATION GUIDELINES:
- The task outcome has been provided - use it to set task_success accordingly
- Focus on WHY the attempt had this outcome:
  * If successful: What sequence or strategy worked well?
  * If unsuccessful: What step failed or was missed?
- Each subtask status must reflect actual trajectory events
- Lessons should explain factors that led to the outcome
- Reference specific elements from trajectory (object IDs, locations, appliances)
- Use null for lessons only if truly not applicable

Output ONLY the JSON evaluation.
"""
