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

# --------------------- WebShop --------------------- #
WEBSHOP_TEMPLATE_NO_HIS = """
You are an expert autonomous agent operating in the WebShop e‑commerce environment. 
{reflections}
Your task is to: {task_description}.
Your current observation is: {current_observation}.
Your admissible actions of the current situation are: 
[
{available_actions}
].

Now it's your turn to take one action for the current step.
You should first reason step-by-step about the current situation, then think carefully which admissible action best advances the shopping goal. This reasoning process MUST be enclosed within <think> </think> tags. 
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
"""

WEBSHOP_TEMPLATE = """
You are an expert autonomous agent operating in the WebShop e‑commerce environment.
{reflections}
Your task is to: {task_description}.
Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took: {action_history}
You are now at step {current_step} and your current observation is: {current_observation}.
Your admissible actions of the current situation are: 
[
{available_actions}
].

Now it's your turn to take one action for the current step.
You should first reason step-by-step about the current situation, then think carefully which admissible action best advances the shopping goal. This reasoning process MUST be enclosed within <think> </think> tags. 
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
"""


# WEBSHOP_REFLECT_TEMPLATE = """
# You are an expert autonomous agent operating in the WebShop e-commerce environment. Your task is to: {task_description}

# You have just completed an attempt at this shopping task. The task was {success} completed.

# ## TRAJECTORY OF ACTIONS AND OBSERVATIONS:
# {current_trajectory}

# ## YOUR REFLECTION TASK:

# You will now reflect on this experience following a two-step process:

# ### Step 1: Analysis (Required)
# Provide a detailed analysis within <think> </think> tags that includes:
# 1. Break down the main task into atomic subtasks (search, filter, select, purchase)
# 2. Check if each subtask was completed based on the observations
# 3. Identify what worked well and what went wrong
# 4. Extract the most critical lesson from this specific attempt

# ### Step 2: Structured Output (Required)
# After your analysis, provide a JSON reflection within <reflection> </reflection> tags using this EXACT structure:

# {{
#   "subtasks": [
#     {{"name": "subtask_1", "description": "Search for space-saving dining room sets", "status": "completed"}},
#     {{"name": "subtask_2", "description": "Select gray-1 color option", "status": "incomplete"}},
#     {{"name": "subtask_3", "description": "Verify price under $340", "status": "completed"}},
#     {{"name": "subtask_4", "description": "Complete purchase", "status": "incomplete"}}
#   ],
#   "task_success": false,
#   "lesson": "Found item B09B7G7P58 at $319 meeting price requirement but clicked gray-1 color option without verifying selection or completing purchase - must confirm color selection shows in item details before buying"
# }}

# ## FORMAT SPECIFICATIONS:

# **Subtask fields:**
# - name: Sequential identifier (subtask_1, subtask_2, etc.)
# - description: Specific action with product/page references
# - status: "completed" or "incomplete" (lowercase only)

# **Task_success:** boolean (true/false, not string)

# **Lesson:** Single most important insight containing:
# - Specific product/page/filter references from trajectory
# - Either the critical failure reason OR key discovery
# - Actionable information for future attempts
# """


# WEBSHOP_REFLECT_TEMPLATE = """
# You are an expert autonomous agent operating in the WebShop e-commerce environment. Your task is to: {task_description}

# You have just completed an attempt at this shopping task. The task was {success} completed.

# ## TRAJECTORY OF ACTIONS AND OBSERVATIONS:
# {current_trajectory}

# ## YOUR REFLECTION TASK:

# You will now reflect on this experience following a two-step process:

# ### Step 1: Analysis (Required)
# Provide a detailed analysis within <think> </think> tags that includes:
# 1. Break down the main task into atomic subtasks (search, filter, select, purchase)
# 2. Check if each subtask was completed based on the observations
# 3. Identify what worked well and what went wrong
# 4. Extract the most critical lesson from this specific attempt

# ### Step 2: Structured Output (Required)
# After your analysis, provide a JSON reflection within <reflection> </reflection> tags using this EXACT structure:

# {{
#   "subtasks": [
#     {{"name": "subtask_1", "description": "Search for space-saving dining room sets", "status": "completed"}},
#     {{"name": "subtask_2", "description": "Select gray-1 color option", "status": "incomplete"}},
#     {{"name": "subtask_3", "description": "Verify price under $340", "status": "completed"}},
#     {{"name": "subtask_4", "description": "Complete purchase", "status": "incomplete"}}
#   ],
#   "task_success": false,
#   "lesson": "Found item B09B7G7P58 at $319 meeting price requirement but clicked gray-1 color option without verifying selection or completing purchase - must confirm color selection shows in item details before buying"
# }}

# ## FORMAT SPECIFICATIONS:

# **Subtask fields:**
# - name: Sequential identifier (subtask_1, subtask_2, etc.)
# - description: Specific action with product/page references
# - status: "completed" or "incomplete" (lowercase only)

# **Task_success:** boolean (true/false, not string)

# **Lesson:** Single most important insight containing:
# - Specific product/page/filter references from trajectory
# - Either the critical failure reason OR key discovery
# - Actionable information for future attempts
# """


# WEBSHOP_REFLECT_TEMPLATE = """
# You are an expert autonomous agent operating in the WebShop e-commerce environment. Your task is to: {task_description}

# You have just completed an attempt at this shopping task. The task was {success} completed.

# ## TRAJECTORY OF ACTIONS AND OBSERVATIONS:
# {current_trajectory}

# ## YOUR REFLECTION TASK:

# You will now reflect on this experience following a two-step process:

# ### Step 1: Analysis (Required)
# Provide a detailed analysis within <think> </think> tags that includes:
# 1. Break down the main task into atomic subtasks (search, filter, select, purchase)
# 2. Check if each subtask was completed based on the observations
# 3. Identify what worked well and what went wrong
# 4. Extract the most critical lesson from this specific attempt

# ### Step 2: Structured Output (Required)
# After your analysis, provide a JSON reflection within <reflection> </reflection> tags using this EXACT structure:

# {{
#   "subtasks": [
#     {{"name": "subtask_1", "description": "Search for space-saving dining room sets", "status": "completed"}},
#     {{"name": "subtask_2", "description": "Select gray-1 color option", "status": "incomplete"}},
#     {{"name": "subtask_3", "description": "Verify price under $340", "status": "completed"}},
#     {{"name": "subtask_4", "description": "Complete purchase", "status": "incomplete"}}
#   ],
#   "task_success": false,
#   "lesson": "Found item B09B7G7P58 at $319 meeting price requirement but clicked gray-1 color option without verifying selection or completing purchase - must confirm color selection shows in item details before buying"
# }}

# ## FORMAT SPECIFICATIONS:

# **Subtask fields:**
# - name: Sequential identifier (subtask_1, subtask_2, etc.)
# - description: Specific action with product/page references
# - status: "completed" or "incomplete" (lowercase only)

# **Task_success:** boolean (true/false, not string)

# **Lesson:** Single most important insight containing:
# - Specific product/page/filter references from trajectory
# - Either the critical failure reason OR key discovery
# - Actionable information for future attempts
# """


# WEBSHOP_REFLECT_TEMPLATE = """
# You are an expert autonomous agent operating in the WebShop e-commerce environment. Your task is to: {task_description}

# You have just completed an attempt at this shopping task. Analyze the trajectory below to determine whether the task was completed successfully.

# ## TRAJECTORY OF ACTIONS AND OBSERVATIONS:
# {current_trajectory}

# ## REFLECTION PROCESS:

# ### Step 1: Analysis
# Within <think> </think> tags, provide:
# 1. Decomposition of the main task into atomic subtasks (e.g., search, filter, select, purchase)
# 2. Assessment of each subtask's completion based on observations
# 3. Overall success determination (true only if ALL required subtasks completed)
# 4. Most critical action-related lesson (what action succeeded/failed and why)
# 5. Most critical environment-related lesson (discoveries about products, pages, or system behavior)

# ### Step 2: Structured Output
# Within <reflection> </reflection> tags, provide a JSON object with this EXACT structure:

# {{
#   "subtasks": [
#     {{"name": "subtask_1", "description": "Search for space-saving dining room sets", "status": "completed"}},
#     {{"name": "subtask_2", "description": "Select gray-1 color option", "status": "incomplete"}},
#     {{"name": "subtask_3", "description": "Verify price under $340", "status": "incomplete"}},
#     {{"name": "subtask_4", "description": "Complete purchase", "status": "incomplete"}}
#   ],
#   "task_success": false,
#   "action_lesson": "Clicked gray-1 color option on product B09B7G7P58 page but failed to verify selection registered - must confirm color choice appears in item details before proceeding to buy",
#   "environment_lesson": "Product B09B7G7P58 had multiple color variants (gray-1, brown, black) with different prices ($319 for gray-1) - color selection affects final price and availability"
# }}

# ## FIELD REQUIREMENTS:
# - subtasks: Array with name (subtask_1, subtask_2...), description (specific actions/products), status ("completed"/"incomplete")
# - task_success: boolean (true/false)
# - action_lesson: Critical action insight with specific references from trajectory
# - environment_lesson: Critical discovery about products/pages/system behavior
# - Set lessons to null if not applicable
# - Reference specific trajectory elements in all lessons
# """


# WEBSHOP_REFLECT_TEMPLATE = """
# You are an expert autonomous agent operating in the WebShop e-commerce environment. Your task is to: {task_description}

# You have just completed an attempt at this shopping task. The task was {success} completed.

# ## TRAJECTORY OF ACTIONS AND OBSERVATIONS:
# {current_trajectory}

# ## REFLECTION PROCESS:

# ### Step 1: Analysis
# Within <think> </think> tags, provide:
# 1. Decomposition of the main task into atomic subtasks (e.g., search, filter, select, purchase)
# 2. Assessment of each subtask's completion based on observations
# 3. Overall success determination (true only if ALL required subtasks completed)
# 4. Most critical action-related lesson (what action succeeded/failed and why)
# 5. Most critical environment-related lesson (discoveries about products, pages, or system behavior)

# ### Step 2: Structured Output
# Within <reflection> </reflection> tags, provide a JSON object with this EXACT structure:

# {{
#   "subtasks": [
#     {{"name": "subtask_1", "description": "Search for space-saving dining room sets", "status": "completed"}},
#     {{"name": "subtask_2", "description": "Select gray-1 color option", "status": "incomplete"}},
#     {{"name": "subtask_3", "description": "Verify price under $340", "status": "incomplete"}},
#     {{"name": "subtask_4", "description": "Complete purchase", "status": "incomplete"}}
#   ],
#   "task_success": false,
#   "action_lesson": "Clicked gray-1 color option on product B09B7G7P58 page but failed to verify selection registered - must confirm color choice appears in item details before proceeding to buy",
#   "environment_lesson": "Product B09B7G7P58 had multiple color variants (gray-1, brown, black) with different prices ($319 for gray-1) - color selection affects final price and availability"
# }}

# ## FIELD REQUIREMENTS:
# - subtasks: Array with name (subtask_1, subtask_2...), description (specific actions/products), status ("completed"/"incomplete")
# - task_success: boolean (true/false)
# - action_lesson: Critical action insight with specific references from trajectory
# - environment_lesson: Critical discovery about products/pages/system behavior
# - Set lessons to null if not applicable
# - Reference specific trajectory elements in all lessons
# """


# WEBSHOP_REFLECT_TEMPLATE = """
# You are an expert autonomous agent operating in the WebShop e-commerce environment.
# Your task is to: {task_description}

# You have just completed an attempt at this shopping task. The task was {success} completed.

# Actions taken and observations:
# {current_trajectory}

# ANALYSIS STEPS:

# 1. Break down the task into atomic subtasks (search, filter, select, purchase, etc.)
# 2. Check if each subtask was completed based on the observations
# 3. Determine overall success (true only if the correct item was purchased within budget)
# 4. Identify the MOST IMPORTANT 1-2 lessons from THIS SPECIFIC ATTEMPT
#    - Focus on the most critical failure point OR most valuable discovery
#    - Lessons must reference specific actions/pages/items from the trajectory

# CRITICAL: Return ONLY a valid JSON object with no additional text, using this EXACT structure:
# {{
# "subtasks": [
# {{"name": "search_product", "description": "Search for product with keywords", "status": "completed"}},
# {{"name": "apply_filters", "description": "Filter by size/color/price requirements", "status": "incomplete"}},
# {{"name": "select_item", "description": "Choose item matching all criteria", "status": "incomplete"}},
# {{"name": "complete_purchase", "description": "Buy the selected item", "status": "incomplete"}}
# ],
# "task_success": false,
# "action_lesson": "Searched 'wireless headphones' but missed the 'noise-cancelling' requirement - should include all key features in initial search query",
# "navigation_lesson": "Page 3 had items closer to budget after sorting by price - should explore multiple pages before selecting"
# }}

# RULES:

# - "status" must be exactly "completed" or "incomplete"
# - "task_success" must be a boolean (true or false, not a string)
# - "action_lesson": ONE most important action/strategy insight from this attempt (search terms, filter usage, navigation choices)
# - "navigation_lesson": ONE most important discovery about the site structure or product organization
# - Both lessons MUST reference specific elements from the trajectory (e.g., "page 3", "filter: under $50", "item B07XYZ")
# - If only one type of lesson is relevant, you may set the other to null
# - Prioritize lessons that explain why the correct item wasn't found or purchased
# - Do not include any text outside the JSON object
# """

# WEBSHOP_REFLECT_TEMPLATE = """
# You are an expert evaluating a WebShop shopping attempt.
# Your task is to: {task_description}

# You have just completed an attempt at this shopping task. The task was {success} completed.

# Trajectory of the attempt:
# {current_trajectory}

# <think>
# Given the task outcome, analyze the trajectory to understand:
# 1. What subtasks were attempted? (search, filter, select, purchase)
# 2. Which subtasks succeeded vs failed based on the observations?
# 3. What specific actions or decisions led to this outcome?
# 4. What are the 1-2 most valuable lessons from this attempt?
# </think>

# Output your evaluation as JSON:

# {{
# "subtasks": [
# {{"name": "search_product", "description": "[describe actual search]", "status": "[completed or incomplete]"}},
# {{"name": "apply_filters", "description": "[describe filters used]", "status": "[completed or incomplete]"}},
# {{"name": "select_item", "description": "[describe selection]", "status": "[completed or incomplete]"}},
# {{"name": "complete_purchase", "description": "[describe purchase]", "status": "[completed or incomplete]"}}
# ],
# "task_success": [true if successfully completed, false if unsuccessfully completed],
# "action_lesson": "[key action insight, e.g., 'Precise search with brand+model found exact match' OR 'Generic search missed required features']",
# "navigation_lesson": "[navigation insight, e.g., 'Efficient use of filters saved time' OR 'Failed to check additional pages with better options']"
# }}

# EVALUATION GUIDELINES:
# - The task outcome has been provided - use it to set task_success accordingly
# - Focus on WHY the attempt had this outcome:
#   * If successful: What strategies worked well?
#   * If unsuccessful: What went wrong and where?
# - Each subtask status must reflect actual trajectory events
# - Lessons should explain factors that led to the outcome
# - Reference specific elements from trajectory (item IDs, pages, search terms)
# - Use null for lessons only if truly not applicable

# Output ONLY the JSON evaluation.
# """

# # # --- UPDATED TEMPLATE TO INCLUDE REFERENCE TRAJECTORY ---
# WEBSHOP_REFLECT_TEMPLATE = """
# You are an expert evaluating a WebShop shopping attempt.
# Task Requirements: {task_description}

# You have just completed an attempt at this shopping task. The task was {success} completed.

# {reference_trajectory}

# Current Trajectory of the attempt:
# {current_trajectory}

# <think>
# If a reference trajectory exists, compare it with the current trajectory.
# Analyze the current trajectory to determine:
# 1. Which subtasks were attempted (search, filter, select, purchase)
# 2. Success or failure of each subtask based on observations
# 3. Specific actions/decisions that caused this outcome
# 4. 1-2 most valuable lessons from this attempt
# </think>

# **Required JSON Output:**
# {{
#   "subtasks": [
#     {{"name": "search_product", "description": "[actual search performed]", "status": "[completed/incomplete]"}},
#     {{"name": "apply_filters", "description": "[filters applied if any]", "status": "[completed/incomplete]"}},
#     {{"name": "select_item", "description": "[item selection details]", "status": "[completed/incomplete]"}},
#     {{"name": "complete_purchase", "description": "[purchase attempt details]", "status": "[completed/incomplete]"}}
#   ],
#   "task_success": [true if successfully completed, false if unsuccessfully completed],
#   "action_lesson": "[Key insight about actions taken, with specific examples]",
#   "navigation_lesson": "[Key insight about navigation strategy, with specific examples]"
# }}

# **Evaluation Rules:**
# • Set task_success to match the provided outcome
# • Analyze causation: What strategies led to success OR what failures caused the unsuccessful outcome
# • Each subtask status must match actual trajectory events
# • Include specific references (item IDs, page numbers, exact search terms) in lessons
# • Use null for lessons only when genuinely not applicable

# Output JSON only.
# """

WEBSHOP_REFLECT_TEMPLATE = """
You are an expert evaluating a WebShop shopping attempt.
Target Task: {task_description}

You have just completed an attempt at this shopping task.
Trajectory of the attempt:
{current_trajectory}

<think>
If a reference trajectory exists, compare it with the current trajectory.
Analyze the trajectory to determine if the task was successful:
1. Identify the specific requirements in the 'Target Task' (attributes, type, options).
2. Examine the final action in the trajectory. Did it end in a 'click[buy]'?
3. If a purchase was made, compare the purchased item's details against the 'Target Task' requirements.
4. Did the purchased item match ALL requirements? (If no purchase was made, it is a failure).
5. What specific actions or decisions led to this outcome?
6. What are the 1-2 most valuable lessons from this attempt?
</think>

Output your evaluation as JSON:

{{
"subtasks": [
{{"name": "search_product", "description": "[describe actual search]", "status": "[completed or incomplete]"}},
{{"name": "apply_filters", "description": "[describe filters used]", "status": "[completed or incomplete]"}},
{{"name": "select_item", "description": "[describe selection]", "status": "[completed or incomplete]"}},
{{"name": "complete_purchase", "description": "[describe purchase]", "status": "[completed or incomplete]"}}
],
"task_success": [true if the correct item was purchased, false otherwise],
"action_lesson": "[key action insight, e.g., 'Precise search with brand+model found exact match' OR 'Generic search missed required features']",
"navigation_lesson": "[navigation insight, e.g., 'Efficient use of filters saved time' OR 'Failed to check additional pages with better options']"
}}

EVALUATION GUIDELINES:
- **Determine Success Yourself:** You must judge 'task_success' by comparing the purchased item in the trajectory to the Target Task.
- **Criteria for Success:** The task is ONLY true if the agent successfully clicked 'buy' on an item that matches all required attributes (color, size, flavor, etc.).
- **Criteria for Failure:** If the trajectory ends without a purchase, or if the wrong item was bought, 'task_success' is false.
- Each subtask status must reflect actual trajectory events.
- Lessons should explain factors that led to the outcome.
- Reference specific elements from trajectory (item IDs, pages, search terms).
- Use null for lessons only if truly not applicable.

Output ONLY the JSON evaluation.
"""