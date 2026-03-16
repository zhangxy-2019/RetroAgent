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

from typing import List, Tuple, Dict, Union, Any
from collections import defaultdict
import torch
import numpy as np
from functools import partial
import os
from agent_system.environments.prompts import *
from agent_system.environments.base import EnvironmentManagerBase, to_numpy
from agent_system.memory import SimpleMemory, SearchMemory, ReflectionMemory, WebshopSimpleMemory
from omegaconf import OmegaConf
import re
import copy

def parse_gamefile(infos):
    gamefile = []
    for info in infos:
        if 'extra.gamefile' in info:
            gamefile.append(info['extra.gamefile'])
        else:
            gamefile.append(None)
    return gamefile

def set_gamefile(infos, gamefile):
    for i in range(len(infos)):
        if 'extra.gamefile' in infos[i]:
            infos[i]['extra.gamefile'] = gamefile[i]
        else:
            infos[i]['extra.gamefile'] = None
    return infos


class SearchEnvironmentManager(EnvironmentManagerBase):
    """
    EnvironmentManager for SearchEnv.
    """
    def __init__(self, envs, projection_f, config):
        self.memory = SearchMemory()
        super().__init__(envs, projection_f, config)

    def reset(self, kwargs) -> Tuple[Dict[str, Any], List[Dict]]:
        obs, infos = self.envs.reset(kwargs=kwargs)
        self.tasks = obs

        self.memory.reset(batch_size=len(obs))

        observations = {
            "text": self.build_text_obs(obs, init=True),
            "image": None,
            "anchor": obs.copy()
        }
        
        return observations, infos

    def step(self, text_actions: List[str]):
        actions, valids = self.projection_f(text_actions)
        next_obs, rewards, dones, infos = self.envs.step(actions)
        self.memory.store({
            "search": actions,
            "information": next_obs,
        })

        next_observations = {
            "text": self.build_text_obs(next_obs),
            "image": None,
            "anchor": next_obs.copy()
        }
        
        for i, info in enumerate(infos):
            info["is_action_valid"] = to_numpy(valids[i])

        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos

    def build_text_obs(
        self,
        text_obs: List[str],
        init: bool = False
    ) -> List[str]:
        postprocess_text_obs: List[str] = []

        if not init and self.config.env.history_length > 0:
            memory_ctx, _ = self.memory.fetch(
                self.config.env.history_length,
                obs_key="information",
                action_key="search"
            )

        for i in range(len(text_obs)):
            if init or self.config.env.history_length <= 0:
                obs_i = SEARCH_TEMPLATE_NO_HIS.format(
                    task_description=self.tasks[i]
                )
            else:
                obs_i = SEARCH_TEMPLATE.format(
                    task_description=self.tasks[i],
                    memory_context=memory_ctx[i],
                    step_count=len(self.memory[i]),
                )
            postprocess_text_obs.append(obs_i)

        return postprocess_text_obs


    def _process_batch(self, batch_idx, total_batch_list, total_infos, success):
        # Find the last entry with active masks
        for i in reversed(range(len(total_batch_list[batch_idx]))):
            batch_item = total_batch_list[batch_idx][i]
            if batch_item['active_masks']:
                info = total_infos[batch_idx][i]
                won_value = float(info['won'])
                success['success_rate'].append(won_value)
                
                data_source = info.get("data_source")
                success[f"{data_source}_success_rate"].append(won_value)
                return  # Exit after finding the first active mask
            

class AlfWorldEnvironmentManager(EnvironmentManagerBase):
    def __init__(self, envs, projection_f, config, retrieve_type):
        self.memory = SimpleMemory()
        # --- NEW: Group and Eval Configuration ---
        self.group_n = config.env.rollout.n  # e.g., 8
        
        # --- Extract Hyperparameters from Config ---
        mem_config = config.env.get('reflection_memory', {})
        filepath = mem_config.get('filepath', "alfworld_reflections.json")
        import os
        if os.path.dirname(filepath):
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
        alpha = mem_config.get('alpha', 0.7)
        beta = mem_config.get('beta', 0.05)
        temp = mem_config.get('temperature', 0.5)
        ucb_scale = mem_config.get('ucb_scale', 1.0)
        self.top_k = mem_config.get('top_k', 1)

        # --- NEW: Memory Start Cutoff Configuration ---
        # Memory retrieval starts only when progress > memory_start_cutoff
        # Default is 0.0 (start immediately)
        self.memory_start_cutoff = mem_config.get('memory_start_cutoff', 0.0) 
        self.current_progress_ratio = 0.0 # Track progress internally
        self.retrieve_mode = mem_config.get('retrieve_mode', 'both')
        self.enable_memory = mem_config.get('enable_memory', True)
        self.group_outperformance = mem_config.get('group_outperformance', False)
        self.full_group_memory = mem_config.get('full_group_memory', False)
        self.group_relative_intrinsic_rewards = mem_config.get('group_relative_intrinsic_rewards', False)

        # --- NEW: Config to only give memory to 1 agent per group ---
        # If True, 1 agent retrieves, (group_n - 1) agents are control.
        # If False, (group_n / 2) agents retrieve, (group_n / 2) are control.
        self.potential_based_on_binary_success = mem_config.get('potential_based_on_binary_success', False)
        self.single_reflection_per_group = mem_config.get('single_reflection_per_group', False)
        # EMA Decay rate for the baseline (matches LaTeX gamma)
        self.ema_gamma = 0.9
        print("memory retrieve_type: ", retrieve_type)
        print("memory retrieve_mode: ", self.retrieve_mode)
        print("top_k_retrieved_memory: ", self.top_k)
        print(f"Memory Start Cutoff: {self.memory_start_cutoff}") 
        print(f"Global Memory Retrieval Enabled: {self.enable_memory}")
        print(f"Single Reflection Per Group: {self.single_reflection_per_group}")
        print(f"Potential Based On Binary Success Only: {self.potential_based_on_binary_success}")

        # Initialize the persistent reflection memory
        self.reflection_memory = ReflectionMemory(
            filepath=filepath,
            alpha=alpha,
            beta=beta,
            temperature=temp,
            retrieve_type=retrieve_type,
            ucb_scale=ucb_scale
        ) 
        # Initialize containers for retrieval tracking
        self.task_trajectory_history = {} # Added initialization
        self.task_potential_history = {} 
        self.batch_previous_potentials = [] 
        self.current_reflections = []      # Formatted strings for the prompt
        self.retrieved_raw_reflections = [] # List of lists of raw strings for utility updates
        self.current_retrieval_types = []
        self.batch_retrieved_types = []
        # Store the trajectories generated during the reflection phase
        # so they can be saved to memory in step_reflect
        self.last_trajectories = []
        super().__init__(envs, projection_f, config)

    # --- NEW: Method to update training progress ---
    def update_training_progress(self, current_step: int, total_steps: int):
        """
        Updates the environment with the current training progress.
        This triggers memory pruning if a 20% milestone is reached.
        """
        if total_steps > 0:
            self.current_progress_ratio = current_step / total_steps
            # Pass the ratio to memory to check for pruning triggers
            # We keep top-3 as requested
            # self.reflection_memory.check_and_prune(progress_ratio=ratio, top_k=3)

    def reset(self, kwargs) -> Dict[str, Any]:
        if kwargs is None:
            kwargs = {}
        print("****** environment resetting ******")
        # Determine mode based on kwargs
        is_eval = not kwargs.get('is_train', True)
        # print("is_eval: ", is_eval)

        text_obs, image_obs, infos = self.envs.reset()
        self.gamefile = parse_gamefile(infos)
        
        # initialize the history buffer
        self.memory.reset(batch_size = len(text_obs))
        self.tasks = []
        self.pre_text_obs = text_obs
        self.extract_task(text_obs)
        self.batch_size = len(text_obs)          # Expected: 128
        assert self.batch_size % self.group_n == 0, "Batch size must be divisible by group size"
        self.num_unique_tasks = self.batch_size // self.group_n

        # --- NEW: Retrieval Logic with Group-based Split ---
        self.current_reflections = []
        self.retrieved_raw_reflections = []
        self.batch_previous_potentials = []
        self.current_retrieval_types = [] 
        self.batch_retrieved_types = [] # Reset the type tracker
        group_split_index = self.group_n // 2
        if self.full_group_memory:
            group_split_index = 0
        # If we are training AND progress <= cutoff, we are in warmup -> Force memory OFF.
        # If progress > cutoff, we allow memory logic to proceed.
        in_warmup_period = (not is_eval) and (self.current_progress_ratio <= self.memory_start_cutoff)
        
        if in_warmup_period:
            # Optional: Log occasionally if needed
            print(f"Warmup Phase: Progress {self.current_progress_ratio:.2f} <= Cutoff {self.memory_start_cutoff}. Memory Disabled.")
            # pass 
        for i, task in enumerate(self.tasks):
            # Retrieve Phi(s) - the historical best FAILED completion for this task
            prev_potential = self.task_potential_history.get(task, 0.0)
            self.batch_previous_potentials.append(prev_potential)
            formatted_reflections = ""
            raw_list_of_dicts = [] # This will hold [{'text':..., 'type':...}]
            current_types_list = [] # List to hold types for this specific agent
            should_retrieve = False
            retrieval_type_str = "control"
            if self.enable_memory:
                if in_warmup_period:
                    # Explicitly disable retrieval during warmup
                    should_retrieve = False
                elif is_eval:
                    # During Eval: Everyone retrieves (or based on config)
                    should_retrieve = True
                    retrieval_type_str = "eval_retrieval"
                else:
                    position_in_group = i % self.group_n
                    if position_in_group >= group_split_index:
                        should_retrieve = True
                        retrieval_type_str = "experiment"
                    else:
                        should_retrieve = False
            else:
                should_retrieve = False
            
            if should_retrieve:
                # Retrieve top_k items
                k = self.top_k if is_eval else 1
                raw_list_of_dicts = self.reflection_memory.retrieve(
                    current_task_description=task, 
                    top_k=k, 
                    filter_type=self.retrieve_mode
                )
                if raw_list_of_dicts:
                    formatted_lines = []
                    for item in raw_list_of_dicts:
                        r_text = item.get('text', '')
                        r_type = item.get('type', 'unknown')
                        
                        # Store the type for logging
                        current_types_list.append(r_type)
                        
                        formatted_lines.append(r_text)
                    
                    formatted_reflections = "Past reflections on similar tasks:\n" + "\n".join(formatted_lines)
                    formatted_reflections += "\nWarning: These lessons may be outdated. Use them only if they align with your current observation."
            
            
            self.current_reflections.append(formatted_reflections)
            self.retrieved_raw_reflections.append(raw_list_of_dicts)
            self.current_retrieval_types.append(retrieval_type_str)
            self.batch_retrieved_types.append(current_types_list)
            print("retrieved_raw_reflections: ", self.retrieved_raw_reflections)
            # print("current_reflections: ", self.current_reflections)
            # --- NEW: Inject types into infos immediately ---
            infos[i]['reflection_types'] = current_types_list
            infos[i]['retrieval_group'] = retrieval_type_str
            print("infos[i]['retrieval_group']: ", infos[i]['retrieval_group'])
            
        assert len(self.current_reflections) == len(self.tasks)

        full_text_obs = self.build_text_obs(text_obs, self.envs.get_admissible_commands, init=True)
        return {'text': full_text_obs, 'image': image_obs, 'anchor': text_obs}, infos

    def reflect(self, infos: List[Dict]):
        """
        Called at the end of the 'play' phase.
        Updates utility based on Group B (Retrieved) vs Group A (Not Retrieved) performance.
        """
        # Build observation creates self.last_trajectories side-effect
        reflect_obs_text = self.build_reflect_text_obs(infos)
        
        observations = {
            'text': reflect_obs_text,
            'image': None,
            'anchor': reflect_obs_text
        }

        # Mark actions as valid for the reflection phase
        for info in infos:
            info['is_action_valid'] = to_numpy(True)

        batch_size = len(self.tasks)
        if batch_size % self.group_n != 0:
            print(f"WARNING: Batch size {batch_size} not divisible by group_n {self.group_n}")

        num_groups = batch_size // self.group_n
        group_split_index = self.group_n // 2
        
        # Iterate over each group independently
        for g in range(num_groups):
            start_idx = g * self.group_n
            end_idx = start_idx + self.group_n
            mid_idx = start_idx + group_split_index
            
            # 1. Calculate Wins for Control (First half)
            control_wins = 0
            for i in range(start_idx, mid_idx):
                if infos[i].get("won", False):
                    control_wins += 1
            
            # 2. Calculate Wins for Experiment (Second half)
            experiment_wins = 0
            for i in range(mid_idx, end_idx):
                if infos[i].get("won", False):
                    experiment_wins += 1
            
            # 3. Determine Utility Score for THIS group
            group_outperformed = experiment_wins > control_wins
            
            # 4. Update Memory ONLY for the Experiment agents
            for i in range(mid_idx, end_idx):
                task_desc = self.tasks[i]
                raw_reflection_items = self.retrieved_raw_reflections[i] # List[Dict]
                is_success = infos[i].get("won", False)
                
                # 1.0 if the agent won AND the retrieval group beat the control group.
                if self.group_outperformance:
                    if is_success and group_outperformed:
                        utility_score = 1.0
                    else:
                        utility_score = 0.0
                else:
                    if is_success:
                        utility_score = 1.0
                    else:
                        utility_score = 0.0


                if raw_reflection_items:
                    for item in raw_reflection_items:
                        # --- UPDATED: Extract text from dict for utility update ---
                        reflection_text = item.get('text', '')
                        if reflection_text:
                            self.reflection_memory.update_utility(
                                task_description=task_desc, 
                                reflection_text=reflection_text, 
                                score=utility_score
                            )

        return observations, infos

    def step_reflect(self, text_actions: List[str], infos: List[Dict]):
        """
        Calculates intrinsic rewards based on improvement over an EMA baseline,
        normalizes them group-wise, and updates the baseline.
        """
        import json
        import re
        import copy
        import numpy as np
        
        def to_numpy(x):
            return np.array(x) if not isinstance(x, np.ndarray) else x

        print("text_actions for reflection:", text_actions)
        
        # 1. Initialize Containers
        reflect_rewards = [] # This is the immediate reward for the reflection step itself (e.g. self-consistency)
        current_scores = np.zeros(self.batch_size) # The raw potential (phi)
        raw_improvements = np.zeros(self.batch_size) # The raw I (improvement)
        is_won_array = np.zeros(self.batch_size, dtype=bool)
        
        # Ensure batch_previous_potentials is synced
        if len(self.batch_previous_potentials) != self.batch_size:
            self.batch_previous_potentials = [0.0] * self.batch_size

        # 2. Calculate Raw Scores (Phi) and Raw Improvements (I)
        for i, reflection_text in enumerate(text_actions):
            task_desc = self.tasks[i]
            current_trajectory = self.last_trajectories[i] if i < len(self.last_trajectories) else ""
            
            # Get the baseline (Phi_{t-1})
            prev_phi = self.batch_previous_potentials[i]
            actual_success = bool(infos[i].get('won', False))
            is_won_array[i] = actual_success
            current_phi = 0.0
            # ... (JSON Parsing Logic - same as before) ...
            try:
                # --- JSON Extraction ---
                json_str = ""
                code_block_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', reflection_text, re.DOTALL)
                if code_block_match:
                    json_str = code_block_match.group(1)
                else:
                    clean_text = reflection_text.strip()
                    start_idx = clean_text.find('{')
                    end_idx = clean_text.rfind('}')
                    if start_idx != -1 and end_idx != -1:
                        json_str = clean_text[start_idx:end_idx+1]
                
                if not json_str: raise ValueError("No JSON found")
                
                reflection_data = json.loads(json_str)
                
                # --- Subtask Scoring (Phi calculation) ---
                subtasks = reflection_data.get('subtasks', [])
                total_subtasks = len(subtasks)
                completed_subtasks = sum(
                    1 for task in subtasks 
                    if isinstance(task, dict) and task.get('status', '').strip().lower() == 'completed'
                )
                
                # Calculate subtask-based potential
                subtask_phi = completed_subtasks / total_subtasks if total_subtasks > 0 else 0.0
                
                # --- DETERMINE CURRENT PHI ---
                if self.potential_based_on_binary_success:
                    # STRICT MODE: Only actual success matters for potential
                    current_phi = 1.0 if actual_success else 0.0
                else:
                    # DEFAULT MODE: Use subtasks, but override if actual success
                    current_phi = subtask_phi
                    if actual_success:
                        current_phi = 1.0

                # --- Reflection Consistency Reward (Auxiliary) ---
                predicted_success = reflection_data.get('task_success', False)
                if isinstance(predicted_success, str):
                    predicted_success = predicted_success.lower() in ['true', '1', 'yes']
                
                current_reward = 10.0 if predicted_success == actual_success and json_str else 0.0
                reflect_rewards.append(current_reward)
                
                # --- Memory Saving Logic (Same as before) ---
                if predicted_success == actual_success and json_str:
                    action_lesson = reflection_data.get('action_lesson')
                    nav_lesson = reflection_data.get('navigation_lesson')
                    lessons_to_save = []
                    if action_lesson and len(str(action_lesson)) > 5: lessons_to_save.append(f"Action Insight: {action_lesson}")
                    if nav_lesson and len(str(nav_lesson)) > 5: lessons_to_save.append(f"Navigation Insight: {nav_lesson}")
                    
                    if lessons_to_save:
                        final_lesson = " | ".join(lessons_to_save)
                        self.reflection_memory.add(
                            task_description=task_desc,
                            reflection_text=final_lesson,
                            trajectory=current_trajectory,
                            initial_score=0.5,
                            attempt_type="success" if actual_success else "failure",
                            current_progress_ratio=self.current_progress_ratio
                        )

            except Exception as e:
                print(f"Error task {i}: {e}")
                reflect_rewards.append(0.0)
                # Fallback logic for Phi on error
                if self.potential_based_on_binary_success:
                    current_phi = 1.0 if actual_success else 0.0
                else:
                    current_phi = 0.0

            # --- Calculate Raw Improvement (I) ---
            current_scores[i] = current_phi
            # Improvement is strictly positive gain over history
            improvement = max(0.0, current_phi - prev_phi)
            raw_improvements[i] = improvement

        # 3. Group-Relative Normalization & Baseline Update
        num_unique_tasks = self.batch_size // self.group_n
        final_intrinsic_rewards = np.zeros(self.batch_size)

        for group_idx in range(num_unique_tasks):
            start_idx = group_idx * self.group_n
            end_idx = start_idx + self.group_n
            
            task_desc = self.tasks[start_idx]
            
            # Extract group data
            group_improvements = raw_improvements[start_idx:end_idx]
            # group_scores = current_scores[start_idx:end_idx]
            
            # A. Normalization (Centering)
            # As per LaTeX Eq (8): R_int = I - Mean(I)
            if self.group_relative_intrinsic_rewards:
                group_mean_imp = np.mean(group_improvements)
                # Note: We do NOT divide by std here, just centering is sufficient 
                # to maintain the zero-sum property for the intrinsic component.
                centered_improvements = group_improvements - group_mean_imp
                final_intrinsic_rewards[start_idx:end_idx] = centered_improvements
            else:
                final_intrinsic_rewards[start_idx:end_idx] = group_improvements

            group_success_rate = np.mean(is_won_array[start_idx:end_idx].astype(float))
            old_baseline = self.task_potential_history.get(task_desc, 0.0)

            if group_success_rate > old_baseline:
                self.task_potential_history[task_desc] = group_success_rate
            # B. Update Historical Baseline (EMA)
            # As per LaTeX Eq (9): Phi_t = gamma * Phi_{t-1} + (1-gamma) * Mean(Phi_t)
            # if len(group_scores) > 0:
            #     current_group_mean_score = np.mean(group_scores)
            #     old_baseline = self.task_potential_history.get(task_desc, 0.0)

            #     if current_group_mean_score > old_baseline:
            #         self.task_potential_history[task_desc] = current_group_mean_score
                # # EMA Update
                # new_baseline = (self.ema_gamma * old_baseline) + ((1 - self.ema_gamma) * current_group_mean_score)
                # self.task_potential_history[task_desc] = new_baseline

        print("raw_improvements: ", raw_improvements)
        print("final_intrinsic_rewards (centered): ", final_intrinsic_rewards)
        infos = copy.deepcopy(infos)
        for info in infos:
            info['is_action_valid'] = to_numpy(True)
        # Convert to numpy for compatibility
        return None, to_numpy(reflect_rewards), to_numpy(final_intrinsic_rewards), None, copy.deepcopy(infos), to_numpy(current_scores)

    def step(self, text_actions: List[str]):
        actions, valids = self.projection_f(text_actions, self.envs.get_admissible_commands)
        text_obs, image_obs, rewards, dones, infos = self.envs.step(actions)
        self.memory.store({'text_obs': self.pre_text_obs, 'action': actions, 'reward': rewards, 'dones': dones, 'won': [info['won'] for info in infos]})
        self.pre_text_obs = text_obs

        full_text_obs = self.build_text_obs(text_obs, self.envs.get_admissible_commands)
        if infos[0].get("extra.gamefile") is None:
            infos = set_gamefile(infos, self.gamefile)

        # add action_valid to infos
        for i, info in enumerate(infos):
            info['is_action_valid'] = to_numpy(valids[i])

        next_observations = {'text': full_text_obs, 'image': image_obs, 'anchor': text_obs}
        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos
    
    def extract_task(self, text_obs: List[str]):
        for obs in text_obs:
            task_start = obs.find('Your task is to: ')
            
            if task_start != -1:
                self.tasks.append(obs[task_start + len('Your task is to: '):].strip())
            else:
                raise ValueError("Task description not found in text observation.")
        

    def build_text_obs(self, text_obs: List[str], admissible_actions: List[List[str]], init: bool = False) -> List[str]:
        """
        This function builds the text observation for the agent.
        """
        postprocess_text_obs = []
        if not init and self.config.env.history_length > 0:
            memory_contexts, valid_lens = self.memory.fetch(
                    self.config.env.history_length,
                    obs_key="text_obs",
                    action_key="action")
            
        for i in range(len(text_obs)):
            # exclude 'help' in admissible_actions[i]
            reformatted_admissible_actions = "\n ".join(f"'{s}'" for s in admissible_actions[i] if s != 'help')

            if init or self.config.env.history_length <= 0:
                obs = ALFWORLD_TEMPLATE_NO_HIS.format(
                    reflections=self.current_reflections[i], # Add this if template supports it
                    current_observation=text_obs[i],
                    admissible_actions=reformatted_admissible_actions
                )
            else:
                obs = ALFWORLD_TEMPLATE.format(
                    task_description=self.tasks[i],
                    reflections=self.current_reflections[i], # <--- INJECTED HERE
                    step_count=len(self.memory[i]),
                    history_length=valid_lens[i],
                    action_history=memory_contexts[i],
                    current_step=len(self.memory[i]) + 1,
                    current_observation=text_obs[i],
                    admissible_actions=reformatted_admissible_actions
                )

            postprocess_text_obs.append(obs)
        return postprocess_text_obs

    def build_reflect_text_obs(self, infos: List[str]) -> List[str]:
        """
        This function builds the text observation for the agent during reflection.
        It uses ALFWORLD_REFLECT_TEMPLATE which requires task_description and current_trajectory.
        """
        postprocess_text_obs = []
        memory_contexts, valid_lens = self.memory.fetch(
                50,
                obs_key="text_obs",
                action_key="action")
        # self.task_trajectory_history[task] = {"successful": [], "failed": []}
        for i in range(len(infos)):
            task = self.tasks[i]
            # Ensure key exists (it should from reset, but safety first)
            if task not in self.task_trajectory_history:
                self.task_trajectory_history[task] = {"successful": [], "failed": []}
                
            if infos[i].get("won", False):
                self.task_trajectory_history[task]["successful"].append(memory_contexts[i])
            else:
                self.task_trajectory_history[task]["failed"].append(memory_contexts[i])

        # --- CRITICAL: Store these so step_reflect can access them ---
        self.last_trajectories = memory_contexts
        for i in range(len(infos)):
            task = self.tasks[i]
            is_won = infos[i].get("won", False)
            
            # Determine success string and select Contrastive Reference
            # If we WON, we want to see a FAIL to understand what to avoid (or just compare)
            # If we LOST, we want to see a SUCCESS to understand what to do
            
            reference_traj_str = "No reference history available yet."
            
            if is_won:
                SUCCESS = "successfully"
                # Try to get a failed example
                failed_hist = self.task_trajectory_history[task]["failed"]
                if failed_hist:
                    # Use the most recent failure
                    reference_traj_str = "Reference Failed Trajectory (for comparison):\n" + failed_hist[-1]
                else:
                    reference_traj_str = "No failed attempts available for comparison."
            else:
                SUCCESS = "unsuccessfully" # Changed from "NOT successfully" for better grammar
                # Try to get a successful example
                success_hist = self.task_trajectory_history[task]["successful"]
                if success_hist:
                    # Use the most recent success
                    reference_traj_str = "Reference Successful Trajectory (for comparison):\n" + success_hist[-1]
                else:
                    reference_traj_str = "No successful attempts available for reference."

            obs = ALFWORLD_REFLECT_TEMPLATE.format(
                task_description=self.tasks[i],
                success=SUCCESS,
                reference_trajectory=reference_traj_str,
                current_trajectory=memory_contexts[i],  
            )
            postprocess_text_obs.append(obs)
        
        # Debug print
        if len(postprocess_text_obs) > 0:
            print("processed_reflect_text [0]: ", postprocess_text_obs[0])
            
        return postprocess_text_obs

    def success_evaluator(self, *args, **kwargs) -> Dict[str, np.ndarray]:
        """
        Evaluate if the episodes are successful or not. 
        
        Args:
            kwargs: Must contain:
                - total_infos (List[List[Dict]]): Info dicts for every step.
                - total_batch_list (List[List[Dict]]): Trajectory data.
                - reflect_rewards (np.ndarray or List, optional): Rewards specifically for the reflection phase.
        
        Returns:
            - success (Dict[str, np.ndarray]): Dictionary of success metrics.
        """
        total_infos = kwargs['total_infos']
        total_batch_list = kwargs['total_batch_list']
        
        # Extract reflect_rewards. It might be None if not in training mode.
        reflect_rewards = kwargs.get('reflect_rewards', None)

        batch_size = len(total_batch_list)
        success = defaultdict(list)
        
        for bs in range(batch_size):
            # Extract the specific reflection reward for this batch index
            r_reward = None
            if reflect_rewards is not None:
                # Handle case where reflect_rewards is a list, numpy array, or tensor
                try:
                    r_reward = reflect_rewards[bs]
                except IndexError:
                    # Fallback if sizes mismatch (though they shouldn't)
                    r_reward = 0.0
            
            self._process_batch(bs, total_batch_list, total_infos, success, reflect_reward=r_reward)
        
        # Ensure consistency in list lengths
        assert len(success['success_rate']) == batch_size

        return {key: np.array(value) for key, value in success.items()}

    def _process_batch(self, batch_idx, total_batch_list, total_infos, success, reflect_reward=None):
        """
        Process a single batch trajectory to extract success metrics.
        
        Args:
            batch_idx: Index of the current batch.
            total_batch_list: The full list of trajectories.
            total_infos: The full list of info dicts.
            success: The dictionary to append results to.
            reflect_reward: The specific reward value for the reflection phase (float, Tensor, or None).
        """
        # --- 1. Process Reflection Phase ---
        # Since reflect_rewards are passed explicitly, we process them directly here.
        if reflect_reward is not None:
            # Convert Tensor or numpy scalar to python float
            if hasattr(reflect_reward, 'item'):
                val = float(reflect_reward.item())
            else:
                val = float(reflect_reward)
            success['reflect_success_rate'].append(val)
        else:
            # If no reflection rewards provided (e.g. eval mode), log 0.0
            success['reflect_success_rate'].append(0.0)

        # --- 2. Process Play Phase ---
        play_success_found = False
        
        # Iterate backwards to find the last active step of the 'play' phase.
        # We do this because the 'won' flag is usually at the end of the trajectory.
        trajectory = total_batch_list[batch_idx]
        
        for i in reversed(range(len(trajectory))):
            batch_item = trajectory[i]
            
            # Skip inactive steps (padding)
            if not batch_item.get('active_masks', True):
                continue
                
            # Check phase. Based on your logs, these are likely 'play'.
            phase = batch_item.get('phase', 'play')
            
            if phase == 'play':
                info = total_infos[batch_idx][i]
                won_value = float(info.get('won', 0.0))
                
                # General Play Success
                success['play_success_rate'].append(won_value)
                success['success_rate'].append(won_value) # Main metric usually tracks play
                
                # Task-Specific Success (e.g., pick_and_place, alfworld)
                # Check 'extra.gamefile' (ALFWorld specific)
                gamefile = info.get("extra.gamefile")
                if gamefile:
                    self._process_gamefile(gamefile, won_value, success)
                
                # Fallback: Check 'data_source'
                elif "data_source" in info:
                    data_source = info.get("data_source")
                    success[f"{data_source}_success_rate"].append(won_value)
                
                play_success_found = True
                
                # Once we find the valid end of the play phase, we stop searching this batch
                break
        
        # --- 3. Handle Missing Play Phase ---
        # If for some reason a trajectory has no 'play' phase or is entirely padding
        if not play_success_found:
             success['play_success_rate'].append(0.0)
             success['success_rate'].append(0.0)

    def _process_gamefile(self, gamefile, won_value, success):
        tasks = [
            "pick_and_place",
            "pick_two_obj_and_place",
            "look_at_obj_in_light",
            "pick_heat_then_place_in_recep",
            "pick_cool_then_place_in_recep",
            "pick_clean_then_place_in_recep",
        ]
        
        for task in tasks:
            if task in gamefile:
                success[f"{task}_success_rate"].append(won_value)
                break

class SokobanEnvironmentManager(EnvironmentManagerBase):
    ACTION_LOOKUP = {
        0: "Still",
        1: "Up",
        2: "Down",
        3: "Left",
        4: "Right",
    }

    def __init__(self, envs, projection_f, config, retrieve_type=None):
        self.is_multi_modal = envs.mode == 'rgb_array'
        self.memory = SimpleMemory()
        self.num_actions_per_turn = config.env.get('num_actions_per_turn', 3)
        self.max_turns = config.env.get('max_turns', 7)
        # --- Reflection Configuration Start ---
        self.group_n = config.env.rollout.n 
        
        mem_config = config.env.get('reflection_memory', {})
        filepath = mem_config.get('filepath', "sokoban_reflections.json")
        import os
        if os.path.dirname(filepath):
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
        alpha = mem_config.get('alpha', 0.7)
        beta = mem_config.get('beta', 0.05)
        temp = mem_config.get('temperature', 0.5)
        ucb_scale = mem_config.get('ucb_scale', 1.0)
        self.top_k = mem_config.get('top_k', 1)
        
        self.memory_start_cutoff = mem_config.get('memory_start_cutoff', 0.0) 
        self.current_progress_ratio = 0.0 # Track progress internally
        self.retrieve_mode = mem_config.get('retrieve_mode', 'both')
        self.enable_memory = mem_config.get('enable_memory', True)
        self.group_outperformance = mem_config.get('group_outperformance', False)
        self.full_group_memory = mem_config.get('full_group_memory', False)
        self.group_relative_intrinsic_rewards = mem_config.get('group_relative_intrinsic_rewards', False)

        self.potential_based_on_binary_success = mem_config.get('potential_based_on_binary_success', False)
        self.single_reflection_per_group = mem_config.get('single_reflection_per_group', False)
        # EMA Decay rate for the baseline (matches LaTeX gamma)
        self.ema_gamma = 0.9
        print("memory retrieve_type: ", retrieve_type)
        print("memory retrieve_mode: ", self.retrieve_mode)
        print("top_k_retrieved_memory: ", self.top_k)
        print(f"Memory Start Cutoff: {self.memory_start_cutoff}") 
        print(f"Global Memory Retrieval Enabled: {self.enable_memory}")
        print(f"Single Reflection Per Group: {self.single_reflection_per_group}")
        print(f"Potential Based On Binary Success Only: {self.potential_based_on_binary_success}")
        # Initialize persistent reflection memory
        self.reflection_memory = ReflectionMemory(
            filepath=filepath,
            alpha=alpha,
            beta=beta,
            temperature=temp,
            retrieve_type=retrieve_type,
            ucb_scale=ucb_scale
        )
        self.task_trajectory_history = {}
        self.task_potential_history = {} 
        self.batch_previous_potentials = [] 
        
        
        # Initialize containers for retrieval tracking
        self.current_reflections = []       # Formatted strings for the prompt
        self.retrieved_raw_reflections = []  # List of lists of raw strings for utility updates
        self.init_states = []
        self.current_retrieval_types = []
        # Store the trajectories generated during the reflection phase
        # so they can be saved to memory in step_reflect
        self.last_trajectories = []        
        super().__init__(envs, projection_f, config)

    def update_training_progress(self, current_step: int, total_steps: int):
        """Updates the environment with the current training progress."""
        if total_steps > 0:
            self.current_progress_ratio = current_step / total_steps
            # self.reflection_memory.check_and_prune(progress_ratio=ratio, top_k=3)

    def reset(self, kwargs):
        if kwargs is None:
            kwargs = {}
            
        # Determine mode based on kwargs
        is_eval = not kwargs.get('is_train', True)
        # print("is_eval:", is_eval)

        obs, infos = self.envs.reset()
        self.init_states = obs # Store initial state for retrieval key
        
        if self.is_multi_modal:
            obs_array = np.array(obs, obs[0].dtype)
            self.pre_text_obs = self.envs.render(mode='tiny_rgb_array')
            # Note: For visual sokoban, we might lack a text description for retrieval.
            # We assume 'obs' or 'infos' contains a level string/ID for retrieval keys.
        else:
            self.pre_text_obs = obs
        
        self.batch_size = len(obs)
        assert self.batch_size % self.group_n == 0, "Batch size must be divisible by group size"
        self.num_unique_tasks = self.batch_size // self.group_n

        self.current_reflections = []
        self.retrieved_raw_reflections = []
        self.batch_previous_potentials = []
        self.current_retrieval_types = [] 
        self.batch_retrieved_types = [] # Reset the type tracker
        in_warmup_period = (not is_eval) and (self.current_progress_ratio <= self.memory_start_cutoff)
        
        if in_warmup_period:
            # Optional: Log occasionally if needed
            print(f"Warmup Phase: Progress {self.current_progress_ratio:.2f} <= Cutoff {self.memory_start_cutoff}. Memory Disabled.")
            pass 
        group_split_index = self.group_n // 2
        if self.full_group_memory:
            group_split_index = 0
        
        for i, task in enumerate(self.init_states):
            prev_potential = self.task_potential_history.get(task, 0.0)
            self.batch_previous_potentials.append(prev_potential)
            formatted_reflections = ""
            raw_list_of_dicts = [] # This will hold [{'text':..., 'type':...}]
            current_types_list = [] # List to hold types for this specific agent

            should_retrieve = False
            retrieval_type_str = "control"
            if self.enable_memory:
                if in_warmup_period:
                    # Explicitly disable retrieval during warmup
                    should_retrieve = False
                elif is_eval:
                    # During Eval: Everyone retrieves (or based on config)
                    should_retrieve = True
                    retrieval_type_str = "eval_retrieval"
                else:
                    position_in_group = i % self.group_n
                    if position_in_group >= group_split_index:
                        should_retrieve = True
                        retrieval_type_str = "experiment"
                    else:
                        should_retrieve = False
            else:
                should_retrieve = False

            if should_retrieve:
                # Retrieve top_k items
                k = self.top_k if is_eval else 1
                raw_list_of_dicts = self.reflection_memory.retrieve(
                    current_task_description=task, 
                    top_k=k, 
                    filter_type=self.retrieve_mode
                )
                if raw_list_of_dicts:
                    formatted_lines = []
                    for item in raw_list_of_dicts:
                        r_text = item.get('text', '')
                        r_type = item.get('type', 'unknown')
                        
                        # Store the type for logging
                        current_types_list.append(r_type)
                        
                        formatted_lines.append(r_text)
                    
                    formatted_reflections = "Past reflections on similar tasks:\n" + "\n".join(formatted_lines)
                    formatted_reflections += "\nWarning: These lessons may be outdated. Use them only if they align with your current observation."
            
            self.current_reflections.append(formatted_reflections)
            self.retrieved_raw_reflections.append(raw_list_of_dicts)
            self.current_retrieval_types.append(retrieval_type_str)
            self.batch_retrieved_types.append(current_types_list)
            print("retrieved_raw_reflections: ", self.retrieved_raw_reflections)
            # print("current_reflections: ", self.current_reflections)
            # --- NEW: Inject types into infos immediately ---
            infos[i]['reflection_types'] = current_types_list
            infos[i]['retrieval_group'] = retrieval_type_str
        
        # -----------------------------------------------------------
        assert len(self.current_reflections) == len(self.init_states)
        # Build observations
        if self.is_multi_modal:
            observations = {
                'text': self.build_text_obs(infos, init=True), 
                'image': obs_array,   
                'anchor': obs_array
            }
        else:
            observations = {
                'text': self.build_text_obs(infos, obs, init=True),
                'image': None,
                'anchor': obs
            }
            
        self.memory.reset(batch_size = len(infos))
        return observations, infos

    def step(self, text_actions: List[str]):
        actions, valids = self.projection_f(text_actions)
        next_obs, rewards, dones, infos = self.envs.step(actions)

        for i, info in enumerate(infos):
            info['is_action_valid'] = to_numpy(valids[i])

        # Store in memory. Note: If is_multi_modal, pre_text_obs is an image array.
        # For reflection purposes, we might prefer storing the text action and maybe a simplified state if available.
        self.memory.store({
            'text_obs': self.pre_text_obs, 
            'action': [self.ACTION_LOOKUP[act] for act in actions],
            'reward': rewards,
            'dones': dones,
            'won': [info.get('won', False) for info in infos] # Ensure 'won' is tracked
        })

        if self.is_multi_modal:
            next_obs_array = np.array(next_obs, next_obs[0].dtype)
            self.pre_text_obs = self.envs.render(mode='tiny_rgb_array')
            next_observations = {
                'text': self.build_text_obs(infos),  
                'image': next_obs_array,
                'anchor': next_obs_array 
            }
        else:
            self.pre_text_obs = next_obs
            next_observations = {
                'text': self.build_text_obs(infos, next_obs),  
                'image': None, 
                'anchor': next_obs 
            }

        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos

    def reflect(self, infos: List[Dict]):
        """
        Called at the end of the 'play' phase.
        Updates utility based on Group B (Retrieved) vs Group A (Not Retrieved) performance.
        """
        observations = {
            'text': self.build_reflect_text_obs(infos),
            'image': None,
            'anchor': self.build_reflect_text_obs(infos) # Anchor needed for some pipelines
        }
        
        # Ensure action validity is set for all
        for info in infos:
            info['is_action_valid'] = to_numpy(True)
        
        batch_size = len(self.init_states)
        
        # Ensure batch size is divisible by group_n
        if batch_size % self.group_n != 0:
            print(f"WARNING: Batch size {batch_size} not divisible by group_n {self.group_n}")
        
        num_groups = batch_size // self.group_n
        group_split_index = self.group_n // 2
        
        # Iterate over each group independently
        for g in range(num_groups):
            start_idx = g * self.group_n
            end_idx = start_idx + self.group_n
            mid_idx = start_idx + group_split_index
            
            # Calculate wins for control group (first half)
            control_wins = 0
            for i in range(start_idx, mid_idx):
                if infos[i].get("won", False):
                    control_wins += 1
            
            # Calculate wins for experiment group (second half)
            experiment_wins = 0
            for i in range(mid_idx, end_idx):
                if infos[i].get("won", False):
                    experiment_wins += 1
            
            # 3. Determine Utility Score for THIS group
            group_outperformed = experiment_wins > control_wins
            
            # 4. Update Memory ONLY for the Experiment agents
            for i in range(mid_idx, end_idx):
                task_desc = self.init_states[i]
                raw_reflection_items = self.retrieved_raw_reflections[i] # List[Dict]
                is_success = infos[i].get("won", False)
                
                # 1.0 if the agent won AND the retrieval group beat the control group.
                if self.group_outperformance:
                    if is_success and group_outperformed:
                        utility_score = 1.0
                    else:
                        utility_score = 0.0
                else:
                    if is_success:
                        utility_score = 1.0
                    else:
                        utility_score = 0.0

                if raw_reflection_items:
                    for item in raw_reflection_items:
                        # --- UPDATED: Extract text from dict for utility update ---
                        reflection_text = item.get('text', '')
                        if reflection_text:
                            self.reflection_memory.update_utility(
                                task_description=task_desc, 
                                reflection_text=reflection_text, 
                                score=utility_score
                            )
        
        return observations, infos

    def step_reflect(self, text_actions: List[str], infos: List[Dict]):
        """
        Stores the generated reflection into the persistent memory.
        Parses the JSON output containing specific subtasks and the critical lesson.
        """
        import json
        import re
        import copy
        import numpy as np
        
        def to_numpy(x):
            return np.array(x) if not isinstance(x, np.ndarray) else x

        print("text_actions for reflection:", text_actions)
            
        
        # 1. Initialize Containers
        reflect_rewards = [] # This is the immediate reward for the reflection step itself (e.g. self-consistency)
        current_scores = np.zeros(self.batch_size) # The raw potential (phi)
        raw_improvements = np.zeros(self.batch_size) # The raw I (improvement)
        is_won_array = np.zeros(self.batch_size, dtype=bool)
        
        # Ensure batch_previous_potentials is synced
        if len(self.batch_previous_potentials) != self.batch_size:
            self.batch_previous_potentials = [0.0] * self.batch_size

        # 2. Calculate Raw Scores (Phi) and Raw Improvements (I)
        for i, reflection_text in enumerate(text_actions):
            task_desc = self.init_states[i]
            current_trajectory = self.last_trajectories[i] if i < len(self.last_trajectories) else ""
            
            # Get the baseline (Phi_{t-1})
            prev_phi = self.batch_previous_potentials[i]
            actual_success = bool(infos[i].get('won', False))
            is_won_array[i] = actual_success
            current_phi = 0.0
            
            try:
                # --- JSON Extraction ---
                json_str = ""       
                # 1. Isolate the content AFTER the <think> block to avoid parsing errors
                # (e.g., preventing the parser from grabbing a '{' inside the reasoning text)
                content_to_parse = reflection_text
                if "</think>" in reflection_text:
                    content_to_parse = reflection_text.split("</think>")[-1]
                
                # 2. Try to find a Markdown JSON code block
                code_block_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', reflection_text, re.DOTALL)
                if code_block_match:
                    json_str = code_block_match.group(1)
                else:
                    clean_text = reflection_text.strip()
                    start_idx = clean_text.find('{')
                    end_idx = clean_text.rfind('}')
                    if start_idx != -1 and end_idx != -1:
                        json_str = clean_text[start_idx:end_idx+1]
                
                if not json_str: raise ValueError("No JSON found")
                
                reflection_data = json.loads(json_str)

                # --- Subtask Scoring (Phi calculation) ---
                subtasks = reflection_data.get('subtasks', [])
                total_subtasks = len(subtasks)
                completed_subtasks = sum(
                    1 for task in subtasks 
                    if isinstance(task, dict) and task.get('status', '').strip().lower() == 'completed'
                )
                
                # Calculate subtask-based potential
                subtask_phi = completed_subtasks / total_subtasks if total_subtasks > 0 else 0.0
                
                # --- DETERMINE CURRENT PHI ---
                if self.potential_based_on_binary_success:
                    current_phi = 1.0 if actual_success else 0.0
                else:
                    current_phi = subtask_phi
                    if actual_success:
                        current_phi = 1.0

                # --- Reflection Consistency Reward (Auxiliary) ---
                predicted_success = reflection_data.get('task_success', False)
                if isinstance(predicted_success, str):
                    predicted_success = predicted_success.lower() in ['true', '1', 'yes']
                
                current_reward = 10.0 if predicted_success == actual_success and json_str else 0.0
                reflect_rewards.append(current_reward)

                # --- Memory Saving Logic ---
                if predicted_success == actual_success and json_str:
                    next_priority = reflection_data.get('next_priority')
                    lessons_to_save = []
                    if next_priority and len(str(next_priority)) > 5:
                        lessons_to_save.append(f"New Plan: {next_priority}")
                    
                    if lessons_to_save:
                        final_lesson = " | ".join(lessons_to_save)
                        self.reflection_memory.add(
                            task_description=task_desc,
                            reflection_text=final_lesson,
                            trajectory=current_trajectory,
                            initial_score=0.5,
                            attempt_type="success" if actual_success else "failure",
                            current_progress_ratio=self.current_progress_ratio
                        )
            except Exception as e:
                print(f"Error task {i}: {e}")
                reflect_rewards.append(0.0)
                if self.potential_based_on_binary_success:
                    current_phi = 1.0 if actual_success else 0.0
                else:
                    current_phi = 0.0

            # --- Calculate Raw Improvement (I) ---
            current_scores[i] = current_phi
            # Improvement is strictly positive gain over history
            improvement = max(0.0, current_phi - prev_phi)
            raw_improvements[i] = improvement
                    
        # 3. Group-Relative Normalization & Baseline Update
        num_unique_tasks = self.batch_size // self.group_n
        final_intrinsic_rewards = np.zeros(self.batch_size)

        for group_idx in range(num_unique_tasks):
            start_idx = group_idx * self.group_n
            end_idx = start_idx + self.group_n
            
            task_desc = self.init_states[start_idx]
            
            # Extract group data
            group_improvements = raw_improvements[start_idx:end_idx]
            # group_scores = current_scores[start_idx:end_idx]
            
            # A. Normalization (Centering)
            # As per LaTeX Eq (8): R_int = I - Mean(I)
            if self.group_relative_intrinsic_rewards:
                group_mean_imp = np.mean(group_improvements)
                # Note: We do NOT divide by std here, just centering is sufficient 
                # to maintain the zero-sum property for the intrinsic component.
                centered_improvements = group_improvements - group_mean_imp
                final_intrinsic_rewards[start_idx:end_idx] = centered_improvements
            else:
                final_intrinsic_rewards[start_idx:end_idx] = group_improvements

            group_success_rate = np.mean(is_won_array[start_idx:end_idx].astype(float))
            old_baseline = self.task_potential_history.get(task_desc, 0.0)

            if group_success_rate > old_baseline:
                self.task_potential_history[task_desc] = group_success_rate
            # B. Update Historical Baseline (EMA)
            # As per LaTeX Eq (9): Phi_t = gamma * Phi_{t-1} + (1-gamma) * Mean(Phi_t)
            # if len(group_scores) > 0:
            #     current_group_mean_score = np.mean(group_scores)
            #     old_baseline = self.task_potential_history.get(task_desc, 0.0)

            #     if current_group_mean_score > old_baseline:
            #         self.task_potential_history[task_desc] = current_group_mean_score
                # # EMA Update
                # new_baseline = (self.ema_gamma * old_baseline) + ((1 - self.ema_gamma) * current_group_mean_score)
                # self.task_potential_history[task_desc] = new_baseline

        print("raw_improvements: ", raw_improvements)
        print("final_intrinsic_rewards (centered): ", final_intrinsic_rewards)
        infos = copy.deepcopy(infos)
        for info in infos:
            info['is_action_valid'] = to_numpy(True)

        return None, to_numpy(reflect_rewards), to_numpy(final_intrinsic_rewards), None, copy.deepcopy(infos), to_numpy(current_scores)

    def build_text_obs(self, infos, text_obs: List[str]=None, init: bool = False) -> List[str]:
        postprocess_text_obs = []

        if not init and self.config.env.history_length > 0:
            memory_contexts, valid_lens = self.memory.fetch(
                    self.config.env.history_length,
                    obs_key="text_obs",
                    action_key="action")
            
        for i in range(len(infos)):
            # Inject reflections into the prompt
            reflections_str = self.current_reflections[i] if hasattr(self, 'current_reflections') else ""
            
            if init or self.config.env.history_length <= 0:
                if self.is_multi_modal:
                    # For visual, we might append reflections to the system prompt elsewhere, 
                    # or assume the model handles visual + text context.
                    obs = SOKOBAN_VISUAL_TEMPLATE 
                else:
                    obs = SOKOBAN_TEMPLATE_NO_HIS.format(
                        reflections=reflections_str, 
                        current_observation=text_obs[i]
                    )
            else:
                if self.is_multi_modal:
                    obs = SOKOBAN_VISUAL_TEMPLATE
                else:
                    obs = SOKOBAN_TEMPLATE.format(
                        step_count=len(self.memory[i]),
                        history_length=valid_lens[i],
                        action_history=memory_contexts[i],
                        current_step=len(self.memory[i]) + 1,
                        reflections=reflections_str,
                        current_observation=text_obs[i]
                    )
            postprocess_text_obs.append(obs)

        return postprocess_text_obs

    def build_reflect_text_obs(self, infos: List[Dict]) -> List[str]:
        """
        Builds the text observation for the reflection phase.
        """
        postprocess_text_obs = []
        memory_contexts, valid_lens = self.memory.fetch(
            15, # Sokoban games can be long
            obs_key="text_obs",
            action_key="action"
        )

        # self.task_trajectory_history[task] = {"successful": [], "failed": []}
        for i in range(len(infos)):
            task = self.init_states[i]
            # Ensure key exists (it should from reset, but safety first)
            if task not in self.task_trajectory_history:
                self.task_trajectory_history[task] = {"successful": [], "failed": []}
                
            if infos[i].get("won", False):
                self.task_trajectory_history[task]["successful"].append(memory_contexts[i])
            else:
                self.task_trajectory_history[task]["failed"].append(memory_contexts[i])
        # --- CRITICAL: Store these so step_reflect can access them ---
        self.last_trajectories = memory_contexts

        for i in range(len(infos)):
            task = self.init_states[i]
            is_won = infos[i].get("won", False)
            reference_traj_str = "No reference history available yet."
            if is_won:
                SUCCESS = "successfully"
                # Try to get a failed example
                failed_hist = self.task_trajectory_history[task]["failed"]
                if failed_hist:
                    # Use the most recent failure
                    reference_traj_str = "Reference Failed Trajectory (for comparison):\n" + failed_hist[-1]
                else:
                    reference_traj_str = "No failed attempts available for comparison."
            else:
                SUCCESS = "unsuccessfully" # Changed from "NOT successfully" for better grammar
                # Try to get a successful example
                success_hist = self.task_trajectory_history[task]["successful"]
                if success_hist:
                    # Use the most recent success
                    reference_traj_str = "Reference Successful Trajectory (for comparison):\n" + success_hist[-1]
                else:
                    reference_traj_str = "No successful attempts available for reference."
            # If multi-modal, memory_contexts[i] might contain image arrays which we can't print to text.
            # We need to sanitize the history for the text-based reflection prompt.
            history_str = ""
            if self.is_multi_modal:
                # If visual, we rely on the action history primarily
                # We assume memory_contexts returns a list of (obs, action) or similar.
                # Since SimpleMemory.fetch usually returns formatted strings if configured, 
                # we might need to manually reconstruct the action sequence here.
                
                # Fallback: just list actions
                actions = self.memory.get_all_actions(i) # Hypothetical helper or manual access
                # If get_all_actions doesn't exist, we rely on what fetch returned.
                # If fetch returned images, we skip them.
                
                # Assuming memory_contexts[i] is a string representation of history:
                if isinstance(memory_contexts[i], str):
                    history_str = memory_contexts[i]
                else:
                    # If it's not string (e.g. list of images), we construct a simple action log
                    raw_actions = self.memory[i]['action'] # Access raw storage
                    history_str = " -> ".join([str(a) for a in raw_actions])
            else:
                history_str = memory_contexts[i]
            obs = SOKOBAN_REFLECT_TEMPLATE.format(
                success=SUCCESS,
                reference_trajectory=reference_traj_str,
                current_trajectory=history_str
            )
            postprocess_text_obs.append(obs)
        
        return postprocess_text_obs

    def success_evaluator(self, *args, **kwargs) -> Dict[str, np.ndarray]:
        from collections import defaultdict
        
        total_infos = kwargs['total_infos']
        total_batch_list = kwargs['total_batch_list']
        reflect_rewards = kwargs.get('reflect_rewards', None)
        
        batch_size = len(total_batch_list)
        success = defaultdict(list)
        
        for bs in range(batch_size):
            r_reward = 0.0
            if reflect_rewards is not None and bs < len(reflect_rewards):
                try:
                    r_reward = float(reflect_rewards[bs])
                except:
                    r_reward = 0.0
            
            success['reflect_success_rate'].append(r_reward)
            
            # Process play phase
            play_success_found = False
            trajectory = total_batch_list[bs]
            
            for i in reversed(range(len(trajectory))):
                batch_item = trajectory[i]
                if not batch_item.get('active_masks', True): continue
                
                phase = batch_item.get('phase', 'play')
                if phase == 'play':
                    info = total_infos[bs][i]
                    won_value = float(info.get('won', 0.0))
                    
                    success['play_success_rate'].append(won_value)
                    success['success_rate'].append(won_value)
                    play_success_found = True
                    break
            
            if not play_success_found:
                success['play_success_rate'].append(0.0)
                success['success_rate'].append(0.0)
        
        return {key: np.array(value) for key, value in success.items()}

class GymCardEnvironmentManager(EnvironmentManagerBase):
    def __init__(self, envs, projection_f, config):
        super().__init__(envs, projection_f, config)
    
    def reset(self, kwargs) -> Dict[str, Any]:
        obs, infos = self.envs.reset()
        # infos = [None] * self.envs.num_envs
        observations = {'text': self.build_text_obs(infos), 'image': obs, 'anchor': obs.copy()}
        
        return observations, infos

    def step(self, text_actions: List[str]):
        next_observations, rewards, dones, infos = super().step(text_actions)
        
        # add text observation to next_observations
        next_observations['text'] = self.build_text_obs(infos)
        next_observations['anchor'] = next_observations['image'].copy()

        return next_observations, rewards, dones, infos


    def build_text_obs(self, infos: Tuple[Dict]=None) -> List[str]:
        """
        This function builds the text observation for the agent.
        """
        postprocess_text_obs = []
        for i in range(len(infos)):
            if 'ezpoints' in self.config.env.env_name.lower():
                text_formula = ''.join(str(element) for element in infos[i]['Formula']) if infos[i] is not None else ''
                obs = GYM_CARDS_EZPOINTS_TEMPLATE.format(text_formula=text_formula)
            elif 'points24' in self.config.env.env_name.lower():
                text_formula = ''.join(str(element) for element in infos[i]['Formula']) if infos[i] is not None else ''
                obs = GYM_CARDS_POINTS24_TEMPLATE.format(text_formula=text_formula)
            elif 'numberline' in self.config.env.env_name.lower():
                obs = GYM_CARDS_NUMBERLINE_TEMPLATE
            elif "blackjack" in self.config.env.env_name.lower():
                obs = GYM_CARDS_BLACKJACK_TEMPLATE
            else:
                raise ValueError(f"Unsupported environment: {self.config.env.env_name}")
            postprocess_text_obs.append(obs)
        return postprocess_text_obs

class WebshopEnvironmentManager(EnvironmentManagerBase):
    def __init__(self, envs, projection_f, config, retrieve_type):
        # print("config: ", config)
        self.memory = WebshopSimpleMemory()
        self.group_n = config.env.rollout.n  # e.g., 8
        
        # --- Extract Hyperparameters from Config ---
        mem_config = config.env.get('reflection_memory', {})
        filepath = mem_config.get('filepath', "webshop_reflections.json")
        import os
        if os.path.dirname(filepath):
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
        alpha = mem_config.get('alpha', 0.7)
        beta = mem_config.get('beta', 0.05)
        temp = mem_config.get('temperature', 0.5)
        ucb_scale = mem_config.get('ucb_scale', 1.0)
        self.top_k = mem_config.get('top_k', 1)
        
        # --- NEW: Memory Start Cutoff Configuration ---
        # Memory retrieval starts only when progress > memory_start_cutoff
        # Default is 0.0 (start immediately)
        self.memory_start_cutoff = mem_config.get('memory_start_cutoff', 0.0) 
        self.current_progress_ratio = 0.0 # Track progress internally
        self.retrieve_mode = mem_config.get('retrieve_mode', 'both')
        self.enable_memory = mem_config.get('enable_memory', True)
        self.group_outperformance = mem_config.get('group_outperformance', False)
        self.full_group_memory = mem_config.get('full_group_memory', False)
        self.group_relative_intrinsic_rewards = mem_config.get('group_relative_intrinsic_rewards', False)
        self.success_reference_induction = mem_config.get('success_reference_induction', False)

        # --- NEW: Potential Calculation Config ---
        # If True, current_phi is strictly 1.0 (won) or 0.0 (lost).
        # If False, current_phi is calculated via subtask completion % from JSON, overridden by 1.0 if won.
        self.potential_based_on_binary_success = mem_config.get('potential_based_on_binary_success', False)
        # --- NEW: Config to only give memory to 1 agent per group ---
        # If True, 1 agent retrieves, (group_n - 1) agents are control.
        # If False, (group_n / 2) agents retrieve, (group_n / 2) are control.
        self.single_reflection_per_group = mem_config.get('single_reflection_per_group', False)
        # --- NEW: Reflection Decay Configuration ---
        # If True, gradually reduce the number of reflection-receiving agents
        # from half-group (at progress=0) to 0 (at progress=1).
        # This overrides full_group_memory and single_reflection_per_group during training.
        self.reflection_decay = mem_config.get('reflection_decay', False)
        # Optional: define the progress point at which reflections fully vanish.
        # Default is 1.0 (reflections reach 0 at the end of training).
        self.reflection_decay_end = mem_config.get('reflection_decay_end', 1.0)
        # EMA Decay rate for the baseline (matches LaTeX gamma)
        self.ema_gamma = 0.9
        print("memory retrieve_type: ", retrieve_type)
        print("memory retrieve_mode: ", self.retrieve_mode)
        print("top_k_retrieved_memory: ", self.top_k)
        print(f"Memory Start Cutoff: {self.memory_start_cutoff}") 
        print(f"Global Memory Retrieval Enabled: {self.enable_memory}")
        print(f"Single Reflection Per Group: {self.single_reflection_per_group}")
        print(f"Reflection Decay Enabled: {self.reflection_decay}")
        print(f"Reflection Decay End Point: {self.reflection_decay_end}")
        print(f"Potential Based On Binary Success Only: {self.potential_based_on_binary_success}")

        # Initialize the persistent reflection memory
        self.reflection_memory = ReflectionMemory(
            filepath=filepath,
            alpha=alpha,
            beta=beta,
            temperature=temp,
            retrieve_type=retrieve_type,
            ucb_scale=ucb_scale
        )
        self.task_trajectory_history = {}
        self.task_potential_history = {} 
        self.batch_previous_potentials = [] 
        
        # Initialize containers for retrieval tracking
        self.current_reflections = []      # Formatted strings for the prompt
        self.retrieved_raw_reflections = [] # List of LISTS OF DICTS (updated structure)  
        # --- NEW: Track the type of retrieval for the current batch ---
        self.current_retrieval_types = []
        # Store the trajectories generated during the reflection phase
        # so they can be saved to memory in step_reflect
        self.last_trajectories = []        
        super().__init__(envs, projection_f, config)

    def update_training_progress(self, current_step: int, total_steps: int):
        """
        Updates the environment with the current training progress.
        This triggers memory pruning if a 20% milestone is reached.
        """
        if total_steps > 0:
            self.current_progress_ratio = current_step / total_steps
            
            # Pass the ratio to memory to check for pruning triggers
            # self.reflection_memory.check_and_prune(progress_ratio=self.current_progress_ratio, top_k=3)
    def _compute_group_split_index(self, is_eval: bool) -> int:
        """
        Computes the group_split_index determining how many agents per group
        receive reflections. Agents at positions >= group_split_index get reflections.

        - Default (no decay): half the group receives reflections (split_index = group_n // 2).
        - full_group_memory: all agents receive reflections (split_index = 0).
        - reflection_decay (training only): linearly reduces the number of reflection
          agents from half-group to 0 as progress goes from 0 to reflection_decay_end.
        """
        half_group = self.group_n // 2
        # --- Reflection Decay Logic (training only) ---
        if self.reflection_decay:
            # Clamp the decay progress ratio to [0, 1]
            if self.reflection_decay_end > 0:
                decay_ratio = min(self.current_progress_ratio / self.reflection_decay_end, 1.0)
            else:
                decay_ratio = 1.0  # If end is 0, immediately decay to 0

            # Number of agents that should receive reflections:
            # Linearly from half_group (at decay_ratio=0) to 0 (at decay_ratio=1)
            num_reflection_agents = max(0, round(half_group * (1.0 - decay_ratio)))

            # group_split_index = group_n - num_reflection_agents
            # e.g., group_n=8, half=4, decay_ratio=0.5 -> agents=2 -> split=6
            group_split_index = self.group_n - num_reflection_agents

            print(
                f"[Reflection Decay] progress={self.current_progress_ratio:.3f}, "
                f"decay_ratio={decay_ratio:.3f}, "
                f"reflection_agents_per_group={num_reflection_agents}/{self.group_n}, "
                f"group_split_index={group_split_index}"
            )
            return group_split_index

        # --- Standard (non-decay) Logic ---
        if self.full_group_memory:
            return 0
        return half_group

    def reset(self, kwargs) -> Dict[str, Any]:
        # 1. Check the flag. Default to False if not provided.
        if kwargs is None:
            kwargs = {}
        print("****** environment resetting ******")
        # Determine mode based on kwargs
        is_eval = not kwargs.get('is_train', True)
        # print("is_eval: ", is_eval)

        obs, infos = self.envs.reset()
        self.tasks = self.extract_task(obs)
        
        obs = self.format_obs(obs)
        observations = {
            'text': self.build_text_obs(obs, infos, init=True), 
            'image': None, 
            'anchor': obs.copy()
        }
        self.pre_text_obs = obs
        self.memory.reset(batch_size=len(infos))
        self.batch_size = len(obs)
        assert self.batch_size % self.group_n == 0, "Batch size must be divisible by group size"
        self.num_unique_tasks = self.batch_size // self.group_n

        self.current_reflections = []
        self.retrieved_raw_reflections = []
        self.batch_previous_potentials = []
        self.current_retrieval_types = [] 
        self.batch_retrieved_types = [] # Reset the type tracker
        # # Example: group_n = 8. split_index = 4.
        group_split_index = self.group_n // 2
        if self.full_group_memory:
            group_split_index = 0
        # --- NEW: Check if we have passed the cutoff ---
        # If we are training AND progress <= cutoff, we are in warmup -> Force memory OFF.
        # If progress > cutoff, we allow memory logic to proceed.
        in_warmup_period = (not is_eval) and (self.current_progress_ratio <= self.memory_start_cutoff)
        
        if in_warmup_period:
            # Optional: Log occasionally if needed
            print(f"Warmup Phase: Progress {self.current_progress_ratio:.2f} <= Cutoff {self.memory_start_cutoff}. Memory Disabled.")
            pass 

        # --- Use the new helper to compute the split index ---
        # group_split_index = self._compute_group_split_index(is_eval)
        for i, task in enumerate(self.tasks):
            prev_potential = self.task_potential_history.get(task, 0.0)
            self.batch_previous_potentials.append(prev_potential)
            formatted_reflections = ""
            raw_list_of_dicts = [] # This will hold [{'text':..., 'type':...}]
            current_types_list = [] # List to hold types for this specific agent
            
            should_retrieve = False
            retrieval_type_str = "control"
            
            if self.enable_memory:
                if in_warmup_period:
                    # Explicitly disable retrieval during warmup
                    should_retrieve = False
                elif is_eval:
                    # During Eval: Everyone retrieves (or based on config)
                    should_retrieve = True
                    retrieval_type_str = "eval_retrieval"
                else:
                    position_in_group = i % self.group_n
                    if position_in_group >= group_split_index:
                        should_retrieve = True
                        retrieval_type_str = "experiment"
                    else:
                        should_retrieve = False
            else:
                should_retrieve = False

            if should_retrieve:
                # Retrieve top_k items
                k = self.top_k if is_eval else 1
                
                raw_list_of_dicts = self.reflection_memory.retrieve(
                    current_task_description=task, 
                    top_k=k, 
                    filter_type=self.retrieve_mode
                )
                
                if raw_list_of_dicts:
                    formatted_lines = []
                    for item in raw_list_of_dicts:
                        r_text = item.get('text', '')
                        r_type = item.get('type', 'unknown')
                        
                        # Store the type for logging
                        current_types_list.append(r_type)
                        
                        formatted_lines.append(r_text)
                    
                    formatted_reflections = "Past reflections on similar tasks:\n" + "\n".join(formatted_lines)
                    formatted_reflections += "\nWarning: These lessons may be outdated. Use them only if they align with your current observation."
            
            
            self.current_reflections.append(formatted_reflections)
            self.retrieved_raw_reflections.append(raw_list_of_dicts)
            self.current_retrieval_types.append(retrieval_type_str)
            self.batch_retrieved_types.append(current_types_list)
            # print("retrieved_raw_reflections: ", self.retrieved_raw_reflections)
            # print("current_reflections: ", self.current_reflections)
            # --- NEW: Inject types into infos immediately ---
            infos[i]['reflection_types'] = current_types_list
            infos[i]['retrieval_group'] = retrieval_type_str
            # print("infos[i]['reflection_types']: ", infos[i]['reflection_types'])
            print("infos[i]['retrieval_group']: ", infos[i]['retrieval_group'])
        # Debug prints
        # print("retrieved_raw_reflections: ", self.retrieved_raw_reflections)
        # exit(0)
        assert len(self.current_reflections) == len(self.tasks)
        return observations, infos

    def reflect(self, infos: List[Dict]):
        """
        Called at the end of the 'play' phase.
        Updates utility based on Group B (Retrieved) vs Group A (Not Retrieved) performance.
        """
        # Build observation creates self.last_trajectories side-effect
        reflect_obs_text = self.build_reflect_text_obs(infos)
        
        observations = {
            'text': reflect_obs_text,
            'image': None,
            'anchor': reflect_obs_text
        }

        # Mark actions as valid for the reflection phase
        for info in infos:
            info['is_action_valid'] = to_numpy(True)

        batch_size = len(self.tasks)
        if batch_size % self.group_n != 0:
            print(f"WARNING: Batch size {batch_size} not divisible by group_n {self.group_n}")

        num_groups = batch_size // self.group_n
        group_split_index = self.group_n // 2
        
        # Iterate over each group independently
        for g in range(num_groups):
            start_idx = g * self.group_n
            end_idx = start_idx + self.group_n
            mid_idx = start_idx + group_split_index
            
            # 1. Calculate Wins for Control (First half)
            control_wins = 0
            for i in range(start_idx, mid_idx):
                if infos[i].get("won", False):
                    control_wins += 1
            
            # 2. Calculate Wins for Experiment (Second half)
            experiment_wins = 0
            for i in range(mid_idx, end_idx):
                if infos[i].get("won", False):
                    experiment_wins += 1
            
            # 3. Determine Utility Score for THIS group
            group_outperformed = experiment_wins > control_wins
            
            # 4. Update Memory ONLY for the Experiment agents
            for i in range(mid_idx, end_idx):
                task_desc = self.tasks[i]
                raw_reflection_items = self.retrieved_raw_reflections[i] # List[Dict]
                is_success = infos[i].get("won", False)
                
                # 1.0 if the agent won AND the retrieval group beat the control group.
                if self.group_outperformance:
                    if is_success and group_outperformed:
                        utility_score = 1.0
                    else:
                        utility_score = 0.0
                else:
                    if is_success:
                        utility_score = 1.0
                    else:
                        utility_score = 0.0


                if raw_reflection_items:
                    for item in raw_reflection_items:
                        # --- UPDATED: Extract text from dict for utility update ---
                        reflection_text = item.get('text', '')
                        if reflection_text:
                            self.reflection_memory.update_utility(
                                task_description=task_desc, 
                                reflection_text=reflection_text, 
                                score=utility_score
                            )

        return observations, infos

    def step_reflect(self, text_actions: List[str], infos: List[Dict]):
        """
        Calculates intrinsic rewards based on improvement over an EMA baseline,
        normalizes them group-wise, and updates the baseline.
        """
        import json
        import re
        import copy
        import numpy as np
        
        def to_numpy(x):
            return np.array(x) if not isinstance(x, np.ndarray) else x

        print("text_actions for reflection:", text_actions)
        
        # 1. Initialize Containers
        reflect_rewards = [] # This is the immediate reward for the reflection step itself (e.g. self-consistency)
        current_scores = np.zeros(self.batch_size) # The raw potential (phi)
        raw_improvements = np.zeros(self.batch_size) # The raw I (improvement)
        is_won_array = np.zeros(self.batch_size, dtype=bool)
        
        # Ensure batch_previous_potentials is synced
        if len(self.batch_previous_potentials) != self.batch_size:
            self.batch_previous_potentials = [0.0] * self.batch_size

        # 2. Calculate Raw Scores (Phi) and Raw Improvements (I)
        for i, reflection_text in enumerate(text_actions):
            task_desc = self.tasks[i]
            current_trajectory = self.last_trajectories[i] if i < len(self.last_trajectories) else ""
            
            # Get the baseline (Phi_{t-1})
            prev_phi = self.batch_previous_potentials[i]
            # --- Extract Actual Success First ---
            actual_success = bool(infos[i].get('won', False))
            is_won_array[i] = actual_success
            current_phi = 0.0

            
            # ... (JSON Parsing Logic - same as before) ...
            try:
                # --- JSON Extraction ---
                json_str = ""
                code_block_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', reflection_text, re.DOTALL)
                if code_block_match:
                    json_str = code_block_match.group(1)
                else:
                    clean_text = reflection_text.strip()
                    start_idx = clean_text.find('{')
                    end_idx = clean_text.rfind('}')
                    if start_idx != -1 and end_idx != -1:
                        json_str = clean_text[start_idx:end_idx+1]
                
                if not json_str: raise ValueError("No JSON found")
                
                reflection_data = json.loads(json_str)
                
                # --- Subtask Scoring (Phi calculation) ---
                subtasks = reflection_data.get('subtasks', [])
                total_subtasks = len(subtasks)
                completed_subtasks = sum(
                    1 for task in subtasks 
                    if isinstance(task, dict) and task.get('status', '').strip().lower() == 'completed'
                )
                
                # Calculate subtask-based potential
                subtask_phi = completed_subtasks / total_subtasks if total_subtasks > 0 else 0.0
                
                # --- DETERMINE CURRENT PHI ---
                if self.potential_based_on_binary_success:
                    # STRICT MODE: Only actual success matters for potential
                    current_phi = 1.0 if actual_success else 0.0
                else:
                    # DEFAULT MODE: Use subtasks, but override if actual success
                    current_phi = subtask_phi
                    if actual_success:
                        current_phi = 1.0

                # --- Reflection Consistency Reward (Auxiliary) ---
                predicted_success = reflection_data.get('task_success', False)
                if isinstance(predicted_success, str):
                    predicted_success = predicted_success.lower() in ['true', '1', 'yes']
                
                current_reward = 10.0 if predicted_success == actual_success and json_str else 0.0
                reflect_rewards.append(current_reward)
                
                # --- Memory Saving Logic (Same as before) ---
                if predicted_success == actual_success and json_str:
                    action_lesson = reflection_data.get('action_lesson')
                    nav_lesson = reflection_data.get('navigation_lesson')
                    lessons_to_save = []
                    if action_lesson and len(str(action_lesson)) > 5: lessons_to_save.append(f"Action Insight: {action_lesson}")
                    if nav_lesson and len(str(nav_lesson)) > 5: lessons_to_save.append(f"Navigation Insight: {nav_lesson}")
                    
                    if lessons_to_save:
                        final_lesson = " | ".join(lessons_to_save)
                        self.reflection_memory.add(
                            task_description=task_desc,
                            reflection_text=final_lesson,
                            trajectory=current_trajectory,
                            initial_score=0.5,
                            attempt_type="success" if actual_success else "failure",
                            current_progress_ratio=self.current_progress_ratio
                        )

            except Exception as e:
                print(f"Error task {i}: {e}")
                reflect_rewards.append(0.0)
                # Fallback logic for Phi on error
                if self.potential_based_on_binary_success:
                    current_phi = 1.0 if actual_success else 0.0
                else:
                    current_phi = 0.0


            # --- Calculate Raw Improvement (I) ---
            current_scores[i] = current_phi
            # Improvement is strictly positive gain over history
            improvement = max(0.0, current_phi - prev_phi)
            raw_improvements[i] = improvement

        # 3. Group-Relative Normalization & Baseline Update
        num_unique_tasks = self.batch_size // self.group_n
        final_intrinsic_rewards = np.zeros(self.batch_size)

        for group_idx in range(num_unique_tasks):
            start_idx = group_idx * self.group_n
            end_idx = start_idx + self.group_n
            
            task_desc = self.tasks[start_idx]
            
            # Extract group data
            group_improvements = raw_improvements[start_idx:end_idx]
            # group_scores = current_scores[start_idx:end_idx]
            
            # A. Normalization (Centering)
            # As per LaTeX Eq (8): R_int = I - Mean(I)
            if self.group_relative_intrinsic_rewards:
                group_mean_imp = np.mean(group_improvements)
                # Note: We do NOT divide by std here, just centering is sufficient 
                # to maintain the zero-sum property for the intrinsic component.
                centered_improvements = group_improvements - group_mean_imp
                final_intrinsic_rewards[start_idx:end_idx] = centered_improvements
            else:
                final_intrinsic_rewards[start_idx:end_idx] = group_improvements

            group_success_rate = np.mean(is_won_array[start_idx:end_idx].astype(float))
            old_baseline = self.task_potential_history.get(task_desc, 0.0)

            if group_success_rate > old_baseline:
                self.task_potential_history[task_desc] = group_success_rate
            # B. Update Historical Baseline (EMA)
            # As per LaTeX Eq (9): Phi_t = gamma * Phi_{t-1} + (1-gamma) * Mean(Phi_t)
            # if len(group_scores) > 0:
            #     current_group_mean_score = np.mean(group_scores)
            #     old_baseline = self.task_potential_history.get(task_desc, 0.0)

            #     if current_group_mean_score > old_baseline:
            #         self.task_potential_history[task_desc] = current_group_mean_score
                # # EMA Update
                # new_baseline = (self.ema_gamma * old_baseline) + ((1 - self.ema_gamma) * current_group_mean_score)
                # self.task_potential_history[task_desc] = new_baseline

        print("raw_improvements: ", raw_improvements)
        print("final_intrinsic_rewards (centered): ", final_intrinsic_rewards)
        infos = copy.deepcopy(infos)
        for info in infos:
            info['is_action_valid'] = to_numpy(True)
        # Convert to numpy for compatibility
        return None, to_numpy(reflect_rewards), to_numpy(final_intrinsic_rewards), None, copy.deepcopy(infos), to_numpy(current_scores)

    def step(self, text_actions: List[str]):
        actions, valids = self.projection_f(text_actions)
        next_obs, rewards, dones, infos = self.envs.step(actions)

        next_obs = self.format_obs(next_obs)

        self.memory.store({'text_obs': self.pre_text_obs, 'action': actions, 'reward': rewards, 'dones': dones, 'won': [info['won'] for info in infos]})
        self.pre_text_obs = next_obs

        next_observations = {
            'text': self.build_text_obs(next_obs, infos),
            'image': None,
            'anchor': next_obs.copy()
        }
        
        for i, info in enumerate(infos):
            info['is_action_valid'] = to_numpy(valids[i])

        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos

    def extract_task(self, text_obs: List[str]):
        tasks = []
        for obs in text_obs:
            parts = obs.split(" [SEP] ")
            if len(parts) > 2 and parts[1] == 'Instruction:':
                tasks.append(parts[2])
            else:
                tasks.append(obs)
        return tasks
    
    def format_obs(self, text_obs):
        postprocess_text_obs = []
        for i in range(len(text_obs)):
            parts = text_obs[i].split(" [SEP] ")
            try:
                index = parts.index(self.tasks[i])
                reformatted_obs = " [SEP] ".join(f"'{p}'" for p in parts[index+1:])
            except (ValueError, IndexError):
                reformatted_obs = text_obs[i]

            postprocess_text_obs.append(reformatted_obs)

        return postprocess_text_obs
    
    def format_avail_actions(self, avail):
        actions = []
        for key in avail.keys():
            if key not in ["has_search_bar", "clickables"]:
                raise ValueError(f"Unknown key in available actions: {key}")

        if avail["has_search_bar"]:
            actions.append("search[<your query>]")

        for txt in avail["clickables"]:
            actions.append(f"click[{txt}]")

        return actions
            
    def build_text_obs(self, text_obs: List[str], infos: List[List[str]], init: bool = False) -> List[str]:
        postprocess_text_obs = []
        
        if not init and self.config.env.history_length > 0:
            memory_contexts, valid_lens = self.memory.fetch(
                    self.config.env.history_length,
                    obs_key="text_obs",
                    action_key="action")
        else:
            memory_contexts = [""] * len(text_obs)
            valid_lens = [0] * len(text_obs)
            
        for i in range(len(text_obs)):
            available_actions = self.format_avail_actions(infos[i]['available_actions'])
            reformatted_available_actions = "\n".join(f"'{s}'," for s in available_actions)
            
            if i < len(self.current_reflections):
                reflection_context = self.current_reflections[i]
            else:
                reflection_context = ""

            if init or self.config.env.history_length <= 0:
                obs = WEBSHOP_TEMPLATE_NO_HIS.format(
                    reflections=reflection_context, 
                    task_description=self.tasks[i],
                    current_observation=text_obs[i],
                    available_actions=reformatted_available_actions
                )
            else:
                obs = WEBSHOP_TEMPLATE.format(
                    reflections=reflection_context, 
                    task_description=self.tasks[i],
                    step_count=len(self.memory[i]),
                    history_length=valid_lens[i],
                    action_history=memory_contexts[i],
                    current_step=len(self.memory[i]) + 1,
                    current_observation=text_obs[i],
                    available_actions=reformatted_available_actions
                )
                
                if len(obs) > 13000:
                    obs = WEBSHOP_TEMPLATE_NO_HIS.format(
                        reflections=reflection_context, 
                        task_description=self.tasks[i],
                        current_observation=text_obs[i],
                        available_actions=reformatted_available_actions
                    )

            postprocess_text_obs.append(obs)

        return postprocess_text_obs

    def build_reflect_text_obs(self, infos: List[str]) -> List[str]:
        postprocess_text_obs = []
        # memory_contexts, valid_lens = self.memory.fetch(
        #         15,
        #         obs_key="text_obs",
        #         action_key="action")
        memory_contexts, valid_lens = self.memory.fetch(
                15,
                obs_key="text_obs",
                action_key="action",
                max_to_show=6)
        # self.task_trajectory_history[task] = {"successful": [], "failed": []}
        for i in range(len(infos)):
            task = self.tasks[i]
            # Ensure key exists (it should from reset, but safety first)
            if task not in self.task_trajectory_history:
                self.task_trajectory_history[task] = {"successful": [], "failed": []}
                
            if infos[i].get("won", False):
                self.task_trajectory_history[task]["successful"].append(memory_contexts[i])
            else:
                self.task_trajectory_history[task]["failed"].append(memory_contexts[i])

        # --- CRITICAL: Store these so step_reflect can access them ---
        self.last_trajectories = memory_contexts
        for i in range(len(infos)):
            task = self.tasks[i]
            is_won = infos[i].get("won", False)
            
            # Determine success string and select Contrastive Reference
            # If we WON, we want to see a FAIL to understand what to avoid (or just compare)
            # If we LOST, we want to see a SUCCESS to understand what to do
            
            reference_traj_str = "No reference history available yet."
            if self.success_reference_induction:
                if is_won:
                    SUCCESS = "successfully"
                else: 
                    SUCCESS = "unsuccessfully"
                # Try to get a successful example
                success_hist = self.task_trajectory_history[task]["successful"]
                if success_hist:
                    # Use the most recent success
                    reference_traj_str = "Reference Successful Trajectory (for comparison):\n" + success_hist[-1]
                else:
                    reference_traj_str = "No successful attempts available for reference."
            else:
                if is_won:
                    SUCCESS = "successfully"
                    # Try to get a failed example
                    failed_hist = self.task_trajectory_history[task]["failed"]
                    if failed_hist:
                        # Use the most recent failure
                        reference_traj_str = "Reference Failed Trajectory (for comparison):\n" + failed_hist[-1]
                    else:
                        reference_traj_str = "No failed attempts available for comparison."
                else:
                    SUCCESS = "unsuccessfully" # Changed from "NOT successfully" for better grammar
                    # Try to get a successful example
                    success_hist = self.task_trajectory_history[task]["successful"]
                    if success_hist:
                        # Use the most recent success
                        reference_traj_str = "Reference Successful Trajectory (for comparison):\n" + success_hist[-1]
                    else:
                        reference_traj_str = "No successful attempts available for reference."

            obs = WEBSHOP_REFLECT_TEMPLATE.format(
                task_description=task,
                success=SUCCESS,
                reference_trajectory=reference_traj_str,
                current_trajectory=memory_contexts[i]
            )
            # obs = WEBSHOP_REFLECT_TEMPLATE.format(
            #     task_description=self.tasks[i],
            #     current_trajectory=memory_contexts[i]
            #     )
            postprocess_text_obs.append(obs)
            
        return postprocess_text_obs

    def success_evaluator(self, *args, **kwargs) -> Dict[str, np.ndarray]:
        total_infos = kwargs['total_infos']
        total_batch_list = kwargs['total_batch_list']
        reflect_rewards = kwargs.get('reflect_rewards', None)

        batch_size = len(total_batch_list)
        success = defaultdict(list)
        
        for bs in range(batch_size):
            r_reward = None
            if reflect_rewards is not None:
                try:
                    r_reward = reflect_rewards[bs]
                except IndexError:
                    r_reward = 0.0
            
            self._process_batch(bs, total_batch_list, total_infos, success, reflect_reward=r_reward)
        
        return {key: np.array(value) for key, value in success.items()}

    def _process_batch(self, batch_idx, total_batch_list, total_infos, success, reflect_reward=None):
            if reflect_reward is not None:
                val = float(reflect_reward.item()) if hasattr(reflect_reward, 'item') else float(reflect_reward)
                success['reflect_success_rate'].append(val)
            elif 'reflect_success_rate' in success:
                success['reflect_success_rate'].append(0.0)

            found_active_step = False
            for i in reversed(range(len(total_batch_list[batch_idx]))):
                batch_item = total_batch_list[batch_idx][i]
                if batch_item['active_masks']:
                    info = total_infos[batch_idx][i]
                    won_value = float(info.get('won', 0.0))
                    score_value = float(info.get('task_score', 0.0))
                    
                    success['success_rate'].append(won_value)
                    success['webshop_task_score (not success_rate)'].append(score_value)
                    found_active_step = True
                    return

            if not found_active_step:
                success['success_rate'].append(0.0)
                success['webshop_task_score (not success_rate)'].append(0.0)

class MineSweeperEnvironmentManager(EnvironmentManagerBase):
    def __init__(self, envs, projection_f, config, retrieve_type=None):
        self.n_mines = config.env.minesweeper.n_mines
        self.board_size = config.env.minesweeper.board_size
        self.memory = SimpleMemory()
        
        # Group and evaluation configuration for reflection
        self.group_n = config.env.rollout.n  # e.g., 8
        # Extract reflection memory hyperparameters from config
        mem_config = config.env.get('reflection_memory', {})
        filepath = mem_config.get('filepath', "minesweeper_reflections.json")
        import os
        if os.path.dirname(filepath):
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
        alpha = mem_config.get('alpha', 0.7)
        beta = mem_config.get('beta', 0.05)
        temp = mem_config.get('temperature', 0.5)
        ucb_scale = mem_config.get('ucb_scale', 1.0)
        self.top_k = mem_config.get('top_k', 1)
        
        self.memory_start_cutoff = mem_config.get('memory_start_cutoff', 0.0) 
        self.current_progress_ratio = 0.0 # Track progress internally
        self.retrieve_mode = mem_config.get('retrieve_mode', 'both')
        self.enable_memory = mem_config.get('enable_memory', True)
        self.group_outperformance = mem_config.get('group_outperformance', False)
        self.full_group_memory = mem_config.get('full_group_memory', False)
        self.group_relative_intrinsic_rewards = mem_config.get('group_relative_intrinsic_rewards', False)

        self.potential_based_on_binary_success = mem_config.get('potential_based_on_binary_success', False)
        self.single_reflection_per_group = mem_config.get('single_reflection_per_group', False)
        # EMA Decay rate for the baseline (matches LaTeX gamma)
        self.ema_gamma = 0.9
        print("memory retrieve_type: ", retrieve_type)
        print("memory retrieve_mode: ", self.retrieve_mode)
        print("top_k_retrieved_memory: ", self.top_k)
        print(f"Memory Start Cutoff: {self.memory_start_cutoff}") 
        print(f"Global Memory Retrieval Enabled: {self.enable_memory}")
        print(f"Single Reflection Per Group: {self.single_reflection_per_group}")
        print(f"Potential Based On Binary Success Only: {self.potential_based_on_binary_success}")

        # Initialize persistent reflection memory
        self.reflection_memory = ReflectionMemory(
            filepath=filepath,
            alpha=alpha,
            beta=beta,
            temperature=temp,
            retrieve_type=retrieve_type,
            ucb_scale=ucb_scale
        )
        self.task_trajectory_history = {}
        self.task_potential_history = {} 
        self.batch_previous_potentials = [] 
        
        
        # Initialize containers for retrieval tracking
        self.current_reflections = []       # Formatted strings for the prompt
        self.retrieved_raw_reflections = []  # List of lists of raw strings for utility updates
        self.init_states = []
        self.current_retrieval_types = []
        # Store the trajectories generated during the reflection phase
        # so they can be saved to memory in step_reflect
        self.last_trajectories = []        
        super().__init__(envs, projection_f, config)

    def update_training_progress(self, current_step: int, total_steps: int):
        """
        Updates the environment with the current training progress.
        This triggers memory pruning if a 20% milestone is reached.
        """
        if total_steps > 0:
            self.current_progress_ratio = current_step / total_steps
            
            # self.reflection_memory.check_and_prune(progress_ratio=ratio, top_k=3)

    def reset(self, kwargs):
        if kwargs is None:
            kwargs = {}
        
        # Determine mode based on kwargs
        is_eval = not kwargs.get('is_train', True)
        # print("is_eval:", is_eval)

        obs, infos = self.envs.reset()
        self.init_states = obs
        assert len(self.init_states) == len(infos)
        
        # print("obs[0]: ", obs[0])
        # print("infos[0]: ", infos[0])
        self.pre_text_obs = obs
        
        self.memory.reset(batch_size = len(infos))
        self.batch_size = len(obs)
        assert self.batch_size % self.group_n == 0, "Batch size must be divisible by group size"
        self.num_unique_tasks = self.batch_size // self.group_n

        self.current_reflections = []
        self.retrieved_raw_reflections = []
        self.batch_previous_potentials = []
        self.current_retrieval_types = [] 
        self.batch_retrieved_types = [] # Reset the type tracker
        in_warmup_period = (not is_eval) and (self.current_progress_ratio <= self.memory_start_cutoff)
        
        if in_warmup_period:
            # Optional: Log occasionally if needed
            print(f"Warmup Phase: Progress {self.current_progress_ratio:.2f} <= Cutoff {self.memory_start_cutoff}. Memory Disabled.")
            pass 
        group_split_index = self.group_n // 2
        if self.full_group_memory:
            group_split_index = 0
        for i, task in enumerate(self.init_states):
            prev_potential = self.task_potential_history.get(task, 0.0)
            self.batch_previous_potentials.append(prev_potential)
            formatted_reflections = ""
            raw_list_of_dicts = [] # This will hold [{'text':..., 'type':...}]
            current_types_list = [] # List to hold types for this specific agent

            should_retrieve = False
            retrieval_type_str = "control"
            if self.enable_memory:
                if in_warmup_period:
                    # Explicitly disable retrieval during warmup
                    should_retrieve = False
                elif is_eval:
                    # During Eval: Everyone retrieves (or based on config)
                    should_retrieve = True
                    retrieval_type_str = "eval_retrieval"
                else:
                    position_in_group = i % self.group_n
                    if position_in_group >= group_split_index:
                        should_retrieve = True
                        retrieval_type_str = "experiment"
                    else:
                        should_retrieve = False
            else:
                should_retrieve = False
            
            if should_retrieve:
                # Retrieve top_k items
                k = self.top_k if is_eval else 1
                raw_list_of_dicts = self.reflection_memory.retrieve(
                    current_task_description=task, 
                    top_k=k, 
                    filter_type=self.retrieve_mode
                )
                if raw_list_of_dicts:
                    formatted_lines = []
                    for item in raw_list_of_dicts:
                        r_text = item.get('text', '')
                        r_type = item.get('type', 'unknown')
                        
                        # Store the type for logging
                        current_types_list.append(r_type)
                        
                        formatted_lines.append(r_text)
                    
                    formatted_reflections = "Past reflections on similar tasks:\n" + "\n".join(formatted_lines)
                    formatted_reflections += "\nWarning: These lessons may be outdated. Use them only if they align with your current observation."
            
            
            self.current_reflections.append(formatted_reflections)
            self.retrieved_raw_reflections.append(raw_list_of_dicts)
            self.current_retrieval_types.append(retrieval_type_str)
            self.batch_retrieved_types.append(current_types_list)
            print("retrieved_raw_reflections: ", self.retrieved_raw_reflections)
            print("current_reflections: ", self.current_reflections)
            # --- NEW: Inject types into infos immediately ---
            infos[i]['reflection_types'] = current_types_list
            infos[i]['retrieval_group'] = retrieval_type_str
        
        # -----------------------------------------------------------
        assert len(self.current_reflections) == len(self.init_states)
        # Now it is safe to build observations
        observations = {
            'text': self.build_text_obs(infos, obs, init=True),
            'image': None, 
            'anchor': obs
        }

        return observations, infos

    def step(self, text_actions: List[str]):
        # print("text_actions: ", text_actions)
        actions, valids = self.projection_f(text_actions)
        next_obs, rewards, dones, infos = self.envs.step(actions)

        # add action_valid to infos
        for i, info in enumerate(infos):
            info['is_action_valid'] = to_numpy(valids[i])

        self.memory.store({
                            'text_obs': self.pre_text_obs,
                            'action': actions, 
                            'reward': rewards,
                            'dones': dones,
                            'won': [info['won'] for info in infos]
                        })
        
        self.pre_text_obs = next_obs
        next_observations = {
            'text': self.build_text_obs(infos, next_obs), 
            'image': None, 
            'anchor': next_obs
        }

        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos

    def reflect(self, infos: List[Dict]):
        """
        Called at the end of the 'play' phase.
        Updates utility based on Group B (Retrieved) vs Group A (Not Retrieved) performance.
        """
        observations = {
            'text': self.build_reflect_text_obs(infos),
            'image': None,
            'anchor': self.build_reflect_text_obs(infos)
        }
        
        # Ensure action validity is set for all
        for info in infos:
            info['is_action_valid'] = to_numpy(True)
        
        batch_size = len(self.init_states)
        assert batch_size == len(infos)
        
        # Ensure batch size is divisible by group_n
        if batch_size % self.group_n != 0:
            print(f"WARNING: Batch size {batch_size} not divisible by group_n {self.group_n}")
        
        num_groups = batch_size // self.group_n
        group_split_index = self.group_n // 2
        
        # Iterate over each group independently
        for g in range(num_groups):
            start_idx = g * self.group_n
            end_idx = start_idx + self.group_n
            mid_idx = start_idx + group_split_index
            
            # Calculate wins for control group (first half)
            control_wins = 0
            for i in range(start_idx, mid_idx):
                if infos[i].get("won", False):
                    control_wins += 1
            
            # Calculate wins for experiment group (second half)
            experiment_wins = 0
            for i in range(mid_idx, end_idx):
                if infos[i].get("won", False):
                    experiment_wins += 1
            
            # 3. Determine Utility Score for THIS group
            group_outperformed = experiment_wins > control_wins
            
            # 4. Update Memory ONLY for the Experiment agents
            for i in range(mid_idx, end_idx):
                task_desc = self.init_states[i]
                raw_reflection_items = self.retrieved_raw_reflections[i] # List[Dict]
                is_success = infos[i].get("won", False)
                
                # 1.0 if the agent won AND the retrieval group beat the control group.
                if self.group_outperformance:
                    if is_success and group_outperformed:
                        utility_score = 1.0
                    else:
                        utility_score = 0.0
                else:
                    if is_success:
                        utility_score = 1.0
                    else:
                        utility_score = 0.0
            
                if raw_reflection_items:
                    for item in raw_reflection_items:
                        # --- UPDATED: Extract text from dict for utility update ---
                        reflection_text = item.get('text', '')
                        if reflection_text:
                            self.reflection_memory.update_utility(
                                task_description=task_desc, 
                                reflection_text=reflection_text, 
                                score=utility_score
                            )
        
        return observations, infos

    def step_reflect(self, text_actions: List[str], infos: List[Dict]):
        """
        Calculates intrinsic rewards based on improvement over an EMA baseline,
        normalizes them group-wise, and updates the baseline.
        Adapted for MINESWEEPER_REFLECT_TEMPLATE.
        """
        import json
        import re
        import copy
        import numpy as np
        
        def to_numpy(x):
            return np.array(x) if not isinstance(x, np.ndarray) else x

        print("text_actions for reflection:", text_actions)
        
        # 1. Initialize Containers
        reflect_rewards = [] # This is the immediate reward for the reflection step itself (e.g. self-consistency)
        current_scores = np.zeros(self.batch_size) # The raw potential (phi)
        raw_improvements = np.zeros(self.batch_size) # The raw I (improvement)
        is_won_array = np.zeros(self.batch_size, dtype=bool)
        
        # Ensure batch_previous_potentials is synced
        if len(self.batch_previous_potentials) != self.batch_size:
            self.batch_previous_potentials = [0.0] * self.batch_size

        # 2. Calculate Raw Scores (Phi) and Raw Improvements (I)
        for i, reflection_text in enumerate(text_actions):
            task_desc = self.init_states[i]
            current_trajectory = self.last_trajectories[i] if i < len(self.last_trajectories) else ""
            
            # Get the baseline (Phi_{t-1})
            prev_phi = self.batch_previous_potentials[i]
            # --- Extract Actual Success First ---
            actual_success = bool(infos[i].get('won', False))
            is_won_array[i] = actual_success
            current_phi = 0.0
            try:
                # --- JSON Extraction ---
                json_str = ""
                code_block_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', reflection_text, re.DOTALL)
                if code_block_match:
                    json_str = code_block_match.group(1)
                else:
                    clean_text = reflection_text.strip()
                    start_idx = clean_text.find('{')
                    end_idx = clean_text.rfind('}')
                    if start_idx != -1 and end_idx != -1:
                        json_str = clean_text[start_idx:end_idx+1]
                
                if not json_str: raise ValueError("No JSON found")
                
                reflection_data = json.loads(json_str)
                
                # 1. Trust but verify the trajectory_value against the subtasks list
                subtasks = reflection_data.get('subtasks', [])
                total_subtasks = len(subtasks)
                completed_subtasks = sum(
                    1 for task in subtasks 
                    if isinstance(task, dict) and task.get('status', '').strip().lower() == 'completed'
                )
                
                # Calculate subtask-based potential
                subtask_phi = completed_subtasks / total_subtasks if total_subtasks > 0 else 0.0
                
                # --- DETERMINE CURRENT PHI ---
                if self.potential_based_on_binary_success:
                    # STRICT MODE: Only actual success matters for potential
                    current_phi = 1.0 if actual_success else 0.0
                else:
                    # DEFAULT MODE: Use subtasks, but override if actual success
                    current_phi = subtask_phi
                    if actual_success:
                        current_phi = 1.0

                # --- Reflection Consistency Reward (Auxiliary) ---
                predicted_success = reflection_data.get('task_success', False)
                if isinstance(predicted_success, str):
                    predicted_success = predicted_success.lower() in ['true', '1', 'yes']
                
                current_reward = 10.0 if predicted_success == actual_success and json_str else 0.0
                reflect_rewards.append(current_reward)
                # --- Memory Saving Logic (Same as before) ---
                if predicted_success == actual_success and json_str:
                    next_priority = reflection_data.get('next_priority')
                    lessons_to_save = []
                    if next_priority and len(str(next_priority)) > 5:
                        lessons_to_save.append(f"New Plan: {next_priority}")
                    
                    if lessons_to_save:
                        final_lesson = " | ".join(lessons_to_save)
                        self.reflection_memory.add(
                            task_description=task_desc,
                            reflection_text=final_lesson,
                            trajectory=current_trajectory,
                            initial_score=0.5,
                            attempt_type="success" if actual_success else "failure",
                            current_progress_ratio=self.current_progress_ratio
                        )
            except Exception as e:
                print(f"Error task {i}: {e}")
                reflect_rewards.append(0.0)
                # Fallback logic for Phi on error
                if self.potential_based_on_binary_success:
                    current_phi = 1.0 if actual_success else 0.0
                else:
                    current_phi = 0.0

            # --- Calculate Raw Improvement (I) ---
            current_scores[i] = current_phi
            # Improvement is strictly positive gain over history
            improvement = max(0.0, current_phi - prev_phi)
            raw_improvements[i] = improvement
                    
        # 3. Group-Relative Normalization & Baseline Update
        num_unique_tasks = self.batch_size // self.group_n
        final_intrinsic_rewards = np.zeros(self.batch_size)

        for group_idx in range(num_unique_tasks):
            start_idx = group_idx * self.group_n
            end_idx = start_idx + self.group_n
            
            task_desc = self.init_states[start_idx]
            
            # Extract group data
            group_improvements = raw_improvements[start_idx:end_idx]
            # group_scores = current_scores[start_idx:end_idx]
            
            # A. Normalization (Centering)
            # As per LaTeX Eq (8): R_int = I - Mean(I)
            if self.group_relative_intrinsic_rewards:
                group_mean_imp = np.mean(group_improvements)
                # Note: We do NOT divide by std here, just centering is sufficient 
                # to maintain the zero-sum property for the intrinsic component.
                centered_improvements = group_improvements - group_mean_imp
                final_intrinsic_rewards[start_idx:end_idx] = centered_improvements
            else:
                final_intrinsic_rewards[start_idx:end_idx] = group_improvements

            group_success_rate = np.mean(is_won_array[start_idx:end_idx].astype(float))
            old_baseline = self.task_potential_history.get(task_desc, 0.0)

            if group_success_rate > old_baseline:
                self.task_potential_history[task_desc] = group_success_rate
            # B. Update Historical Baseline (EMA)
            # As per LaTeX Eq (9): Phi_t = gamma * Phi_{t-1} + (1-gamma) * Mean(Phi_t)
            # if len(group_scores) > 0:
            #     current_group_mean_score = np.mean(group_scores)
            #     old_baseline = self.task_potential_history.get(task_desc, 0.0)

            #     if current_group_mean_score > old_baseline:
            #         self.task_potential_history[task_desc] = current_group_mean_score
                # # EMA Update
                # new_baseline = (self.ema_gamma * old_baseline) + ((1 - self.ema_gamma) * current_group_mean_score)
                # self.task_potential_history[task_desc] = new_baseline

        print("raw_improvements: ", raw_improvements)
        print("final_intrinsic_rewards (centered): ", final_intrinsic_rewards)
        infos = copy.deepcopy(infos)
        for info in infos:
            info['is_action_valid'] = to_numpy(True)

        # Convert to numpy for compatibility
        return None, to_numpy(reflect_rewards), to_numpy(final_intrinsic_rewards), None, copy.deepcopy(infos), to_numpy(current_scores)

    def build_text_obs(self, infos, text_obs: List[str]=None, init: bool = False) -> List[str]:
        """
        This function builds the text observation for the agent.
        """
        postprocess_text_obs = []

        if not init and self.config.env.history_length > 0:
            memory_contexts, valid_lens = self.memory.fetch(
                self.config.env.history_length,
                obs_key="text_obs",
                action_key="action"
            )

        for i in range(len(infos)):
            if init or self.config.env.history_length <= 0:
                obs = MINESWEEPER_TEMPLATE_NO_HIS.format(
                    board_size=self.board_size,
                    n_mines=self.n_mines,
                    reflections=self.current_reflections[i],  # Add reflections
                    current_observation=text_obs[i],
                )
            else:
                obs = MINESWEEPER_TEMPLATE.format(
                    board_size=self.board_size,
                    n_mines=self.n_mines,
                    reflections=self.current_reflections[i],  # Add reflections
                    step_count=len(self.memory[i]),
                    history_length=valid_lens[i],
                    action_history=memory_contexts[i],
                    current_step=len(self.memory[i]) + 1,
                    current_observation=text_obs[i],
                )
            postprocess_text_obs.append(obs)
        
        print("postprocessed_text_obs [0]:", postprocess_text_obs[0])
        return postprocess_text_obs

    def build_reflect_text_obs(self, infos: List[Dict]) -> List[str]:
        """
        This function builds the text observation for the agent during reflection.
        """
        postprocess_text_obs = []
        memory_contexts, valid_lens = self.memory.fetch(
            15,  # Get full game history for reflection
            obs_key="text_obs",
            action_key="action"
        )
        # self.task_trajectory_history[task] = {"successful": [], "failed": []}
        for i in range(len(infos)):
            task = self.init_states[i]
            # Ensure key exists (it should from reset, but safety first)
            if task not in self.task_trajectory_history:
                self.task_trajectory_history[task] = {"successful": [], "failed": []}
                
            if infos[i].get("won", False):
                self.task_trajectory_history[task]["successful"].append(memory_contexts[i])
            else:
                self.task_trajectory_history[task]["failed"].append(memory_contexts[i])

        # --- CRITICAL: Store these so step_reflect can access them ---
        self.last_trajectories = memory_contexts

        for i in range(len(infos)):
            task = self.init_states[i]
            is_won = infos[i].get("won", False)
            
            # Determine success string and select Contrastive Reference
            # If we WON, we want to see a FAIL to understand what to avoid (or just compare)
            # If we LOST, we want to see a SUCCESS to understand what to do
            
            reference_traj_str = "No reference history available yet."
            
            if is_won:
                SUCCESS = "successfully"
                # Try to get a failed example
                failed_hist = self.task_trajectory_history[task]["failed"]
                if failed_hist:
                    # Use the most recent failure
                    reference_traj_str = "Reference Failed Trajectory (for comparison):\n" + failed_hist[-1]
                else:
                    reference_traj_str = "No failed attempts available for comparison."
            else:
                SUCCESS = "unsuccessfully" # Changed from "NOT successfully" for better grammar
                # Try to get a successful example
                success_hist = self.task_trajectory_history[task]["successful"]
                if success_hist:
                    # Use the most recent success
                    reference_traj_str = "Reference Successful Trajectory (for comparison):\n" + success_hist[-1]
                else:
                    reference_traj_str = "No successful attempts available for reference."
            obs = MINESWEEPER_REFLECT_TEMPLATE.format(
                board_size=self.board_size,
                n_mines=self.n_mines,
                success=SUCCESS,
                reference_trajectory=reference_traj_str,
                current_trajectory=memory_contexts[i]
            )
            postprocess_text_obs.append(obs)
        
        if len(postprocess_text_obs) > 0:
            print("processed_reflect_text [0]:", postprocess_text_obs[0])
        
        return postprocess_text_obs

    def success_evaluator(self, *args, **kwargs) -> Dict[str, np.ndarray]:
        """
        Evaluate if the episodes are successful or not.
        """
        from collections import defaultdict
        
        total_infos = kwargs['total_infos']
        total_batch_list = kwargs['total_batch_list']
        reflect_rewards = kwargs.get('reflect_rewards', None)
        
        batch_size = len(total_batch_list)
        success = defaultdict(list)
        
        for bs in range(batch_size):
            r_reward = None
            if reflect_rewards is not None:
                try:
                    r_reward = reflect_rewards[bs]
                except IndexError:
                    r_reward = 0.0
            
            self._process_batch(bs, total_batch_list, total_infos, success, reflect_reward=r_reward)
        
        assert len(success['success_rate']) == batch_size
        
        return {key: np.array(value) for key, value in success.items()}
    
    def _process_batch(self, batch_idx, total_batch_list, total_infos, success, reflect_reward=None):
        """
        Process a single batch trajectory to extract success metrics.
        """
        # Process reflection phase
        if reflect_reward is not None:
            if hasattr(reflect_reward, 'item'):
                val = float(reflect_reward.item())
            else:
                val = float(reflect_reward)
            success['reflect_success_rate'].append(val)
        else:
            success['reflect_success_rate'].append(0.0)
        
        # Process play phase
        play_success_found = False
        trajectory = total_batch_list[batch_idx]
        
        for i in reversed(range(len(trajectory))):
            batch_item = trajectory[i]
            
            if not batch_item.get('active_masks', True):
                continue
            
            phase = batch_item.get('phase', 'play')
            
            if phase == 'play':
                info = total_infos[batch_idx][i]
                won_value = float(info.get('won', 0.0))
                
                success['play_success_rate'].append(won_value)
                success['success_rate'].append(won_value)
                
                # Add Minesweeper-specific success metrics
                if self.board_size and self.n_mines:
                    difficulty = f"minesweeper_{self.board_size}x{self.board_size}_{self.n_mines}mines"
                    success[f"{difficulty}_success_rate"].append(won_value)
                
                play_success_found = True
                break
        
        if not play_success_found:
            success['play_success_rate'].append(0.0)
            success['success_rate'].append(0.0)


class AppWorldEnvironmentManager(EnvironmentManagerBase):
    def __init__(self, envs, projection_f, config):
        self.memory = SimpleMemory()
        super().__init__(envs, projection_f, config)
    
    def reset(self, kwargs):
        text_obs, infos = self.envs.reset()
        
        self.supervisors = [info['supervisor'] for info in infos]
        self.memory.reset(batch_size = len(text_obs))
        self.tasks = text_obs.copy()
        self.pre_text_obs = text_obs

        full_text_obs = self.build_text_obs(text_obs, init=True)
        return {'text': full_text_obs, 'image': None, 'anchor': text_obs}, infos
    
    def step(self, text_actions: List[str]):
        actions, valids = self.projection_f(text_actions)

        text_obs, rewards, dones, infos = self.envs.step(actions)

        self.memory.store({'text_obs': text_obs, 'action': actions})
        self.pre_text_obs = text_obs

        full_text_obs = self.build_text_obs(text_obs)

        # add action_valid to infos
        for i, info in enumerate(infos):
            info['is_action_valid'] = to_numpy(valids[i])

        next_observations = {'text': full_text_obs, 'image': None, 'anchor': text_obs}
        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos
    

    def build_text_obs(self, text_obs: List[str], init: bool = False) -> List[str]:
        """
        This function builds the text observation for the agent.
        """
        postprocess_text_obs = []
        if init and self.supervisors is not None:
            for i in range(len(text_obs)):
                obs = APPWORLD_TEMPLATE_NO_HIS.format(
                        supervisor_first_name=self.supervisors[i]['first_name'],
                        supervisor_last_name=self.supervisors[i]['last_name'],
                        supervisor_email=self.supervisors[i]['email'],
                        supervisor_phone_number=self.supervisors[i]['phone_number'],
                        task_description=self.tasks[i],
                    )
                postprocess_text_obs.append(obs)
        else:
            for i in range(len(text_obs)):
                # Get last `history_length` steps
                recent_history = self.memory[i][-self.config.env.history_length:]
                valid_history_length = len(recent_history)
                start_index = len(self.memory[i]) - valid_history_length
                action_history = ""
                for j, record in enumerate(recent_history):
                    step_number = start_index + j + 1
                    action = record["action"]
                    env_obs = record["text_obs"]
                    action_history += f"\nCode {step_number}: \n{action}\n\nResult {step_number}: \n{env_obs}\n"
                
                if len(action_history) > 10000:
                    action_history = "... " + action_history[-10000:]

                obs = APPWORLD_TEMPLATE.format(
                        supervisor_first_name=self.supervisors[i]['first_name'],
                        supervisor_last_name=self.supervisors[i]['last_name'],
                        supervisor_email=self.supervisors[i]['email'],
                        supervisor_phone_number=self.supervisors[i]['phone_number'],
                        task_description=self.tasks[i],
                        step_count=len(self.memory[i]),
                        history_length=valid_history_length,
                        action_history=action_history.strip(),
                        current_step=len(self.memory[i]) + 1,
                        current_observation=text_obs[i],
                    )
                postprocess_text_obs.append(obs)
        return postprocess_text_obs

def make_envs(config):
    """
    Create enviroments 
    """ 
    # check if config.env.rollout.n is an integer
    if not isinstance(config.env.rollout.n, int):
        raise ValueError("config.env.rollout.n should be an integer")
    group_n = config.env.rollout.n if config.env.rollout.n > 0 else 1
    resources_per_worker = OmegaConf.to_container(config.env.resources_per_worker, resolve=True)

    if "search" in config.env.env_name.lower():
        from agent_system.environments.env_package.search import build_search_envs, search_projection
        _envs = build_search_envs(seed=config.env.seed, env_num=config.data.train_batch_size, group_n=group_n, is_train=True, env_config=config.env)
        _val_envs = build_search_envs(seed=config.env.seed + 1000, env_num=config.data.val_batch_size, group_n=1, is_train=False, env_config=config.env)

        projection_f = partial(search_projection)
        envs = SearchEnvironmentManager(_envs, projection_f, config)
        val_envs = SearchEnvironmentManager(_val_envs, projection_f, config)
        return envs, val_envs
    elif "gym_cards" in config.env.env_name.lower():
        from agent_system.environments.env_package.gym_cards import build_gymcards_envs, gym_projection
        _envs = build_gymcards_envs(env_name=config.env.env_name, seed=config.env.seed, env_num=config.data.train_batch_size, group_n=group_n, is_train=True, resources_per_worker=resources_per_worker)
        _val_envs = build_gymcards_envs(env_name=config.env.env_name, seed=config.env.seed + 1000, env_num=config.data.val_batch_size, group_n=1, is_train=False, resources_per_worker=resources_per_worker)
        
        projection_f = partial(gym_projection, env_name=config.env.env_name)
        envs = GymCardEnvironmentManager(_envs, projection_f, config)
        val_envs = GymCardEnvironmentManager(_val_envs, projection_f, config)
        return envs, val_envs
    elif "alfworld" in config.env.env_name.lower():
        from agent_system.environments.env_package.alfworld import build_alfworld_envs, alfworld_projection
        if config.env.env_name == 'alfworld/AlfredThorEnv':
            alf_config_path = os.path.join(os.path.dirname(__file__), 'env_package/alfworld/configs/config_tw.yaml')
        elif config.env.env_name == 'alfworld/AlfredTWEnv':
            alf_config_path = os.path.join(os.path.dirname(__file__), 'env_package/alfworld/configs/config_tw.yaml')
        else:
            raise ValueError(f"Unsupported environment: {config.env.env_name}")

        env_kwargs = {
            'eval_dataset': config.env.alfworld.eval_dataset, # 'eval_in_distribution' or 'eval_out_of_distribution'
        }
        _envs = build_alfworld_envs(alf_config_path, config.env.seed, config.data.train_batch_size, group_n, is_train=True, env_kwargs=env_kwargs, resources_per_worker=resources_per_worker)
        _val_envs = build_alfworld_envs(alf_config_path, config.env.seed + 1000, config.data.val_batch_size, 1, is_train=False, env_kwargs=env_kwargs, resources_per_worker=resources_per_worker)
        
        projection_f = partial(alfworld_projection)
        # print("config information: ", config)
        # Ensure 'config' is passed as the 3rd argument to both constructors
        envs = AlfWorldEnvironmentManager(
            _envs, 
            projection_f, 
            config, 
            config.env.train_retrieve_type
        )
        
        val_envs = AlfWorldEnvironmentManager(
            _val_envs, 
            projection_f, 
            config, 
            config.env.eval_retrieve_type
        )
        # --- FIX END ---
        return envs, val_envs

    elif "sokoban" in config.env.env_name.lower():
        from agent_system.environments.env_package.sokoban import build_sokoban_envs, sokoban_projection
        env_kwargs = {
            'dim_room': config.env.sokoban.dim_room,
            'num_boxes': config.env.sokoban.num_boxes,
            'max_steps': config.env.max_steps,
            'search_depth': config.env.sokoban.search_depth,
            'min_steps': config.env.get('min_steps', 5),  # default to 3 if not specified
            'max_sol_steps': config.env.get('max_sol_steps', config.env.max_steps) 
        }
        _envs = build_sokoban_envs(seed=config.env.seed, env_num=config.data.train_batch_size, group_n=group_n, mode=config.env.sokoban.mode, is_train=True, env_kwargs=env_kwargs, resources_per_worker=resources_per_worker)
        _val_envs = build_sokoban_envs(seed=config.env.seed + 1000, env_num=config.data.val_batch_size, group_n=1, mode=config.env.sokoban.mode, is_train=False, env_kwargs=env_kwargs, resources_per_worker=resources_per_worker)
        
        projection_f = partial(sokoban_projection)
        envs = SokobanEnvironmentManager(_envs, projection_f, config, config.env.train_retrieve_type)
        val_envs = SokobanEnvironmentManager(_val_envs, projection_f, config, config.env.eval_retrieve_type)
        return envs, val_envs

    elif "minesweeper" in config.env.env_name.lower():
        from agent_system.environments.env_package.minesweeper import build_minesweeper_envs, minesweeper_projection
        env_kwargs = {
            "board_size": config.env.minesweeper.board_size,  # e.g., 8 for 8x8 board
            "n_mines": config.env.minesweeper.n_mines,
            "board_type": config.env.minesweeper.board_type
        }
        _envs = build_minesweeper_envs(seed=config.env.seed, env_num=config.data.train_batch_size, group_n=group_n, is_train=True, env_kwargs=env_kwargs, resources_per_worker=resources_per_worker)
        _val_envs = build_minesweeper_envs(seed=config.env.seed + 1000, env_num=config.data.val_batch_size, group_n=1, is_train=False, env_kwargs=env_kwargs, resources_per_worker=resources_per_worker)

        projection_f = partial(minesweeper_projection)
        envs = MineSweeperEnvironmentManager(_envs, projection_f, config, config.env.train_retrieve_type)
        val_envs = MineSweeperEnvironmentManager(_val_envs, projection_f, config, config.env.eval_retrieve_type)
        return envs, val_envs

    elif "webshop" in config.env.env_name.lower():
        from agent_system.environments.env_package.webshop import build_webshop_envs, webshop_projection
        if config.env.webshop.use_small:
            file_path = os.path.join(os.path.dirname(__file__), 'env_package/webshop/webshop/data/items_shuffle_1000.json')
            attr_path = os.path.join(os.path.dirname(__file__), 'env_package/webshop/webshop/data/items_ins_v2_1000.json')
        else:
            file_path = os.path.join(os.path.dirname(__file__), 'env_package/webshop/webshop/data/items_shuffle.json')
            attr_path = os.path.join(os.path.dirname(__file__), 'env_package/webshop/webshop/data/items_ins_v2.json')
        env_kwargs = {
                    'observation_mode': 'text', 
                    'num_products': None, 
                    'human_goals': config.env.webshop.human_goals,
                    'file_path': file_path,
                    'attr_path': attr_path
                    }
        _envs = build_webshop_envs(seed=config.env.seed, env_num=config.data.train_batch_size, group_n=group_n, is_train=True, env_kwargs=env_kwargs, resources_per_worker=resources_per_worker)
        _val_envs = build_webshop_envs(seed=config.env.seed + 1000, env_num=config.data.val_batch_size, group_n=1, is_train=False, env_kwargs=env_kwargs, resources_per_worker=resources_per_worker)

        projection_f = partial(webshop_projection)
        # envs = WebshopEnvironmentManager(_envs, projection_f, config)
        # val_envs = WebshopEnvironmentManager(_val_envs, projection_f, config)
        envs = WebshopEnvironmentManager(
            _envs, 
            projection_f, 
            config, 
            config.env.train_retrieve_type
        )
        val_envs = WebshopEnvironmentManager(
            _val_envs, 
            projection_f, 
            config, 
            config.env.eval_retrieve_type
        )
        import time
        time.sleep((config.data.train_batch_size * group_n + config.data.val_batch_size) * 0.1) # wait for the envs to be ready
        return envs, val_envs

    elif "appworld" in config.env.env_name.lower():
        from agent_system.environments.env_package.appworld import build_appworld_envs, appworld_projection
        _envs = build_appworld_envs(dataset_name='train', seed=config.env.seed, env_num=config.data.train_batch_size, group_n=group_n, start_server_id=0, resources_per_worker=resources_per_worker)
        _val_envs = build_appworld_envs(dataset_name='test_normal', seed=config.env.seed + 1000, env_num=config.data.val_batch_size, group_n=1, start_server_id=config.data.train_batch_size*group_n, resources_per_worker=resources_per_worker)
        
        projection_f = partial(appworld_projection)
        envs = AppWorldEnvironmentManager(_envs, projection_f, config)
        val_envs = AppWorldEnvironmentManager(_val_envs, projection_f, config)
        return envs, val_envs
    else:
        print("Environment not supported")
        exit(1)