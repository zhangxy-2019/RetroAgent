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

import torch
import numpy as np
from verl import DataProto
from verl.utils.dataset.rl_dataset import collate_fn
from verl.utils.model import compute_position_id_with_mask
import verl.utils.torch_functional as verl_F
from transformers import PreTrainedTokenizer
import uuid
from agent_system.multi_turn_rollout.utils import process_image, to_list_of_dict, torch_to_numpy, filter_group_data
from agent_system.environments import EnvironmentManagerBase
from typing import List, Dict, Optional
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
import json
import collections

class TrajectoryCollector:
    def __init__(self, config, tokenizer: PreTrainedTokenizer, processor=None):
        """
        Initialize the TrajectoryProcessor class.
        
        Parameters:
            config: Configuration object containing data processing settings
            tokenizer (PreTrainedTokenizer): Tokenizer for text encoding and decoding
            processor: Image processor for multimodal inputs
        """
        self.config = config
        self.step_gamma = config.algorithm.get('step_gamma', 0.95)
        self.traj_gamma = config.algorithm.get('traj_gamma', 0.6)
        # self.enable_intrinsic_rewards = config.algorithm.get('enable_intrinsic_rewards', False)
        self.intrinsic_reward_coefficient = config.algorithm.get('intrinsic_reward_coefficient', -1)
        # <<< CHANGE: Add config for credit assignment >>>
        self.enable_credit_assignment = config.algorithm.get('credit_assignment', False)
        # <<< END CHANGE >>>
        # <<< CHANGE: Add config for reflection policy >>>
        self.use_ref_policy_for_reflection = config.algorithm.get('reflection_reference_policy', False)
        # <<< END CHANGE >>>
        self.tokenizer = tokenizer
        self.processor = processor

    def _calculate_reflection_coefficient(self, current_step: int, total_steps: int) -> float:
        """
        Calculates the reflection coefficient.
        If hard_cutoff is enabled, the coefficient decays to ~0.001 by the cutoff point 
        and is 0.0 thereafter to ensure a smooth transition.
        """
        if total_steps <= 0:
            return 1.0
            
        progress = current_step / total_steps
        progress = max(0.0, min(1.0, progress))
        
        hard_cutoff = self.config.algorithm.get('intrinsic_hard_cutoff', False)
        
        if hard_cutoff:
            cutoff_point = 0.10
            if progress > cutoff_point:
                return 0.0
            normalized_progress = progress / cutoff_point
            alpha = 6.9
            coefficient = np.exp(-alpha * normalized_progress)
        else:
            alpha = 5.0
            coefficient = np.exp(-alpha * progress)
        
        return float(coefficient)

    def preprocess_single_sample(
        self,
        item: int,
        gen_batch: DataProto,
        obs: Dict,
    ):
        """
        Process a single observation sample, organizing environment observations (text and/or images) 
        into a format processable by the model.
        """
        raw_prompt = gen_batch.non_tensor_batch['raw_prompt'][item]
        data_source = gen_batch.non_tensor_batch['data_source'][item]
        apply_chat_template_kwargs = self.config.data.get("apply_chat_template_kwargs", {})
        
        obs_texts = obs.get('text', None)
        obs_images = obs.get('image', None)
        obs_anchors = obs.get('anchor', None)
        obs_text = obs_texts[item] if obs_texts is not None else None
        obs_image = obs_images[item] if obs_images is not None else None
        obs_anchor = obs_anchors[item] if obs_anchors is not None else None
        is_multi_modal = obs_image is not None

        _obs_anchor = torch_to_numpy(obs_anchor, is_object=True) if isinstance(obs_anchor, torch.Tensor) else obs_anchor

        obs_content = ''
        if obs_text is not None:
            obs_content += obs_text
        else:
            print(f"Warning: No text observation found!", flush=True)

        chat = np.array([{
            "content": obs_content,
            "role": "user",
        }])
        
        prompt_with_chat_template = self.tokenizer.apply_chat_template(
            chat,
            add_generation_prompt=True,
            tokenize=False,
            **apply_chat_template_kwargs
        )
        
        row_dict = {}
        
        if is_multi_modal:
            raw_prompt = prompt_with_chat_template.replace('<image>', '<|vision_start|><|image_pad|><|vision_end|>')
            row_dict['multi_modal_data'] = {'image': [process_image(obs_image)]}
            image_inputs = self.processor.image_processor(row_dict['multi_modal_data']['image'], return_tensors='pt')
            image_grid_thw = image_inputs['image_grid_thw']
            row_dict['multi_modal_inputs'] = {key: val for key, val in image_inputs.items()}
            if image_grid_thw is not None:
                merge_length = self.processor.image_processor.merge_size**2
                index = 0
                while '<image>' in prompt_with_chat_template:
                    prompt_with_chat_template = prompt_with_chat_template.replace(
                        '<image>',
                        '<|vision_start|>' + '<|placeholder|>' * (image_grid_thw[index].prod() // merge_length) +
                        '<|vision_end|>',
                        1,
                    )
                    index += 1
                prompt_with_chat_template = prompt_with_chat_template.replace('<|placeholder|>',
                                                                                self.processor.image_token)
        else:
            raw_prompt = prompt_with_chat_template
        
        input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(
            prompt=prompt_with_chat_template,
            tokenizer=self.tokenizer,
            max_length=self.config.data.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.config.data.truncation,
        )

        if is_multi_modal:
            if "Qwen3VLProcessor" in self.processor.__class__.__name__:
                from verl.models.transformers.qwen3_vl import get_rope_index
            else:
                from verl.models.transformers.qwen2_vl import get_rope_index

            vision_position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids[0],
                image_grid_thw=image_grid_thw,
                attention_mask=attention_mask[0],
            )
            valid_mask = attention_mask[0].bool()
            text_position_ids = torch.ones((1, len(input_ids[0])), dtype=torch.long)
            text_position_ids[0, valid_mask] = torch.arange(valid_mask.sum().item())
            position_ids = [torch.cat((text_position_ids, vision_position_ids), dim=0)]
        else:
            position_ids = compute_position_id_with_mask(attention_mask)

        raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.config.data.max_prompt_length:
            if self.config.data.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.config.data.max_prompt_length:]
            elif self.config.data.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[:self.config.data.max_prompt_length]
            elif self.config.data.truncation == "middle":
                left_half = self.config.data.max_prompt_length // 2
                right_half = self.config.data.max_prompt_length - left_half
                raw_prompt_ids = raw_prompt_ids[:left_half] + raw_prompt_ids[-right_half:]
            elif self.config.data.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.config.data.max_prompt_length}.")

        row_dict.update({
            'input_ids': input_ids[0],
            'attention_mask': attention_mask[0],
            'position_ids': position_ids[0],
            'raw_prompt_ids': raw_prompt_ids,
            'anchor_obs': _obs_anchor,
            'index': item,
            'data_source': data_source
        })

        if self.config.data.get('return_raw_chat', False):
            row_dict['raw_prompt'] = chat.tolist()
        
        return row_dict

    def preprocess_batch(
        self,
        gen_batch: DataProto, 
        obs: Dict, 
    ) -> DataProto:
        """
        Process a batch of observation samples, converting environment observations into model-processable format.
        """
        batch_size = len(gen_batch.batch['input_ids'])
        processed_samples = []
        
        for item in range(batch_size):
            processed = self.preprocess_single_sample(
                item=item,
                gen_batch=gen_batch,
                obs=obs,
            )
            processed_samples.append(processed)
        
        batch = collate_fn(processed_samples)
        
        new_batch = DataProto.from_single_dict(
            data=batch,
            meta_info=gen_batch.meta_info
        )

        return new_batch

    def gather_rollout_data(
            self,
            total_batch_list: List[List[Dict]],
            episode_rewards: np.ndarray,
            discounted_returns: np.ndarray,
            episode_lengths: np.ndarray,
            success: Dict[str, np.ndarray],
            traj_uid: np.ndarray,
            tool_callings: np.ndarray,
            reflection_types_list: List[List[str]] = None,
            retrieval_groups_list: List[str] = None
            ) -> DataProto:
        """
        Collect and organize trajectory data.

        Parameters:
            total_batch_list (List[List[Dict]]): List of trajectory data for each environment.
            episode_rewards (np.ndarray): 1-D object array of np.ndarray, one per trajectory.
                Each inner array has length == len(total_batch_list[bs]) with the per-step
                episode reward (cumulative play reward assigned to every active step).
            discounted_returns (np.ndarray): 1-D object array of np.ndarray, one per trajectory.
                Each inner array has length == len(total_batch_list[bs]) with the per-step
                discounted return.
            episode_lengths (np.ndarray): Total steps for each environment.
            success (Dict[str, np.ndarray]): Success samples for each environment.
            traj_uid (np.ndarray): Trajectory unique identifiers.
            tool_callings (np.ndarray): Number of tool callings for each environment.
            reflection_types_list (List[List[str]]): Reflection types per trajectory.
            retrieval_groups_list (List[str]): Retrieval group per trajectory.

        Returns:
            DataProto: Collected and organized trajectory data.
        """
        batch_size = len(total_batch_list)

        # Compute aggregate stats from all per-step episode rewards
        all_ep_rewards = np.concatenate([episode_rewards[bs] for bs in range(batch_size)])
        episode_rewards_mean = float(np.mean(all_ep_rewards)) if len(all_ep_rewards) > 0 else 0.0
        episode_rewards_min = float(np.min(all_ep_rewards)) if len(all_ep_rewards) > 0 else 0.0
        episode_rewards_max = float(np.max(all_ep_rewards)) if len(all_ep_rewards) > 0 else 0.0

        episode_lengths_mean = float(np.mean(episode_lengths))
        episode_lengths_min = float(np.min(episode_lengths))
        episode_lengths_max = float(np.max(episode_lengths))

        success_rate = {}
        for key, value in success.items():
            success_rate[key] = np.mean(value)

        effective_batch = []
        for bs in range(batch_size):
            current_reflection_types = []
            current_retrieval_group = "unknown"

            if reflection_types_list is not None and bs < len(reflection_types_list):
                current_reflection_types = reflection_types_list[bs]

            if retrieval_groups_list is not None and bs < len(retrieval_groups_list):
                current_retrieval_group = retrieval_groups_list[bs]

            ep_rew_arr = episode_rewards[bs]       # np.ndarray of shape (num_steps,)
            disc_ret_arr = discounted_returns[bs]   # np.ndarray of shape (num_steps,)

            for t, data in enumerate(total_batch_list[bs]):
                assert traj_uid[bs] == data['traj_uid'], "data is not from the same trajectory"
                if data['active_masks']:
                    # Per-step credit-assigned values
                    data['episode_rewards'] = float(ep_rew_arr[t])
                    data['step_returns'] = torch.tensor(float(disc_ret_arr[t]), dtype=torch.float32)
                    # Aggregate stats
                    data['episode_rewards_mean'] = episode_rewards_mean
                    data['episode_rewards_min'] = episode_rewards_min
                    data['episode_rewards_max'] = episode_rewards_max
                    data['episode_lengths'] = episode_lengths[bs]
                    data['episode_lengths_mean'] = episode_lengths_mean
                    data['episode_lengths_min'] = episode_lengths_min
                    data['episode_lengths_max'] = episode_lengths_max
                    data['tool_callings'] = tool_callings[bs]
                    data['reflection_types'] = current_reflection_types
                    data['retrieval_group'] = current_retrieval_group
                    for key, value in success_rate.items():
                        data[key] = value

                    effective_batch.append(data)

        gen_batch_output = DataProto.from_single_dict(
            data=collate_fn(effective_batch)
        )
        return gen_batch_output

    def vanilla_multi_turn_loop(
            self,
            gen_batch: DataProto, 
            actor_rollout_wg, 
            envs: EnvironmentManagerBase,
            current_training_steps: int,
            total_training_steps: int,
            is_train: bool = True,
            ref_rollout_wg=None,
            ) -> DataProto:
        """
        Collects trajectories through parallel agent-environment agent_loop.
        """
        if total_training_steps > 0:
            if hasattr(envs, 'update_training_progress'):
                envs.update_training_progress(current_training_steps, total_training_steps)
            elif hasattr(envs, 'env') and hasattr(envs.env, 'update_training_progress'):
                envs.env.update_training_progress(current_training_steps, total_training_steps)

        batch_size = len(gen_batch.batch)

        env_kwargs = gen_batch.non_tensor_batch.pop('env_kwargs', {})
        if env_kwargs is None:
            env_kwargs = {}
        env_kwargs['is_train'] = is_train

        obs, infos = envs.reset(kwargs=env_kwargs)

        lenght_obs = len(obs['text']) if obs['text'] is not None else len(obs['image'])
        assert len(gen_batch.batch) == lenght_obs, f"gen_batch size {len(gen_batch.batch)} does not match obs size {lenght_obs}"

        if self.config.env.rollout.n > 0:
            uid_batch = []
            for i in range(batch_size):
                if i % self.config.env.rollout.n == 0:
                    uid = str(uuid.uuid4())
                uid_batch.append(uid)
            uid_batch = np.array(uid_batch, dtype=object)
        else:
            uid = str(uuid.uuid4())
            uid_batch = np.array([uid for _ in range(len(gen_batch.batch))], dtype=object)

        is_done = np.zeros(batch_size, dtype=bool)
        traj_uid = np.array([str(uuid.uuid4()) for _ in range(batch_size)], dtype=object)
        total_batch_list = [[] for _ in range(batch_size)]
        total_infos = [[] for _ in range(batch_size)]
        episode_lengths = np.zeros(batch_size, dtype=np.float32)
        episode_rewards = np.zeros(batch_size, dtype=np.float32)
        tool_callings = np.zeros(batch_size, dtype=np.float32)
        reflect_rewards = np.zeros(batch_size, dtype=np.float32)
        reflect_rewards_step = np.zeros(batch_size, dtype=np.float32)
        reflect_rewards_before_clipping = np.zeros(batch_size, dtype=np.float32)
        extrinsic_episode_rewards = np.zeros(batch_size, dtype=np.float32)
        final_intrinsic_rewards = np.zeros(batch_size, dtype=np.float32)

        trajectory_reflection_types = [[] for _ in range(batch_size)]
        trajectory_retrieval_groups = ["unknown"] * batch_size
        unknown_reflections = np.zeros(batch_size, dtype=np.float32)
        failure_reflections = np.zeros(batch_size, dtype=np.float32)
        success_reflections = np.zeros(batch_size, dtype=np.float32)
        # Initialize total_reflect_batch_list to avoid NameError if is_train=False
        total_reflect_batch_list = []

        for i, info in enumerate(infos):
            if 'reflection_types' in info:
                trajectory_reflection_types[i] = info['reflection_types']
                if "failure" in info['reflection_types']:
                    failure_reflections[i] += 1
                elif "success" in info['reflection_types']:
                    success_reflections[i] += 1
            if 'retrieval_group' in info:
                trajectory_retrieval_groups[i] = info['retrieval_group']

        phase = 'play'
        for _step in range(self.config.env.max_steps):
            active_masks = np.logical_not(is_done)

            batch = self.preprocess_batch(gen_batch=gen_batch, obs=obs)

            batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
            non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
            if "multi_modal_data" in batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("multi_modal_data")
            if "raw_prompt" in batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("raw_prompt")
            if "tools_kwargs" in batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("tools_kwargs")
            batch_input = batch.pop(
                batch_keys=batch_keys_to_pop,
                non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
            )

            batch_input.meta_info = gen_batch.meta_info
            batch_input_padded, pad_size = pad_dataproto_to_divisor(batch_input, actor_rollout_wg.world_size)
            batch_output_padded = actor_rollout_wg.generate_sequences(batch_input_padded)
            batch_output = unpad_dataproto(batch_output_padded, pad_size=pad_size)

            batch.non_tensor_batch['uid'] = uid_batch
            batch.non_tensor_batch['traj_uid'] = traj_uid
            batch.non_tensor_batch['phase'] = [phase] * batch_size
            batch = batch.union(batch_output)

            text_actions = self.tokenizer.batch_decode(batch.batch['responses'], skip_special_tokens=True)

            next_obs, rewards, dones, infos = envs.step(text_actions)

            if len(rewards.shape) == 2:
                rewards = rewards.squeeze(1)
            if len(dones.shape) == 2:
                dones = dones.squeeze(1)

            if 'is_action_valid' in infos[0]:
                batch.non_tensor_batch['is_action_valid'] = np.array([info['is_action_valid'] for info in infos], dtype=bool)
            else:
                batch.non_tensor_batch['is_action_valid'] = np.ones(batch_size, dtype=bool)

            if 'tool_calling' in infos[0]:
                tool_callings[active_masks] += np.array([info['tool_calling'] for info in infos], dtype=np.float32)[active_masks]

            episode_rewards[active_masks] += torch_to_numpy(rewards)[active_masks]
            episode_lengths[active_masks] += 1

            assert len(rewards) == batch_size
            batch.non_tensor_batch['rewards'] = torch_to_numpy(rewards, is_object=True)
            batch.non_tensor_batch['active_masks'] = torch_to_numpy(active_masks, is_object=True)

            batch_list: list[dict] = to_list_of_dict(batch)

            for i in range(batch_size):
                total_batch_list[i].append(batch_list[i])
                total_infos[i].append(infos[i])

            is_done = np.logical_or(is_done, dones)
            obs = next_obs

            if is_done.all():
                break

        extrinsic_episode_rewards = episode_rewards.copy()

        # --- Phase 2: Reflection (Training Only) ---
        if is_train:
            phase = 'reflect'
            obs, infos = envs.reflect(infos)
            reflect_batch = self.preprocess_batch(gen_batch=gen_batch, obs=obs)
            batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
            non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
            if "multi_modal_data" in reflect_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("multi_modal_data")
            if "raw_prompt" in reflect_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("raw_prompt")
            if "tools_kwargs" in reflect_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("tools_kwargs")
            batch_input = reflect_batch.pop(
                batch_keys=batch_keys_to_pop,
                non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
            )

            # <<< CHANGE: Configure reflection generation policy >>>
            # If reflection_reference_policy is True, we use ref_rollout_wg (if available)
            # or rely on the trainer to pass the correct worker group with a flag.
            # In the updated RayPPOTrainer, we passed ref_rollout_wg=actor_rollout_wg
            # but we need to signal the hybrid engine to use reference weights.
            
            if self.use_ref_policy_for_reflection and ref_rollout_wg is not None:
                reflect_wg = ref_rollout_wg
                # Create a copy of meta_info to avoid mutating the original
                reflect_meta_info = dict(gen_batch.meta_info) if gen_batch.meta_info else {}
                reflect_meta_info['use_ref_policy'] = True
                batch_input.meta_info = reflect_meta_info
                # print("utilizing reference policy for reflection .....")
            else:
                reflect_wg = actor_rollout_wg
                batch_input.meta_info = gen_batch.meta_info
                # print("utilizing actor policy for reflection .....")
            # <<< END CHANGE >>>

            reflect_batch_input_padded, pad_size = pad_dataproto_to_divisor(batch_input, reflect_wg.world_size)
            reflect_batch_output_padded = reflect_wg.generate_sequences(reflect_batch_input_padded)
            reflect_batch_output = unpad_dataproto(reflect_batch_output_padded, pad_size=pad_size)
            # 'rollout_log_probs': rollout_log_probs, # we will recompute old log prob with actor
            reflect_batch.non_tensor_batch['uid'] = uid_batch
            reflect_batch.non_tensor_batch['traj_uid'] = traj_uid
            reflect_batch.non_tensor_batch['phase'] = [phase] * batch_size
            
            reflect_batch = reflect_batch.union(reflect_batch_output)

            text_actions = self.tokenizer.batch_decode(reflect_batch.batch['responses'], skip_special_tokens=True)

            _, reflect_rewards_step, final_intrinsic_rewards, _, reflect_infos, completion_percentages = envs.step_reflect(text_actions, infos)
            if len(reflect_rewards_step.shape) == 2:
                reflect_rewards_step = reflect_rewards_step.squeeze(1)

            if len(final_intrinsic_rewards.shape) == 2:
                final_intrinsic_rewards = final_intrinsic_rewards.squeeze(1)

            if self.intrinsic_reward_coefficient > -1:
                reflection_coeff = self.intrinsic_reward_coefficient
            else:
                reflection_coeff = self._calculate_reflection_coefficient(current_training_steps, total_training_steps)

            reflect_rewards = final_intrinsic_rewards * reflection_coeff
            reflect_rewards_before_clipping = reflect_rewards.copy()

            instrinsic_rewards_upper_clipping_ratio = self.config.algorithm.get('instrinsic_rewards_upper_clipping_ratio', -5)
            instrinsic_rewards_lower_clipping_ratio = self.config.algorithm.get('instrinsic_rewards_lower_clipping_ratio', -5)
            if instrinsic_rewards_upper_clipping_ratio > -1:
                reflect_rewards = np.clip(reflect_rewards, -instrinsic_rewards_lower_clipping_ratio, instrinsic_rewards_upper_clipping_ratio)
            # 2. Collect Reflection Data for Training (REINFORCE)
            reflect_batch.non_tensor_batch['rewards'] = torch_to_numpy(reflect_rewards, is_object=True)
            reflect_batch.non_tensor_batch['active_masks'] = torch_to_numpy(np.ones(batch_size, dtype=bool), is_object=True)
            
            # For reflection data, we treat the reflection step as the episode reward
            reflect_batch.non_tensor_batch['episode_rewards'] = torch_to_numpy(reflect_rewards)
            
            # [CRITICAL ADDITION] Add metadata to reflect batch to match play batch schema for union
            # Reflection is typically 1 step, so episode length is 1.
            reflect_batch.non_tensor_batch['episode_lengths'] = torch_to_numpy(np.ones(batch_size, dtype=np.float32))
            # Extrinsic reward for reflection is 0 (or equal to reward if considered extrinsic, but usually 0 for distinction)
            reflect_batch.non_tensor_batch['extrinsic_episode_rewards'] = torch_to_numpy(np.zeros(batch_size, dtype=np.float32))
            # Tool callings for reflection (usually 0)
            reflect_batch.non_tensor_batch['tool_callings'] = torch_to_numpy(np.zeros(batch_size, dtype=np.float32))
            reflect_batch.non_tensor_batch['is_action_valid'] = np.ones(batch_size, dtype=bool)
                        
            reflect_batch_list_dicts = to_list_of_dict(reflect_batch)
            total_reflect_batch_list.extend(reflect_batch_list_dicts)

            self.get_reflect_logs(
                reflect_obs=obs["text"],
                text_actions=text_actions,
                trajectory_success=episode_rewards,
                reflect_rewards=reflect_rewards_step,
                completion_percentages=completion_percentages,
                current_training_steps=current_training_steps,
                total_training_steps=total_training_steps,
                reflect_rewards_before_clipping=reflect_rewards_before_clipping,
                reflect_rewards_after_clipping=reflect_rewards
            )
            for i in range(batch_size):
                episode_rewards[i] += reflect_rewards[i]

        # --- Prepare for Reflection Success Tracking ---
        reflection_type_stats = collections.defaultdict(list)
        retrieval_group_rewards = collections.defaultdict(list)
        for i in range(batch_size):
            current_group = trajectory_retrieval_groups[i]
            retrieval_group_rewards[current_group].append(extrinsic_episode_rewards[i])

        for i in range(batch_size):
            current_traj_r_types = trajectory_reflection_types[i]

            for step_data in total_batch_list[i]:
                step_data['reflect_rewards_before_clipping'] = reflect_rewards_before_clipping[i]
                step_data['reflect_reward'] = reflect_rewards[i]
                step_data['raw_reflect_reward'] = reflect_rewards_step[i]
                step_data['intrinsic_reward'] = final_intrinsic_rewards[i]
                step_data['extrinsic_episode_reward'] = extrinsic_episode_rewards[i]
                step_data["failure_reflection"] = failure_reflections[i]
                step_data["success_reflection"] = success_reflections[i]
                step_data['retrieval_group'] = trajectory_retrieval_groups[i]
                for group_name, rewards_list in retrieval_group_rewards.items():
                    if len(rewards_list) > 0:
                        step_data[f'extrinsic_reward_{group_name}'] = np.mean(rewards_list)

            current_traj_success = 0.0
            for step_idx in reversed(range(len(total_batch_list[i]))):
                batch_item = total_batch_list[i][step_idx]
                if batch_item['active_masks']:
                    info = total_infos[i][step_idx]
                    current_traj_success = float(info.get('won', 0.0))
                    break

            r_types = trajectory_reflection_types[i]
            if not r_types:
                reflection_type_stats['none'].append(current_traj_success)
            else:
                for r_type in r_types:
                    reflection_type_stats[r_type].append(current_traj_success)

        self.get_traj_cot_logs(total_batch_list, current_training_steps, total_training_steps, trajectory_reflection_types, trajectory_retrieval_groups)

        success: Dict[str, np.ndarray] = envs.success_evaluator(
            total_infos=total_infos,
            total_batch_list=total_batch_list,
            episode_rewards=episode_rewards,
            episode_lengths=episode_lengths,
            reflect_rewards=reflect_rewards_step,
        )

        return total_batch_list, episode_rewards, episode_lengths, success, traj_uid, tool_callings, trajectory_reflection_types, trajectory_retrieval_groups, total_reflect_batch_list

    def dynamic_multi_turn_loop(
            self,
            gen_batch: DataProto, 
            actor_rollout_wg, 
            envs: EnvironmentManagerBase,
            current_training_steps: int,
            total_training_steps: int,
            ref_rollout_wg=None,
            ) -> DataProto:
        """
        Conduct dynamic rollouts until a target batch size is met.
        """
        total_batch_list = []
        total_episode_rewards = []
        total_episode_lengths = []
        total_success = []
        total_traj_uid = []
        total_tool_callings = []
        total_reflection_types = []
        total_retrieval_groups = []
        total_reflect_batch_list = [] # Initialize accumulator for reflection data
        try_count: int = 0
        max_try_count = self.config.algorithm.filter_groups.max_num_gen_batches

        while len(total_batch_list) < self.config.data.train_batch_size * self.config.env.rollout.n and try_count < max_try_count:

            if len(total_batch_list) > 0:
                print(f"valid num={len(total_batch_list)} < target num={self.config.data.train_batch_size * self.config.env.rollout.n}. Keep generating... ({try_count}/{max_try_count})")
            try_count += 1

            batch_list, episode_rewards, episode_lengths, success, traj_uid, tool_callings, reflection_types, retrieval_groups, reflect_batch_list = self.vanilla_multi_turn_loop(
                gen_batch=gen_batch,
                actor_rollout_wg=actor_rollout_wg,
                envs=envs,
                current_training_steps=current_training_steps,
                total_training_steps=total_training_steps,
                is_train=True,
                ref_rollout_wg=ref_rollout_wg,
            )
            for idx, item_list in enumerate(batch_list):
                for step in item_list:
                    step['_temp_reflection_types'] = reflection_types[idx]
                    step['_temp_retrieval_group'] = retrieval_groups[idx]
            batch_list, episode_rewards, episode_lengths, success, traj_uid, tool_callings = filter_group_data(
                batch_list=batch_list,
                episode_rewards=episode_rewards,
                episode_lengths=episode_lengths,
                success=success,
                traj_uid=traj_uid,
                tool_callings=tool_callings,
                config=self.config,
                last_try=(try_count == max_try_count),
            )
            surviving_reflection_types = []
            surviving_retrieval_groups = []
            for item_list in batch_list:
                if len(item_list) > 0:
                    if '_temp_reflection_types' in item_list[0]:
                        surviving_reflection_types.append(item_list[0]['_temp_reflection_types'])
                    else:
                        surviving_reflection_types.append([])

                    if '_temp_retrieval_group' in item_list[0]:
                        surviving_retrieval_groups.append(item_list[0]['_temp_retrieval_group'])
                    else:
                        surviving_retrieval_groups.append("unknown")

                    for step in item_list:
                        if '_temp_reflection_types' in step: del step['_temp_reflection_types']
                        if '_temp_retrieval_group' in step: del step['_temp_retrieval_group']
                else:
                    surviving_reflection_types.append([])
                    surviving_retrieval_groups.append("unknown")

            total_batch_list += batch_list
            total_episode_rewards.append(episode_rewards)
            total_episode_lengths.append(episode_lengths)
            total_success.append(success)
            total_traj_uid.append(traj_uid)
            total_tool_callings.append(tool_callings)
            # Accumulate reflection data (no filtering applied to reflection usually, or apply if needed)
            if reflect_batch_list:
                total_reflect_batch_list.extend(reflect_batch_list)
            total_reflection_types.extend(surviving_reflection_types)
            total_retrieval_groups.extend(surviving_retrieval_groups)

        total_episode_rewards = np.concatenate(total_episode_rewards, axis=0)
        total_episode_lengths = np.concatenate(total_episode_lengths, axis=0)
        total_success = {key: np.concatenate([success[key] for success in total_success], axis=0) for key in total_success[0].keys()}
        total_traj_uid = np.concatenate(total_traj_uid, axis=0)
        total_tool_callings = np.concatenate(total_tool_callings, axis=0)

        return total_batch_list, total_episode_rewards, total_episode_lengths, total_success, total_traj_uid, total_tool_callings, total_reflection_types, total_retrieval_groups, total_reflect_batch_list

    def multi_turn_loop(
            self,
            gen_batch: DataProto, 
            actor_rollout_wg, 
            envs: EnvironmentManagerBase,
            current_training_steps: int,
            total_training_steps: int,
            is_train: bool = True,
            ref_rollout_wg=None,
            ) -> DataProto:
        """
        Select and run the appropriate rollout loop (dynamic or vanilla).
        """
        if is_train:
            gen_batch = gen_batch.repeat(repeat_times=self.config.env.rollout.n, interleave=True)

        if self.config.algorithm.filter_groups.enable and is_train:
            total_batch_list, total_episode_rewards, total_episode_lengths, total_success, total_traj_uid, total_tool_callings, total_reflection_types, total_retrieval_groups, total_reflect_batch_list  = \
                self.dynamic_multi_turn_loop(
                gen_batch=gen_batch,
                actor_rollout_wg=actor_rollout_wg,
                envs=envs,
                current_training_steps=current_training_steps,
                total_training_steps=total_training_steps,
                ref_rollout_wg=ref_rollout_wg,
            )
        else:
            total_batch_list, total_episode_rewards, total_episode_lengths, total_success, total_traj_uid, total_tool_callings, total_reflection_types, total_retrieval_groups, total_reflect_batch_list  = \
                self.vanilla_multi_turn_loop(
                gen_batch=gen_batch,
                actor_rollout_wg=actor_rollout_wg,
                envs=envs,
                current_training_steps=current_training_steps,
                total_training_steps=total_training_steps,
                is_train=is_train,
                ref_rollout_wg=ref_rollout_wg,
            )

        # Validate dimensions (total_episode_rewards is flat np.ndarray from loop)
        assert len(total_batch_list) == len(total_episode_rewards)
        assert len(total_batch_list) == len(total_episode_lengths)
        assert len(total_batch_list) == len(total_traj_uid)
        assert len(total_batch_list) == len(total_tool_callings)

        # Credit assignment: produces np.ndarray (object dtype) of per-step np.ndarray values
        # <<< CHANGE: Conditional credit assignment >>>
        if self.enable_credit_assignment:
            per_step_episode_rewards, total_discounted_returns = self.credit_assignment(
                total_batch_list, self.step_gamma
            )
        else:
            # If disabled, we assign the final cumulative reward to every step (similar to standard GRPO)
            # and set discounted returns to 0 (or simply equal to reward, depending on algorithm needs, 
            # but usually GRPO uses the full episode reward for all steps).
            per_step_episode_rewards, total_discounted_returns = self.no_credit_assignment(
                total_batch_list, total_episode_rewards
            )
        # <<< END CHANGE >>>

        # Create trajectory data
        gen_batch_output: DataProto = self.gather_rollout_data(
            total_batch_list=total_batch_list,
            episode_rewards=per_step_episode_rewards,
            discounted_returns=total_discounted_returns,
            episode_lengths=total_episode_lengths,
            success=total_success,
            traj_uid=total_traj_uid,
            tool_callings=total_tool_callings,
            reflection_types_list=total_reflection_types,
            retrieval_groups_list=total_retrieval_groups
        )
        print('rollout finished')
        # 2. Create Reflect DataProto (if exists) and Union
        # Only do this if is_train is True and we actually collected reflection data
        reflect_batch_output = None
        if is_train and len(total_reflect_batch_list) > 0:
            print(f"Collecting {len(total_reflect_batch_list)} reflection samples.")
            reflect_batch_output = DataProto.from_single_dict(
                data=collate_fn(total_reflect_batch_list)
            )


        return gen_batch_output, reflect_batch_output

    def credit_assignment(self, total_batch_list, step_gamma=0.95):
        """
        Compute the 1) per-step episode reward and 2) step discounted return
        for each step in the trajectory.  Both outputs are 1-D np.ndarray of
        dtype=object, where each element is itself a np.ndarray(float64) whose
        length equals the number of steps in that trajectory.

        Parameters:
            total_batch_list (List[List[Dict]]): Per-trajectory list of step dicts.
                Each dict should contain 'rewards' (scalar) and 'active_masks' (bool).
            step_gamma (float): Discount factor for future rewards between
                consecutive steps within a trajectory.

        Returns:
            total_episode_rewards (np.ndarray[object]): 1-D object array of length
                num_trajectories.  total_episode_rewards[i] is a np.ndarray of shape
                (num_steps_i,) where every active step receives the cumulative
                (undiscounted) play-phase reward of trajectory i.
            total_discounted_returns (np.ndarray[object]): 1-D object array of length
                num_trajectories.  total_discounted_returns[i] is a np.ndarray of
                shape (num_steps_i,) with the backward-discounted return from each step.
        """
        num_trajectories = len(total_batch_list)
        total_episode_rewards = np.empty(num_trajectories, dtype=object)
        total_discounted_returns = np.empty(num_trajectories, dtype=object)

        for traj_idx, steps in enumerate(total_batch_list):
            n = len(steps)

            if n == 0:
                total_episode_rewards[traj_idx] = np.zeros(0, dtype=np.float64)
                total_discounted_returns[traj_idx] = np.zeros(0, dtype=np.float64)
                continue

            episode_rewards = np.zeros(n, dtype=np.float64)
            discounted_returns = np.zeros(n, dtype=np.float64)

            # --- 1. Compute cumulative (undiscounted) reward for the trajectory ---
            cumulative_reward = 0.0
            for t in range(n):
                step = steps[t]
                if step.get('active_masks', True):
                    reward_val = step.get('rewards', 0.0)
                    if hasattr(reward_val, 'item'):
                        reward_val = reward_val.item()
                    cumulative_reward += float(reward_val)

            # Assign cumulative reward to every active step
            for t in range(n):
                if steps[t].get('active_masks', True):
                    episode_rewards[t] = cumulative_reward

            # --- 2. Compute discounted returns (backward pass) ---
            running_return = 0.0
            for t in reversed(range(n)):
                step = steps[t]
                if not step.get('active_masks', True):
                    discounted_returns[t] = 0.0
                    continue
                reward_val = step.get('rewards', 0.0)
                if hasattr(reward_val, 'item'):
                    reward_val = reward_val.item()
                running_return = float(reward_val) + step_gamma * running_return
                discounted_returns[t] = running_return

            total_episode_rewards[traj_idx] = episode_rewards
            total_discounted_returns[traj_idx] = discounted_returns

        return total_episode_rewards, total_discounted_returns

    # <<< CHANGE: Add helper for no credit assignment >>>
    def no_credit_assignment(self, total_batch_list, final_episode_rewards):
        """
        Assigns the total episode reward to every step, effectively disabling
        per-step credit assignment. This mimics standard GRPO behavior where
        the outcome reward is applied to all tokens/steps.
        """
        num_trajectories = len(total_batch_list)
        total_step_rewards = np.empty(num_trajectories, dtype=object)
        total_discounted_returns = np.empty(num_trajectories, dtype=object)

        for traj_idx, steps in enumerate(total_batch_list):
            n = len(steps)
            if n == 0:
                total_step_rewards[traj_idx] = np.zeros(0, dtype=np.float64)
                total_discounted_returns[traj_idx] = np.zeros(0, dtype=np.float64)
                continue
            
            # Retrieve the final accumulated reward for this trajectory
            # final_episode_rewards is a 1D array of floats
            final_reward = float(final_episode_rewards[traj_idx])

            episode_rewards = np.zeros(n, dtype=np.float64)
            # For no credit assignment, we usually don't use discounted returns in the same way,
            # but to keep data structures consistent, we can just fill it with the final reward
            # or zeros depending on how the algorithm uses it. 
            # If the algorithm uses 'step_returns' for advantage, setting it to final_reward
            # makes it equivalent to outcome supervision.
            discounted_returns = np.zeros(n, dtype=np.float64)

            for t in range(n):
                if steps[t].get('active_masks', True):
                    episode_rewards[t] = final_reward
                    discounted_returns[t] = final_reward 

            total_step_rewards[traj_idx] = episode_rewards
            total_discounted_returns[traj_idx] = discounted_returns

        return total_step_rewards, total_discounted_returns
    # <<< END CHANGE >>>
    def get_reflect_logs(self, reflect_obs, text_actions, trajectory_success, reflect_rewards, completion_percentages, current_training_steps, total_training_steps, reflect_rewards_before_clipping=None, reflect_rewards_after_clipping=None):
        """
        Saves reflection logs to a JSONL file for analysis.
        """
        file_path = self.config.data.get("reflect_log_path", "./reflect_analysis.jsonl")
        print("collecting reflection logs ....")
        if isinstance(reflect_rewards, torch.Tensor):
            reflect_rewards = reflect_rewards.cpu().numpy()

        if completion_percentages is None:
            completion_percentages = [0.0] * len(reflect_obs)
        elif isinstance(completion_percentages, (np.ndarray, torch.Tensor)):
            if isinstance(completion_percentages, torch.Tensor):
                completion_percentages = completion_percentages.cpu().numpy()

        samples_list = []
        current_step = current_training_steps

        for i, (reflect_ob, generated_action, reward, completion, traj_success) in enumerate(zip(reflect_obs, text_actions, reflect_rewards, completion_percentages, trajectory_success)):
            if hasattr(reward, 'item'):
                reward = reward.item()
            if hasattr(completion, 'item'):
                completion = completion.item()

            simple_reflect_log = {
                'trajectory_success': float(traj_success),
                'reflect_observation': reflect_ob,
                'generated_reflection': generated_action,
                'reflect_reward': float(reward),
                'task_completion_percentage': float(completion),
                'reflect_reward_before_clipping': float(reflect_rewards_before_clipping[i]) if reflect_rewards_before_clipping is not None else None,
                'reflect_reward_after_clipping': float(reflect_rewards_after_clipping[i]) if reflect_rewards_after_clipping is not None else None,
            }
            samples_list.append(simple_reflect_log)

        step_log_entry = {
            "step": current_step,
            "total_steps": total_training_steps,
            "reflection_policy": "reference" if self.use_ref_policy_for_reflection else "actor",
            "batch_samples": samples_list
        }

        try:
            with open(file_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(step_log_entry, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"Error saving reflect logs: {e}")

    def get_traj_cot_logs(self, total_batch_list, current_training_steps, total_training_steps, trajectory_reflection_types=None, trajectory_retrieval_groups=None):
        '''
        Collects trajectories in text, assigns accumulated rewards, and saves them to a log file.
        '''
        file_path = self.config.data.get("trajectory_log_path", "./trajectory_analysis.jsonl")
        import os
        if os.path.dirname(file_path):
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

        print(f"Collecting trajectory logs for {len(total_batch_list)} trajectories...", flush=True)

        traj_cot_logs = []

        for traj_idx, traj_batch in enumerate(total_batch_list):
            cot_log = {'reward': 0.0, 'trajectory': '', 'reflection_types': [], 'retrieval_group': 'unknown'}

            if trajectory_reflection_types is not None and traj_idx < len(trajectory_reflection_types):
                cot_log['reflection_types'] = trajectory_reflection_types[traj_idx]

            if trajectory_retrieval_groups is not None and traj_idx < len(trajectory_retrieval_groups):
                cot_log['retrieval_group'] = trajectory_retrieval_groups[traj_idx]

            if not cot_log['reflection_types'] and len(traj_batch) > 0:
                if 'reflection_types' in traj_batch[0]:
                    cot_log['reflection_types'] = traj_batch[0]['reflection_types']

            if cot_log['retrieval_group'] == 'unknown' and len(traj_batch) > 0:
                if 'retrieval_group' in traj_batch[0]:
                    cot_log['retrieval_group'] = traj_batch[0]['retrieval_group']

            action_idx = 0

            for i, step in enumerate(traj_batch):
                input_text = self.tokenizer.decode(step['input_ids'], skip_special_tokens=True)
                input_text = input_text.replace("You are Qwen, created by Alibaba Cloud. You are a helpful assistant", "").split('assistant')[0]

                text_action = self.tokenizer.decode(step['responses'], skip_special_tokens=True)

                if step.get('active_masks', True):
                    cot_log['trajectory'] += f"\n#### step {action_idx} #### \n"
                    cot_log['trajectory'] += f"[Input]\n {input_text.strip()}\n"
                    cot_log['trajectory'] += f"[Response]\n {text_action.strip()}\n"
                    ## rollout_log_prob

                    if hasattr(step['rewards'], 'item'):
                        cot_log['reward'] = float(step['rewards'].item())
                    else:
                        cot_log['reward'] = float(step['rewards'])

                    action_idx += 1

            traj_cot_logs.append(cot_log)

        step_log_entry = {
            "step": current_training_steps,
            "total_steps": total_training_steps,
            "trajectories": traj_cot_logs
        }

        try:
            with open(file_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(step_log_entry, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"Error saving trajectory logs: {e}")

        return traj_cot_logs