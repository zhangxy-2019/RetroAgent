## 🚀 Quick Start

### Clone the Repository

```bash
git clone https://github.com/zhangxy-2019/RetroAgent.git
cd RetroAgent
```

### 2. Install veRL (Base Environment)
We recommend using Conda to manage your environment.

```bash
conda create -n verl-agent python==3.12 -y
conda activate verl-agent

pip3 install vllm==0.11.0
pip3 install flash-attn==2.7.4.post1 --no-build-isolation --no-cache-dir
pip install -e .
```

### 3. Install Supported Environments
⚠️ Important: To run an agent in any of these environments, you must first install and configure the corresponding environment. We strongly recommend installing each environment in its own dedicated conda environment to avoid potential package version conflicts.

### Option B: w/ RL-Trained Self-Reflection
Navigate to the in-context directory and run the desired environment:

cd ./rl_trained_self_reflection

ALFWorld
```bash
conda env create -f agent-alfworld-env.yaml
bash ./examples/grpo_trainer/run_alfworld_retroagent_rl_trained_self_reflection.sh
```

Webshop
```bash
conda env create -f agent-webshop-env.yaml
bash ./examples/grpo_trainer/run_webshop_retroagent_rl_trained_self_reflection.sh
```

Sokoban
```bash
conda env create -f agent-sokoban-env.yaml
bash ./examples/grpo_trainer/run_sokoban_retroagent_rl_trained_self_reflection.sh
```

MineSweeper
```bash
conda env create -f agent-sokoban-env.yaml
bash ./examples/grpo_trainer/run_minesweeper_retroagent_rl_trained_self_reflection.sh
```

## 📝 Citation

If you find our work useful, please consider giving us a ⭐ and citing our paper:

```bibtex
@article{zhang2026retroagent,
  title={RetroAgent: From Solving to Evolving via Retrospective Dual Intrinsic Feedback},
  author={Zhang, Xiaoying and Liu, Zichen and Zhang, Yipeng and Hu, Xia and Shao, Wenqi},
  journal={arXiv preprint arXiv:2603.08561},
  year={2026}
}
