# RetroAgent

**RETROAGENT: From Solving to Evolving via Retrospective Dual Intrinsic Feedback**

[![Paper](https://img.shields.io/badge/Paper-arXiv-red)](https://arxiv.org/abs/XXXX.XXXXX)
[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://github.com/zhangxy-2019/RetroAgent)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> 🚧 **We are intensively preparing for the code release. Stay tuned!**

---

## Overview

**RetroAgent** is an online reinforcement learning framework that empowers LLM-based agents to master complex interactive environments — not just by *solving*, but by *evolving*.

Standard RL paradigms favor static problem-solving over continuous adaptation: agents often converge to suboptimal strategies due to insufficient exploration, while learned knowledge remains implicit within parameters rather than explicitly retrievable. RetroAgent addresses these limitations through a hindsight self-reflection mechanism that produces **dual intrinsic feedback**:

1. **Intrinsic Numerical Feedback** — tracks incremental subtask completion relative to prior attempts, rewarding promising explorations.
2. **Intrinsic Language Feedback** — distills reusable lessons into a memory buffer, retrieved via our proposed **Similarity & Utility-Aware Upper Confidence Bound (SimUtil-UCB)** strategy that balances relevance, utility, and exploration to effectively leverage past experiences.


## Key Results

RetroAgent significantly outperforms existing methods across four challenging agentic tasks, achieving state-of-the-art results. Compared to GRPO-trained agents:

| Task         | Improvement |
|:-------------|:-----------:|
| ALFWorld     | **+18.3%**  |
| WebShop      | **+15.4%**  |
| Sokoban      | **+27.1%**  |
| MineSweeper  | **+8.9%**   |

RetroAgent also exhibits strong test-time adaptation and generalization to out-of-distribution scenarios, validated across two model families.

## Installation

```bash
# Coming soon
git clone https://github.com/zhangxy-2019/RetroAgent.git
cd RetroAgent