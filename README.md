# 🚀 PSO-Tuned Agent for OpenAI Lunar Lander

This repository contains a PSO-optimized policy for the [LunarLander-v3](https://gymnasium.farama.org/environments/box2d/lunar_lander/) environment, using Particle Swarm Optimization to fine-tune a pre-trained DDQN agent.

## 🧠 Overview

- Base policy trained using DDQN.
- PSO fine-tuning enhances reward performance by optimizing policy weights.
- Lightweight 2-layer neural network agent.
- No deep RL libraries — pure NumPy + Gymnasium.

## 📁 Project Structure

```
src/
├── train_agent_PSO_locally.py    # Main PSO fine-tuning script
├── compare_pso_vs_ddqn.py              # Compares DDQN vs PSO-tuned policies
├── evaluate_policy.py            # Renders agent playing visually
├── policy_mine.py                # Neural net forward pass logic
data/
├── best_policy_ddqn.npy          # Initial DDQN weights
├── checkpoint.npy                # Best PSO-tuned weights
```

## ⚙️ Installation

```bash
pip install -r requirements.txt
```

### `requirements.txt` contents:
```
gymnasium
numpy
```

## 🚀 How to Run

### PSO Tuning
```bash
python src/train_agent_PSO_locally.py --policyfile data/best_policy_ddqn.npy --pso_iter 50
```

### Evaluation (without render)
```bash
python src/compare_pso_vs_ddqn.py
```

### Visual Evaluation
```bash
python src/evaluate_policy.py
```

## 🧠 Author

Made by [Dipanshu Raj]  
