import gymnasium as gym
import numpy as np
import random
import argparse
from collections import deque
import math

# Network architecture dimensions (for a 2-layer network)
INPUT_DIM = 8
HIDDEN_DIM = 64
OUTPUT_DIM = 4
PARAM_SIZE = (INPUT_DIM * HIDDEN_DIM) + HIDDEN_DIM + (HIDDEN_DIM * OUTPUT_DIM) + OUTPUT_DIM  # 836

def forward_pass(params, observation):
    idx = 0
    W1 = params[idx:idx+INPUT_DIM*HIDDEN_DIM].reshape(INPUT_DIM, HIDDEN_DIM)
    idx += INPUT_DIM*HIDDEN_DIM
    b1 = params[idx:idx+HIDDEN_DIM]
    idx += HIDDEN_DIM
    W2 = params[idx:idx+HIDDEN_DIM*OUTPUT_DIM].reshape(HIDDEN_DIM, OUTPUT_DIM)
    idx += HIDDEN_DIM*OUTPUT_DIM
    b2 = params[idx:idx+OUTPUT_DIM]
    hidden = np.maximum(0, np.dot(observation, W1) + b1)
    logits = np.dot(hidden, W2) + b2
    return logits

def evaluate_policy_with_domain(params, env, episodes=50):
    total_reward = 0.0
    for _ in range(episodes):
        state, _ = env.reset()
        state = np.array(state)
        done = False
        episode_reward = 0.0
        while not done:
            logits = forward_pass(params, state)
            action = np.argmax(logits)
            next_state, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward  
            state = np.array(next_state)
            done = terminated or truncated
        total_reward += episode_reward
    return total_reward / episodes

def pso_finetune(initial_params, env, swarm_size=20, iterations=50, 
                 w=0.5, c1=1.0, c2=1.0, noise_scale=0.05, 
                 stagnation_limit=10, restart_fraction=0.2,
                 checkpoint_interval=2, checkpoint_prefix="checkpoint"):
    dim = initial_params.shape[0]
    # Initialize swarm: particles are initialized near the initial parameters.
    swarm = [initial_params + np.random.randn(dim) * noise_scale for _ in range(swarm_size)]
    velocities = [np.zeros(dim) for _ in range(swarm_size)]
    
    # Evaluate initial swarm fitness using the domain-informed evaluation function.
    p_best = list(swarm)
    p_best_fit = [evaluate_policy_with_domain(p, env) for p in swarm]
    
    best_idx = np.argmax(p_best_fit)
    g_best = p_best[best_idx].copy()
    g_best_fit = p_best_fit[best_idx]
    
    stagnation_counter = 0
    for it in range(iterations):
        improvement = False
        for i in range(swarm_size):
            r1 = np.random.rand(dim)
            r2 = np.random.rand(dim)
            velocities[i] = (w * velocities[i] +
                             c1 * r1 * (p_best[i] - swarm[i]) +
                             c2 * r2 * (g_best - swarm[i]))
            swarm[i] = swarm[i] + velocities[i]
            fitness = evaluate_policy_with_domain(swarm[i], env)
            if fitness > p_best_fit[i]:
                p_best[i] = swarm[i].copy()
                p_best_fit[i] = fitness
                if fitness > g_best_fit:
                    g_best = swarm[i].copy()
                    g_best_fit = fitness
                    improvement = True
        if not improvement:
            stagnation_counter += 1
        else:
            stagnation_counter = 0
        
        # Random restart if no improvement for stagnation_limit iterations.
        if stagnation_counter >= stagnation_limit:
            print(f"Stagnation detected at iteration {it+1}. Restarting {int(swarm_size * restart_fraction)} particles.")
            num_restarts = int(swarm_size * restart_fraction)
            for i in range(num_restarts):
                if i % 2 == 0:
                    swarm[i] = g_best + np.random.randn(dim) * 0.1
                else:
                    swarm[i] = np.random.randn(dim) * 0.1
                velocities[i] = np.zeros(dim)
                p_best[i] = swarm[i].copy()
                p_best_fit[i] = evaluate_policy_with_domain(swarm[i], env)
            stagnation_counter = 0
        
        # Print the global best and personal best fitnesses each iteration.
        print(f"Iteration {it+1}: Global Best Fitness = {g_best_fit:.2f}")
        
        # Save checkpoint every checkpoint_interval iterations (overwriting the same file)
        if (it + 1) % checkpoint_interval == 0:
            np.save("../data/best_policy_pso.npy", g_best)
            print(f"Checkpoint saved to data/best_policy_pso.npy with Global Best Fitness = {g_best_fit:.2f}")
    
    return g_best

def main():
    parser = argparse.ArgumentParser(description="PSO fine-tuning for DDQN parameters using local search")
    parser.add_argument("--policyfile", type=str, default="../data/best_policy_ddqn.npy", help="File with initial good parameters")
    parser.add_argument("--pso_iter", type=int, default=50, help="Number of PSO iterations")
    parser.add_argument("--swarm_size", type=int, default=20, help="PSO swarm size")
    parser.add_argument("--local_search", action="store_true", help="Use very low exploration settings for local search")
    parser.add_argument("--checkpoint_interval", type=int, default=2, help="Interval (in iterations) to save checkpoint")
    parser.add_argument("--checkpoint_prefix", type=str, default="checkpoint", help="Filename for checkpoint (will be overwritten)")
    args = parser.parse_args()
    
    # Set hyperparameters for local search if requested.
    if args.local_search:
        noise_scale = 0.008 # Very small perturbations
        w = 0.5
        c1 = 0.7
        c2 = 0.7
        stagnation_limit = 10     # Restart if no improvement for 10 iterations
        restart_fraction = 0.3
        print("Local search activated: using minimal exploration settings.")
    else:
        noise_scale = 0.05
        w = 0.7
        c1 = 2.0
        c2 = 2.0
        stagnation_limit = 10
        restart_fraction = 0.2
    
    initial_params = np.load(args.policyfile)
    env = gym.make("LunarLander-v3", render_mode="rgb_array")
    
    best_params = pso_finetune(initial_params, env, swarm_size=args.swarm_size, 
                               iterations=args.pso_iter, noise_scale=noise_scale,
                               w=w, c1=c1, c2=c2, stagnation_limit=stagnation_limit,
                               restart_fraction=restart_fraction,
                               checkpoint_interval=args.checkpoint_interval,
                               checkpoint_prefix=args.checkpoint_prefix)

if __name__ == "__main__":
    main()
