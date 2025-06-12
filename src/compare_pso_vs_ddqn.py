import numpy as np
import gymnasium as gym
from policy_mine import policy_action

def evaluate_policy(policy_params, episodes=100):
    env = gym.make("LunarLander-v3", render_mode=None)
    total_reward = 0.0
    
    for episode in range(episodes):
        state, _ = env.reset()
        done = False
        episode_reward = 0.0

        while not done:
            action = policy_action(policy_params, state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            state = next_state
            done = terminated or truncated

        print(f"Episode {episode+1}: Reward = {episode_reward}")
        total_reward += episode_reward

    avg_reward = total_reward / episodes
    print(f"\n Average reward over {episodes} episodes: {avg_reward:.2f}")
    env.close()

# Load and evaluate the given DDQN policy
if __name__ == "__main__":
    try:
        tuned_params = np.load("../data/best_policy_pso.npy")
        print("\nEvaluating the PSO-tuned policy checkpoint:")
        evaluate_policy(tuned_params, episodes=100)
    except FileNotFoundError:
        print("\nNo PSO checkpoint found.")
    try:
        ddqn_params = np.load("../data/best_policy_ddqn.npy")
        print("Evaluating the original DDQN policy (before PSO tuning):")
        evaluate_policy(ddqn_params, episodes=100)
    except FileNotFoundError:
        print("\nNo ddqn found.")

    
