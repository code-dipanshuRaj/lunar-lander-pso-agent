import numpy as np
import gymnasium as gym
from policy_mine import policy_action

def evaluate_policy(policy_file, episodes=100):
    params = np.load(policy_file)
    env = gym.make("LunarLander-v3", render_mode="human")  
    
    total_reward = 0.0
    for episode in range(episodes):
        state, _ = env.reset()
        done = False
        episode_reward = 0.0
        while not done:
            action = policy_action(params, state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            state = next_state
            done = terminated or truncated
        print(f"Episode {episode+1}: Reward = {episode_reward}")
        total_reward += episode_reward
    
    avg_reward = total_reward / episodes
    print(f"\n Average reward over {episodes} episodes: {avg_reward:.2f}")
    env.close()

if __name__ == "__main__":
    evaluate_policy("../data/best_policy_pso.npy", episodes=100)
