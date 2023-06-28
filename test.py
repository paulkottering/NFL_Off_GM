from envs.nfl_game_env_shaped import NFLGame_shaped
from envs.nfl_game_env import NFLGame

from stable_baselines import PPO2
from stable_baselines.common.evaluation import evaluate_policy

# Load the trained model from file
model = PPO2.load("PPO_nfl_shaped")

# Initialize the environment
env = NFLGame()
# Evaluate the trained model
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)
print(f"Mean reward: {mean_reward} +/- {std_reward}")

# Reset the environment and get the initial observation
obs = env.reset()

# Explanation of the observation
print(f"Yard line: {obs[0]}, Distance to go: {obs[1]}, Down: {obs[2]}, Time: {obs[3]}, Score difference: {obs[4]}")

# Explanation of the action
action_dict = {0:'Run Short', 1:'Run Long', 2:'Short pass', 3:'Medium pass', 4:'Long pass', 5:'Punt', 6:'Field goal'}
# Loop for a predetermined number of steps
for i in range(100):
    # Predict the action using the model
    action, _ = model.predict(obs)
    print('Action chosen: ', action_dict[action])
    # Take the action in the environment
    obs, reward, done, info = env.step(action)
    # Explanation of the observation after action
    print(f"Yard line: {obs[0]} \nDistance to go: {obs[1]} \nDown: {obs[2]} \nTime: {obs[3]} \nScore difference: {obs[4]}")
    # Print the reward
    print('Reward = ', reward)
    if done:
        break
