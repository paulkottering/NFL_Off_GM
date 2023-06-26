from nfl_game_env import NFLGame
import optuna
from stable_baselines import PPO2
from stable_baselines.common.evaluation import evaluate_policy

env = NFLGame()

# Training final model with the best parameters
model = PPO2('MlpPolicy', env, verbose=1)

model.learn(total_timesteps=20000000)

# Save the agent
model.save("PPO_nfl")

del model  # delete trained model to demonstrate loading

# Load the trained agent
model = PPO2.load("PPO_nfl")

# Evaluate the agent
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)

print(f"Mean reward: {mean_reward} +/- {std_reward}")
