from envs.nfl_game_env import NFLGame
from envs.nfl_game_env_shaped import NFLGame_shaped
from stable_baselines import PPO2
from stable_baselines.common.evaluation import evaluate_policy
import matplotlib.pyplot as plt
import numpy as np

env_train = NFLGame_shaped()
env_test = NFLGame()

# Training final model with the best parameters
model = PPO2('MlpPolicy', env_train, verbose=1, cliprange=0.4)

mean_rewards = []
std_rewards = []
eval_timesteps = 10000
n_eval_episodes = 1000
total_timesteps = 500000
model_name = "PPO_nfl_shaped_2"

for i in range(total_timesteps // eval_timesteps):
    model.learn(total_timesteps=eval_timesteps)
    mean_reward, std_reward = evaluate_policy(model, env_test, n_eval_episodes=n_eval_episodes)
    mean_rewards.append(mean_reward)
    std_rewards.append(std_reward)
    print('Mean = ',mean_reward)

# Save the agent
model.save(model_name)


# Plotting mean rewards with std as error bars
plt.errorbar(np.arange(len(mean_rewards)) * eval_timesteps, mean_rewards, yerr=std_rewards, fmt='o')
plt.title('Training progress')
plt.xlabel('Timesteps')
plt.ylabel('Mean Reward')
plt.savefig('train_plot_'+ model_name + '.png')
