from envs.nfl_game_env import NFLGame
from envs.nfl_game_env_shaped import NFLGame_shaped
from stable_baselines import PPO2
from stable_baselines.common.evaluation import evaluate_policy
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines.common.env_checker import check_env


# Initialize the training environment using the shaped reward structure
env_train = NFLGame_shaped()
# Initialize the testing environments
env_test_simple = NFLGame()
env_test_shaped = NFLGame_shaped()

# check_env(env_train)
# check_env(env_test_simple)
# check_env(env_test_shaped)

# Initialize the PPO2 model
# The 'MlpPolicy' is a type of policy optimized for PPO,
# which uses a multilayer perceptron (neural network) for function approximation.
model = PPO2('MlpPolicy', env_train, verbose=1, cliprange=0.2, ent_coef=0.05, learning_rate=0.00015)

# Initialize empty lists to store mean rewards and standard deviations
mean_rewards_simple = []
std_rewards_simple = []
mean_rewards_shaped = []
std_rewards_shaped = []

# Set the number of timesteps for each evaluation period
eval_timesteps = 200000
# Set the number of episodes to evaluate for each period
n_eval_episodes = 50
# Set the total number of training timesteps
total_timesteps = 20000000
# Set the name of the model for saving
model_name = "PPO_nfl_shaped_3"

# Train the model for the specified total number of timesteps,
# evaluating and recording the mean and std reward every eval_timesteps
for i in range(total_timesteps // eval_timesteps):
    model.learn(total_timesteps=eval_timesteps)

    mean_reward, std_reward = evaluate_policy(model, env_test_simple, n_eval_episodes=n_eval_episodes)
    mean_rewards_simple.append(mean_reward)
    std_rewards_simple.append(std_reward)

    mean_reward, std_reward = evaluate_policy(model, env_test_shaped, n_eval_episodes=n_eval_episodes)
    mean_rewards_shaped.append(mean_reward)
    std_rewards_shaped.append(std_reward)

    print('Eval Number = ', i)

# Save the trained model
model.save('models/'+model_name)

# Create a figure with two subplots
fig, axs = plt.subplots(2)

# Plotting mean rewards with std as error bars for the simple reward environment
axs[0].plot(np.arange(len(mean_rewards_simple)) * eval_timesteps, mean_rewards_simple)
axs[0].set_title('Training progress on Simple Reward Environment')
axs[0].set_xlabel('Timesteps')
axs[0].set_ylabel('Mean Reward')

# Plotting mean rewards with std as error bars for the shaped reward environment
axs[1].plot(np.arange(len(mean_rewards_shaped)) * eval_timesteps, mean_rewards_shaped)
axs[1].set_title('Training progress on Shaped Reward Environment')
axs[1].set_xlabel('Timesteps')
axs[1].set_ylabel('Mean Reward')

# Add space between subplots to avoid overlapping labels
plt.subplots_adjust(hspace=0.6)

# Save the figure
plt.savefig('plotting/train_plot_'+ model_name + '.png')

