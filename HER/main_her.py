from nfl_game_her_env import NFLGame_HER
from stable_baselines import DQN
from stable_baselines.her import HERGoalEnvWrapper
from stable_baselines.common.evaluation import evaluate_policy

# Create environment
env = NFLGame_HER()

# Wrap the environment
env = HERGoalEnvWrapper(env)

model = DQN('MlpPolicy', env, verbose=1,
            exploration_fraction=0.5,  # 50% of the total timesteps the model will explore
            exploration_initial_eps=1.0,  # initial exploration rate
            exploration_final_eps=0.1  # minimum exploration probability
            )

model.learn(total_timesteps=5000000)  # increase the total timesteps

# Save the agent
model.save("DQN_HER_nfl")

del model  # delete trained model to demonstrate loading

# Load the trained agent
model = DQN.load("DQN_HER_nfl")

# Evaluate the agent
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)

print(f"Mean reward: {mean_reward} +/- {std_reward}")
