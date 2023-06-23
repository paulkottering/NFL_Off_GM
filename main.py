from stable_baselines import PPO2
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.evaluation import evaluate_policy

from nfl_game_env import nfl_game

env = nfl_game()
env = DummyVecEnv([lambda: env])

# Instantiate the agent
model = PPO2('MlpPolicy', env, verbose=1)

# Train the agent
model.learn(total_timesteps=100000)

# Save the agent
model.save("ppo_nfl")

# Load the trained agent
model = PPO2.load("ppo_nfl")

# Evaluate the agent
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)

print(f"Mean reward: {mean_reward} +/- {std_reward}")

# You can use the trained model to take actions in your environment as follows:
obs = env.reset()
for i in range(1000):
    action, _ = model.predict(obs)
    obs, reward, done, info = env.step(action)
    if done:
        obs = env.reset()

