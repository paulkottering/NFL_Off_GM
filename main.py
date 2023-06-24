from stable_baselines import PPO2, DQN
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.evaluation import evaluate_policy

from nfl_game_env import nfl_game

env2 = nfl_game()
env2 = DummyVecEnv([lambda: env2])

# Instantiate the agent
model = DQN('MlpPolicy', env2, verbose=1)

# Train the agent
model.learn(total_timesteps=50000)

# Save the agent
model.save("dqn_nfl")

del model

# Load the trained agent
model = DQN.load("dqn_nfl")

# Evaluate the agent
mean_reward, std_reward = evaluate_policy(model, env2, n_eval_episodes=10)

print(f"Mean reward: {mean_reward} +/- {std_reward}")

# You can use the trained model to take actions in your environment as follows:
obs = env2.reset()
print('obs = ', obs)
for i in range(100):
    action, _ = model.predict(obs)
    print('action = ', action)
    obs, reward, done, info = env2.step(action)
    print('obs = ', obs)
    if done:
        break



