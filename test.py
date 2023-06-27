from envs.nfl_game_env_shaped import NFLGame_shaped

from stable_baselines import PPO2
from stable_baselines.common.evaluation import evaluate_policy

# You can use the trained model to take actions in your environment as follows:
env = NFLGame_shaped()

model = PPO2.load("PPO_nfl_shaped")

mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)
print(f"Mean reward: {mean_reward} +/- {std_reward}")

obs = env.reset()

print('obs = ', obs)
for i in range(100):
    action, _ = model.predict(obs)
    print('action = ', action)
    obs, reward, done, info = env.step(action)
    print('reward = ', reward)
    print('obs = ', obs)
    if done:
        break
