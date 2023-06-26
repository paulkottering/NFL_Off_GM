from nfl_game_env import NFLGame
from stable_baselines import DQN,PPO2

# You can use the trained model to take actions in your environment as follows:
env = NFLGame()
model = PPO2.load("PPO_nfl_optuna_def")
obs = env.reset()
print('obs = ', obs)
for i in range(100):
    action, _ = model.predict(obs)
    print('action = ', action)
    obs, reward, done, info = env.step(action)
    print('obs = ', obs)
    if done:
        break