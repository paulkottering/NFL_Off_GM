from HER.nfl_game_her_env import NFLGame_HER
from stable_baselines import DQN
from stable_baselines.her import HERGoalEnvWrapper

# You can use the trained model to take actions in your environment as follows:
env = NFLGame_HER()
env = HERGoalEnvWrapper(env)  # Wrap the environment during testing as well

model = DQN.load("DQN_HER_nfl")
obs = env.reset()
print('obs = ', obs)
for i in range(100):
    action, _ = model.predict(obs)
    print('action = ', action)
    obs, reward, done, info = env.step(action)
    print('obs = ', obs)
    if done:
        break
