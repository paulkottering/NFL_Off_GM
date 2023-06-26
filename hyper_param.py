from nfl_game_env import NFLGame
import optuna
from stable_baselines import DQN, PPO2
from stable_baselines.common.evaluation import evaluate_policy

env = NFLGame()

def objective(trial):
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-6, 1.0)
    gamma = trial.suggest_uniform('gamma', 0.8, 1.0)
    model = PPO2('MlpPolicy', env, verbose=0, learning_rate=learning_rate, gamma=gamma)
    model.learn(total_timesteps=10000)

    mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=100)
    return mean_reward

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)

df = study.trials_dataframe()
df.to_csv('PPO_nfl_optuna_study_simple_reward.csv')

best_params = study.best_params
best_reward = study.best_value

print(f"Best params: {best_params}, with reward: {best_reward}")

# Training final model with the best parameters
model = PPO2('MlpPolicy', env, verbose=1, **best_params)
model.learn(total_timesteps=100000)

# Save the agent
model.save("PPO_nfl_optuna_simple_reward")

del model  # delete trained model to demonstrate loading

# Load the trained agent
model = PPO2.load("PPO_nfl_optuna_simple_reward")

# Evaluate the agent
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)

print(f"Mean reward: {mean_reward} +/- {std_reward}")
