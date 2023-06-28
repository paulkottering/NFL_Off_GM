# Import required modules
from envs.nfl_game_env import NFLGame
import optuna
from stable_baselines import PPO2
from stable_baselines.common.evaluation import evaluate_policy

# Create an instance of the NFLGame environment
env = NFLGame()

# Set the name of the model
model_name = "PPO_nfl_optuna_study_simple_reward"


# Define the objective function for the Optuna hyperparameter search
def objective(trial):
    # Suggest values for the learning_rate and gamma hyperparameters
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-6, 1.0)
    gamma = trial.suggest_uniform('gamma', 0.8, 1.0)

    # Create a PPO2 model with the suggested hyperparameters
    model = PPO2('MlpPolicy', env, verbose=0, learning_rate=learning_rate, gamma=gamma)

    # Train the model for 10,000 timesteps
    model.learn(total_timesteps=10000)

    # Evaluate the model and get the mean reward
    mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=100)

    # Return the mean reward to be maximized by Optuna
    return mean_reward


# Create an Optuna study object to optimize the hyperparameters
study = optuna.create_study(direction="maximize")

# Optimize the study with the defined objective function, and perform 50 trials
study.optimize(objective, n_trials=50)

# Convert the study history to a dataframe and save it as a CSV file
df = study.trials_dataframe()
df.to_csv('hyperparam_search/' + model_name + '.csv')

# Get the best hyperparameters and the corresponding reward
best_params = study.best_params
best_reward = study.best_value

# Print the best hyperparameters and their reward
print(f"Best params: {best_params}, with reward: {best_reward}")

# Create a new PPO2 model with the best hyperparameters
model = PPO2('MlpPolicy', env, verbose=1, **best_params)

# Train the model for 100,000 timesteps
model.learn(total_timesteps=100000)

# Save the trained model
model.save(model_name)

# Delete the trained model to demonstrate loading
del model

# Load the trained model from file
model = PPO2.load(model_name)

# Evaluate the loaded model
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)

# Print the mean reward and standard deviation
print(f"Mean reward: {mean_reward} +/- {std_reward}")
