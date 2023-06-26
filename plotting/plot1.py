import matplotlib.pyplot as plt
import numpy as np
from stable_baselines import PPO2, DQN, ACER

labels = ['Short Run','Long Run','Short Throw','Medium Throw','Long Throw','Punt','Field Goal Attempt']
variables = ['yard_line', 'distance_to_go', 'down', 'time', 'score_difference']

variable_index = 0
variable_param = variables[variable_index]
values = np.linspace(0, 100, 100)
fixed_params = [90, 4, 100, 0]

model = PPO2.load("../PPO_nfl_optuna_def")

action_probabilities = []

for var in values:
    obs = np.array(fixed_params[:variable_index] + [var] + fixed_params[variable_index:])
    probs = model.action_probability(obs)
    action_probabilities.append(probs)

action_probabilities = np.array(action_probabilities)

# Create a line plot for each action
for i in range(action_probabilities.shape[1]):
    plt.plot(values, action_probabilities[:, i], label=f'Action {i}')

plt.xlabel(variable_param)
plt.ylabel('Action Probability')
plt.title('Action Probabilities vs ' + variable_param)
plt.legend()
plt.savefig('Plot.png')
