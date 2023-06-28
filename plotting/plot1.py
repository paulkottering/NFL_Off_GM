import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from stable_baselines import PPO2

# Define the action labels and the variable names
labels = ['Short Run','Long Run','Short Throw','Medium Throw','Long Throw','Punt','Field Goal Attempt']
variables = ['yard_line', 'distance_to_go', 'down', 'time', 'score_difference']

# Set the index of the variable to change
variable_index = 4
# Extract the corresponding variable name
variable_param = variables[variable_index]
# Generate an array of values to substitute for the variable
values = np.linspace(-10, 20, 40)
# Define the values of the fixed parameters
fixed_params = [70, 5, 4, 200]

# Load the pre-trained model
model = PPO2.load("../models/PPO_nfl_shaped_2")

# Initialize an empty list to store the action probabilities
action_probabilities = []

# Loop over each value to substitute for the variable
for var in values:
    # Construct the observation by replacing the variable with the new value
    obs = np.array(fixed_params[:variable_index] + [var] + fixed_params[variable_index:])
    # Predict the action probabilities for the observation
    probs = model.action_probability(obs)
    # Append the action probabilities to the list
    action_probabilities.append(probs)

# Convert the list of action probabilities to a numpy array
action_probabilities = np.array(action_probabilities)

# Set the style of the plot to a seaborn theme
sns.set()

# Create a line plot for each action
for i in range(action_probabilities.shape[1]):
    plt.plot(values, action_probabilities[:, i], label=labels[i])

# Set the labels of the plot
plt.xlabel(variable_param)
plt.ylabel('Action Probability')
plt.title('Action Probabilities vs ' + variable_param)

# Display the legend
plt.legend(loc='upper left')

# Display the fixed parameters below the plot
plt.figtext(0.5, -0.1, f'Fixed parameters: {np.delete(variables, variable_index)} = {fixed_params}', ha='center')

# Save the plot as an image file
plt.savefig('Plot.png', bbox_inches='tight')
