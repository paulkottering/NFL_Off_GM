# NFL Game Offensive Play Caller

This project trains a reinforcement learning model to perform plays in a simulated NFL game environment. The goal of the agent is to win the game. 

## Description

The agent is trained using the Proximal Policy Optimization (PPO) algorithm. The project includes code for:

- Defining the game environment and its rules.
- Training the agent in the environment using PPO.
- Evaluating the trained agent's performance.
- Hyperparameter optimization using Optuna.

The game environment comes in two forms (they can be found in the 'envs' folder):

- NFLGame: This environment has a simple reward structure, with rewards equal to 1 for winning and -1 for losing.
- NFLGame_shaped: This environment has a shaped reward structure, which includes additional rewards and penalties for other game situations.

## Installation

Clone this repository using git:

```
git clone https://github.com/username/repository.git
```

Then install the required dependencies:

```
pip install -r requirements.txt
```

## Usage

There are several main scripts in the project:

- `main.py`: This script trains the agent using PPO and plots the rewards over time.
- `optuna_hyperparam_tuning.py`: This script uses Optuna to optimize the hyperparameters of the PPO model.
- `test.py`: This script loads a trained agent and evaluates its performance in the game environment.

Also I have experiment with using Hindsight Experience Replay. This can be found in the 'HER' folder. 

To run the scripts, navigate to the project directory in your terminal and use the command:

```
python <script_name>.py
```

Replace `<script_name>` with the name of the script you want to run.

## Contributing

Contributions are welcome! Please open an issue to discuss your proposed changes, or open a pull request if you have code ready to merge.

## License

This project is licensed under the MIT License.

## Credits

This project was created by Paul Koettering.
