import numpy as np
import gym
from gym import spaces
from utils import run, throw, punt, field_goal, defensive_possession
from collections import OrderedDict


class NFLGame_HER(gym.Env):
    """
    This class defines the gym environment for the NFL game.
    """

    def __init__(self):
        """
        This function initializes the NFL game environment.
        """

        # Initial values for the time, down, yard line, distance to go, and score difference
        self.time = 1000
        self.down = 1
        self.yard_line = 25
        self.distance_to_go = 10
        self.score_difference = 0

        # Define the action space (7 possible actions) and observation space (5 observations)
        self.action_space = spaces.Discrete(7)
        # self.observation_space = spaces.Box(low=-100, high=1000, shape=(5,), dtype=int)
        self.observation_space = spaces.Dict({
            'observation': spaces.Box(low=-100, high=1000, shape=(5,), dtype=int),
            'achieved_goal': spaces.Box(low=-100, high=100, shape=(1,), dtype=int),
            'desired_goal': spaces.Box(low=-100, high=100, shape=(1,), dtype=int),
        })

        # The goal will be a desired score difference
        self.goal_space = spaces.Box(low=-100, high=100, shape=(1,), dtype=np.int)
        self.goal = np.array([0])

    def set_goal(self, goal):
        self.goal = goal

    def get_goal(self):
        return self.goal

    def compute_reward(self, achieved_goal, desired_goal, info):

        return float(achieved_goal >= desired_goal)

    def reset(self):
        """
        This function resets the game to an initial state.
        """
        prob = np.random.rand()
        if prob > 0.9:
            self.time = 1000
            self.down = 1
            self.yard_line = 25
            self.distance_to_go = 10
            self.score_difference = 0
            self.goal = np.array([1])
        else:
            self.time = np.random.randint(0, 1000)
            self.down = np.random.randint(1, 4)
            self.yard_line = np.random.randint(0, 100)
            self.distance_to_go = min(np.random.randint(0, 20), 100 - self.yard_line)
            self.score_difference = np.random.randint(-7, 7)
            self.goal = np.array([self.score_difference+1])

        return self._get_obs()

    def step(self, action):
        """
        This function defines the transitions of the game for each step based on the chosen action.
        """

        reward = 0

        # run plays
        if action == 0 or action == 1:
            yards_gained, retain_ball = run(action)
            # The below logic is applied if the ball is retained after the action
            if retain_ball:
                self.time -= 35
                self.yard_line = np.clip(yards_gained + self.yard_line, 0, 100)

                if yards_gained > self.distance_to_go:
                    self.down = 1
                    self.distance_to_go = min(10, 100 - self.yard_line)
                else:
                    self.down += 1
                    self.distance_to_go -= yards_gained

        # pass plays
        if action == 2 or action == 3 or action == 4:
            yards_gained, retain_ball = throw(action)
            if retain_ball:
                self.yard_line = np.clip(yards_gained + self.yard_line, 0, 100)

                if yards_gained == 0:
                    self.time -= 5
                else:
                    chance = np.random.rand()
                    if chance < 0.5:
                        self.time -= 35
                    else:
                        self.time -= 7

                if yards_gained > self.distance_to_go:
                    self.down = 1
                    self.distance_to_go = min(10, 100 - self.yard_line)
                else:
                    self.down += 1
                    self.distance_to_go -= yards_gained

        # punt action
        if action == 5:
            self.time -= 7
            self.yard_line = punt(self.yard_line)
            retain_ball = False

        # field goal action
        if action == 6:
            distance = 100 - self.yard_line + 17
            kick_success = field_goal(distance)
            retain_ball = False
            self.time -= 2
            if kick_success:
                self.score_difference += 3
                self.yard_line = 75
            else:
                self.yard_line = min(np.clip(self.yard_line - 7, 0, 100), 80)

        if retain_ball and self.yard_line >= 100:
            self.score_difference += 7
            retain_ball = False
            self.yard_line = 75

        if retain_ball and self.down == 5:
            retain_ball = False

        if not retain_ball:
            start_yard_line, opp_change_in_score, time_taken = defensive_possession(self.yard_line, self.time)
            self.yard_line = start_yard_line
            self.distance_to_go = 10
            self.down = 1
            self.score_difference -= opp_change_in_score
            self.time -= time_taken

        # End the game if time runs out
        if self.time < 0:
            done = True
        else:
            done = False

        achieved_goal = np.array([self.score_difference])
        reward = self.compute_reward(achieved_goal, self.goal, {})
        return self._get_obs(), reward, done, {}

    def construct_obs(self):
        """
        This function constructs the observations of the current game state.
        """

        obs = [self.yard_line, self.distance_to_go, self.down, self.time, self.score_difference]

        return obs

    def _get_obs(self):
        """
        Helper to create the observation.

        :return: (OrderedDict<int or ndarray>)
        """
        return OrderedDict([
            ('observation', self.construct_obs().copy()),
            ('achieved_goal', np.array([self.score_difference])),
            ('desired_goal', self.goal.copy())
        ])
