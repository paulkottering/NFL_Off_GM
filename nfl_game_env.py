import numpy as np
import gym
from gym import spaces
from utils import run, throw, punt, field_goal, defensive_possession


class NFLGame(gym.Env):
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
        self.observation_space = spaces.Box(low=-100, high=1000, shape=(5,), dtype=int)

    def reset(self):
        """
        This function resets the game to its initial state.
        """

        self.time = 1000
        self.down = 1
        self.yard_line = 25
        self.distance_to_go = 10
        self.score_difference = 0

        # Return the initial state of the game
        start = [self.yard_line, self.distance_to_go, self.down, self.time, self.score_difference]
        return start

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
                self.yard_line = np.clip(yards_gained+self.yard_line,0,100)

                if yards_gained > self.distance_to_go:
                    self.down = 1
                    self.distance_to_go = 10
                else:
                    self.down += 1
                    self.distance_to_go -= yards_gained

        # pass plays
        if action == 2 or action == 3 or action == 4:
            yards_gained, retain_ball = throw(action)
            if retain_ball:
                self.yard_line = np.clip(yards_gained+self.yard_line,0,100)

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
                    self.distance_to_go = 10
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
                self.yard_line = min(np.clip(self.yard_line-7,0,100),80)

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
            if self.score_difference > 0:
                return self.construct_obs(), 10, True, {}
            elif self.score_difference == 0:
                return self.construct_obs(), 3, True, {}
            else:
                return self.construct_obs(), -3, True, {}

        return self.construct_obs(), reward, False, {}

    def construct_obs(self):
        """
        This function constructs the observations of the current game state.
        """

        obs = [self.yard_line, self.distance_to_go, self.down, self.time, self.score_difference]
        return obs
