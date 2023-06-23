import numpy as np
import gym
from gym import spaces
from utils import run, throw,punt, field_goal, defensive_possession

class nfl_game(gym.Env):

    def __init__(self):
        self.time = 1800
        self.down = 1
        self.yard_line = 25
        self.half = 1
        self.distance_to_go = 10
        self.score_home = 0
        self.score_away = 0

        self.action_space = spaces.Discrete(7)
        self.observation_space = spaces.Box(low=0, high=1800, shape=(7,), dtype=int)
        # self.observation_space = spaces.Dict({"yard_line": spaces.Discrete(100),
        #                                       "distance_to_go": spaces.Discrete(100),
        #                                       "down": spaces.Discrete(4),
        #                                       "time_left_in_half": spaces.Discrete(1800),
        #                                       "half": spaces.Discrete(2),
        #                                       "score": spaces.Box(low=0, high=np.inf, shape=(2,), dtype=int),
        #                                       })

    def reset(self):
        self.time = 1800
        self.down = 1
        self.yard_line = 25
        self.half = 1
        self.distance_to_go = 10
        self.score_home = 0
        self.score_away = 0
        start = [self.yard_line,self.distance_to_go,self.down,self.time,self.half,self.score_home,self.score_away]

        # start = {"yard_line": self.yard_line,
        #          "distance_to_go": self.distance_to_go,
        #          "down": self.down,
        #          "time_left_in_half": self.time,
        #          "half": self.half,
        #          "score": self.score,
        #          }
        return start

    def step(self, action):

        # run plays
        if action == 0 or action == 1:
            yards_gained, retain_ball = run(action)
            if retain_ball:
                self.time -= 35
                self.yard_line = np.clip(yards_gained+self.yard_line,0,100)

                if yards_gained > self.distance_to_go:
                    self.down = 1
                    self.distance_to_go = 10
                else:
                    self.down += 1
                    self.distance_to_go -= yards_gained

        # throw plays
        if action == 2 or action == 3 or action == 4:
            yards_gained, retain_ball = throw(action)
            if retain_ball:
                self.yard_line = np.clip(yards_gained+self.yard_line,0,100)

                # if incomplete, clock is stopped, else assume with 50% chance player runs OOB
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

        if action == 5:
            self.time -= 7
            yards_gained = punt(self.yard_line)
            self.yard_line = np.clip(yards_gained+self.yard_line,0,100)
            retain_ball = False

        if action == 6:
            distance = 100 - self.yard_line + 17
            kick_success = field_goal(distance)
            retain_ball = False
            self.time -= 2
            if kick_success:
                self.score_home += 3
                self.yard_line = 75
            else:
                self.yard_line = min(np.clip(self.yard_line-7,0,100),80)

        if retain_ball and self.yard_line > 100:
            self.score_home += 7
            retain_ball = False

        #check if loss of down due to 4 downs
        if retain_ball and self.down == 5:
            retain_ball = False

        # check if possession is
        if not retain_ball:
            start_yard_line, opp_change_in_score, time_taken = defensive_possession(self.yard_line, self.time)
            self.yard_line = start_yard_line
            self.distance_to_go = 10
            self.down = 1
            self.score_away += opp_change_in_score
            self.time -= time_taken


        # update half or end game
        if self.time < 0 :
            if self.half == 1:
                self.time = 1800
                self.half = 2
            else:
                if self.score_home > self.score_away:
                    return self.construct_obs(), 1, True, {}
                elif self.score_home == self.score_away:
                    return self.construct_obs(), -0.5, True, {}
                else:
                    return self.construct_obs(), -1, True, {}
        return self.construct_obs(), 0, False, {}

    def construct_obs(self):
        # obs = {"yard_line": self.yard_line,
        #        "distance_to_go": self.distance_to_go,
        #        "down": self.down,
        #        "time_left_in_half": self.time,
        #        "half": self.half,
        #        "score": self.score
        #        }
        obs = [self.yard_line,self.distance_to_go,self.down,self.time,self.half,self.score_home,self.score_away]
        return obs