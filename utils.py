import numpy as np
from scipy.stats import beta, norm

def run(action):
    """
    This function defines the run action in the game.
    It returns the yards gained and whether the ball is retained.
    """

    # Short run. Under 5 yards. Completion is around 75%. Lose ball 1% time.
    if action == 0:

        # Decide if the ball is retained after the action
        retain_ball = np.random.choice([True, False], p=[0.99, 0.01])

        # Calculate yards gained
        if retain_ball:
            intervals = np.array([-10, 0, 5, 10, 15, 20, 30, 50, 100])
            probabilities = np.array([0.2, 0.5, 0.2, 0.05, 0.02, 0.01, 0.01, 0.01])
            yards_gained = sample_custom_distribution(intervals, probabilities)
            return yards_gained, retain_ball

        # If ball is not retained, return 0 yards gained
        else:
            return 0, retain_ball

    # Long run. Over 5 yards. Completion is around 50%. Lose ball 1% time.
    elif action == 1:

        # Decide if the ball is retained after the action
        retain_ball = np.random.choice([True, False], p=[0.99, 0.01])

        # Calculate yards gained
        if retain_ball:
            intervals = np.array([-10, 0, 5, 10, 15, 20, 30, 50, 100])
            probabilities = np.array([0.35, 0.2, 0.3, 0.1, 0.02, 0.01, 0.01, 0.01])
            yards_gained = sample_custom_distribution(intervals, probabilities)
            return yards_gained, retain_ball

        # If ball is not retained, return 0 yards gained
        else:
            return 0, retain_ball


def throw(action):
    """
    This function defines the throw action in the game.
    It returns the yards gained and whether the ball is retained.
    """

    # Short throw. Under 15 yards. Completion is around 50%. Lose ball 2% time.
    if action == 2:

        # Decide if the ball is retained after the action
        retain_ball = np.random.choice([True, False], p=[0.98, 0.02])

        # Calculate yards gained
        if retain_ball:
            intervals = np.array([-10, 0, 0.01, 5, 10, 15, 20, 30, 50, 100])
            probabilities = np.array([0.04, 0.5, 0.25, 0.15, 0.02, 0.01, 0.01, 0.01, 0.01])
            yards_gained = sample_custom_distribution(intervals, probabilities)
            return yards_gained, retain_ball

        # If ball is not retained, return 0 yards gained
        else:
            return 0, retain_ball

    # Medium throw. 15 to 25 yards. Completion is around 30%. Lose ball 4% time.
    elif action == 3:

        # Decide if the ball is retained after the action
        retain_ball = np.random.choice([True, False], p=[0.96, 0.04])

        # Calculate yards gained
        if retain_ball:
            intervals = np.array([-10, 0, 0.01, 5, 10, 15, 20, 30, 50, 100])
            probabilities = np.array([0.07, 0.7, 0.04, 0.08, 0.07, 0.02, 0.01, 0.005, 0.005])
            yards_gained = sample_custom_distribution(intervals, probabilities)
            return yards_gained, retain_ball

        # If ball is not retained, return 0 yards gained
        else:
            return 0, retain_ball

    # Long throw. Over 25 yards. Completion is around 10%. Lose ball 8% time.
    elif action == 4:

        # Decide if the ball is retained after the action
        retain_ball = np.random.choice([True, False], p=[0.92, 0.08])

        # Calculate yards gained
        if retain_ball:
            intervals = np.array([-10, 0, 0.01, 5, 10, 15, 20, 30, 50, 100])
            probabilities = np.array([0.07, 0.85, 0.005, 0.005, 0.005, 0.02, 0.015, 0.015, 0.015])
            yards_gained = sample_custom_distribution(intervals, probabilities)
            return yards_gained, retain_ball

        # If ball is not retained, return 0 yards gained
        else:
            return 0, retain_ball


def punt(yard_line):
    """
    This function calculates the new yard line after a punt.
    """
    mu = 65  # Average punt distance in the NFL
    sigma = 5  # Standard deviation, representing variability in punt distance
    # Uncomment below line for more realistic simulation
    # new_yard_line = np.clip(yard_line + int(np.random.normal(mu, sigma)), 0, 95)
    return 95  # Temporary return statement


def field_goal(distance):
    """
    This function calculates if a field goal attempt is successful.
    """
    if distance < 35:
        kick_success = np.random.choice([True, False], p=[0.9, 0.1])
        return kick_success
    elif distance < 65:
        prob = -(0.5/30)*distance + (0.8+(0.5/30)*35)
        kick_success = np.random.choice([True, False], p=[prob, 1-prob])
        return kick_success
    else:
        return False


def defensive_possession(yard_line, time):
    """
    This function simulates the defensive possession in the game.
    It returns the starting yard line, change in score, and time taken.
    """
    # Scaling the probabilities with respect to the yard_line
    x = yard_line
    prob_td = 0.9 - x*0.016 + 0.00009*x**2
    prob_fg = 0.1 + x*0.007 - 0.00008*x**2
    prob_none = 1 - (prob_td + prob_fg)

    prob = np.random.choice([1, 2, 3], p=[prob_td, prob_fg, prob_none])
    time_taken = np.clip(int(norm.rvs(loc=60, scale=20)), 0, 100)

    if prob == 1:
        opp_change_in_score = 7
        start_yard_line = 25
        return start_yard_line, opp_change_in_score, time_taken

    if prob == 2:
        opp_change_in_score = 3
        start_yard_line = 25
        return start_yard_line, opp_change_in_score, time_taken

    else:
        opp_change_in_score = 0
        a = 4
        b = 8
        new_min, new_max = 0, yard_line
        sample = beta.rvs(a, b)
        start_yard_line = sample * (new_max - new_min) + new_min

        return int(start_yard_line), opp_change_in_score, time_taken


def sample_custom_distribution(intervals, probabilities):
    """
    This function generates a sample from a custom distribution.
    """
    # Normalize probabilities if they don't sum to 1
    probabilities = probabilities / np.sum(probabilities)

    # Choose an interval
    chosen_interval = np.random.choice(len(probabilities), p=probabilities)

    # Generate a uniformly distributed random number within the chosen interval
    sample = np.random.uniform(intervals[chosen_interval], intervals[chosen_interval + 1])

    return int(sample)
