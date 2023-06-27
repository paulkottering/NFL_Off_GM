import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils import run, throw, field_goal

def create_and_plot_distribution():
    # Number of trials
    n = 50000

    # Initialize arrays to store results
    run_yards_0 = np.zeros(n)
    run_yards_1 = np.zeros(n)
    throw_yards_2 = np.zeros(n)
    throw_yards_3 = np.zeros(n)
    throw_yards_4 = np.zeros(n)
    kick_successes = np.zeros(100)

    # Simulate runs, throws and kicks
    for i in range(n):
        run_yards_0[i], _ = run(0)
        run_yards_1[i], _ = run(1)
        throw_yards_2[i], _ = throw(2)
        throw_yards_3[i], _ = throw(3)
        throw_yards_4[i], _ = throw(4)

    # Simulate field goals for each possible distance
    for i in range(100):
        kick_successes[i] = np.mean([field_goal(i) for _ in range(n)])

    # Filter out zero results
    run_yards_0 = run_yards_0[run_yards_0 != 0]
    run_yards_1 = run_yards_1[run_yards_1 != 0]
    throw_yards_2 = throw_yards_2[throw_yards_2 != 0]
    throw_yards_3 = throw_yards_3[throw_yards_3 != 0]
    throw_yards_4 = throw_yards_4[throw_yards_4 != 0]

    # Plot distributions
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 2)

    axs = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[1, 1]),
        fig.add_subplot(gs[2, 1]),
        fig.add_subplot(gs[2, 0])
    ]

    fig.suptitle('Football Game Simulation Results', fontsize=20)
    for ax in axs[:5]:  # set shared y-axis limit to first five plots
        ax.set_ylim(0, 0.5)

    sns.histplot(run_yards_0, ax=axs[0], kde=False, stat="probability",bins=range(-10, 100, 5))
    axs[0].set_title('Run (Short) Action', fontsize=16)
    axs[0].set_xlabel('Yards Gained', fontsize=14)
    axs[0].set_ylabel('Probability', fontsize=14)

    sns.histplot(run_yards_1, ax=axs[1], kde=False, stat="probability",bins=range(-10, 100, 5))
    axs[1].set_title('Run (Long) Action', fontsize=16)
    axs[1].set_xlabel('Yards Gained', fontsize=14)

    sns.histplot(throw_yards_2, ax=axs[2], kde=False, stat="probability",bins=range(-10, 100, 5))
    axs[2].set_title('Throw (Short) Action - Completion Probability = 50%', fontsize=16)
    axs[2].set_xlabel('Yards Gained', fontsize=14)
    axs[2].set_ylabel('Probability', fontsize=14)

    sns.histplot(throw_yards_3, ax=axs[3], kde=False, stat="probability",bins=range(-10, 100, 5))
    axs[3].set_title('Throw (Medium) Action - Completion Probability = 30%', fontsize=16)
    axs[3].set_xlabel('Yards Gained', fontsize=14)

    sns.histplot(throw_yards_4, ax=axs[4], kde=False, stat="probability",bins=range(-10, 100, 5))
    axs[4].set_title('Throw (Long) Action - Completion Probability = 15%', fontsize=16)
    axs[4].set_xlabel('Yards Gained', fontsize=14)

    axs[5].plot(range(100), kick_successes, label='Field Goal Success')
    axs[5].set_title('Field Goal Success Probability', fontsize=16)
    axs[5].set_xlabel('Kick Distance', fontsize=14)
    axs[5].set_ylabel('Success Probability', fontsize=14)

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.savefig('distributions')

# Call the function
create_and_plot_distribution()
