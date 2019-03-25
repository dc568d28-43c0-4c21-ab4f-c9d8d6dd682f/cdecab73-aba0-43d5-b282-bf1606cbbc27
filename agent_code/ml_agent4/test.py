import json
import matplotlib.pyplot as plt

GENERATION_SIZE = 100

def draw_rewards():
    try:
        history = open('./weights/ml_agent4_history.json', 'r').readlines()
    except Exception:
        print('Couldn\'t load history file!')
        return
    epochs = list()
    rewards = list()
    generations = list()
    winrates = list()
    history_size = len(history)
    for i in range(history_size):
        epochs.append(i)
        rewards.append(json.loads(history[i])[2])
    for i in range(0, history_size, GENERATION_SIZE):
        generations.append(i // GENERATION_SIZE)
        games = min(i + GENERATION_SIZE, history_size) - i
        wins = 0
        for j in range(i, i + games):
            wins += json.loads(history[j])[3]
        winrates.append(wins / games)
        
    plt.subplot(1, 2, 1)
    plt.xlabel('Epoch')
    plt.ylabel('Total Reward')
    plt.plot(
        # epochs, rewards,
        epochs, rewards, 'o'
    )
    
    plt.subplot(1, 2, 2)
    plt.xlabel('Generation ({} Epochs)'.format(GENERATION_SIZE))
    plt.ylabel('Win rate')
    plt.plot(
        generations, winrates,
        generations, winrates, 'o'
    )

    plt.tight_layout()
    plt.show()
    

draw_rewards()