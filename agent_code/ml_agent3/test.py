import json
import matplotlib.pyplot as plt

GENERATION_SIZE = 1000

def draw_rewards():
    try:
        history = open('./weights/ml_agent3_history.json', 'r').readlines()
    except Exception:
        print('Couldn\'t load history file!')
        return
    epochs = list()
    rewards = list()
    generations = list()
    winrates = list()
    avgrewards = list()
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
    for i in range(0, history_size, GENERATION_SIZE):
        sum_rewards = rewards[i:i+GENERATION_SIZE]
        avgrewards.append(sum(sum_rewards) / len(sum_rewards))
        
    # plt.subplot(1, 2, 1)
    fig1, ax1 = plt.subplots()
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Total Reward')
    ax1.plot(
        # epochs, rewards,
        epochs, rewards, 'o'
    )
    
    fig1.tight_layout()
    fig1.savefig('agent3_0.png')
    # plt.show()
    # plt.clf()

    # plt.subplot(1, 2, 2)
    fig2, ax2 = plt.subplots()
    ax2.set_xlabel('Generation ({} Epochs)'.format(GENERATION_SIZE))
    ax2.set_ylabel('Win rate')
    ax2.plot(
        generations, winrates,
        generations, winrates, 'o'
    )

    fig2.tight_layout()
    fig2.savefig('agent3_1.png')
    # plt.show()

    fig3, ax3 = plt.subplots()
    ax3.set_xlabel('Generation ({} Epochs)'.format(GENERATION_SIZE))
    ax3.set_ylabel('Avg reward')
    ax3.plot(
        generations, avgrewards,
        generations, avgrewards, 'o'
    )

    fig3.tight_layout()
    fig3.savefig('agent3_2.png')
    

draw_rewards()