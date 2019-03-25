import pickle
import matplotlib.pyplot as plt

def draw_rewards():
    try:
        history = pickle.load(open('./weights/ml_agent1_0.history', 'rb'))
    except Exception:
        print('Couldn\'t load history file!')
        return
    rewards = list()
    for state in history:
        rewards.append(state[2])
    plt.plot(rewards)
    plt.show()
    

draw_rewards()