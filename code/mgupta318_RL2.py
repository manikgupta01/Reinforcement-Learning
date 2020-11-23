from hiive import mdptoolbox
import hiive.mdptoolbox.example
import hiive.mdptoolbox.mdp
import gym
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt

def calculate_value(xi):
    xi_result = xi.run()
    # print(xi_result)

    xi_reward, xi_time, xi_mean = [],[],[]

    for x in range(len(xi_result)):
        xi_reward.append(xi_result[x]["Reward"])
        xi_time.append(xi_result[x]["Time"])
        xi_mean.append(xi_result[x]["Mean V"])

    return xi_reward, xi_time, xi_mean

def plot_iteration(title, ylabel, x1, x2, x3, x4, x5):
    plt.figure()
    temp = max(len(x1), len(x2), len(x3), len(x4), len(x5))
    x_1 = np.linspace(1, temp, len(x1))
    x_2 = np.linspace(1, temp, len(x2))
    x_3 = np.linspace(1, temp, len(x3))
    x_4 = np.linspace(1, temp, len(x4))
    x_5 = np.linspace(1, temp, len(x5))
    # print(title, ylabel)
    # print("1", len(x1))
    # print("2", len(x2))
    # print("3", len(x3))
    # print("4", len(x4))
    # print("5", len(x5))
    y_1 = x1
    y_2 = x2
    y_3 = x3
    y_4 = x4
    y_5 = x5
    plt.plot(x_1, y_1, label="Discount F=0.5")
    plt.plot(x_2, y_2, label="Discount F=0.6")
    plt.plot(x_3, y_3, label="Discount F=0.7")
    plt.plot(x_4, y_4, label="Discount F=0.8")
    plt.plot(x_5, y_5, label="Discount F=0.9")
    plt.legend(loc="best")
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel("Iterations")
    plt.ylabel(ylabel)
    plt.grid()
    plt.savefig(title+"_"+ylabel+".png", bbox_inches = "tight")

def plot_iteration_qeps(title, ylabel, x1, x2, x3, x4, x5):
    plt.figure()
    temp = max(len(x1), len(x2), len(x3), len(x4), len(x5))
    x_1 = np.linspace(1, temp, len(x1))
    x_2 = np.linspace(1, temp, len(x2))
    x_3 = np.linspace(1, temp, len(x3))
    x_4 = np.linspace(1, temp, len(x4))
    x_5 = np.linspace(1, temp, len(x5))
    # print(title, ylabel)
    # print("1", len(x1))
    # print("2", len(x2))
    # print("3", len(x3))
    # print("4", len(x4))
    # print("5", len(x5))
    y_1 = x1
    y_2 = x2
    y_3 = x3
    y_4 = x4
    y_5 = x5
    plt.plot(x_1, y_1, label="epsilon=1.0")
    plt.plot(x_2, y_2, label="epsilon=0.7")
    plt.plot(x_3, y_3, label="epsilon=0.5")
    plt.plot(x_4, y_4, label="epsilon=0.3")
    plt.plot(x_5, y_5, label="epsilon=0.1")
    plt.legend(loc="best")
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel("Iterations")
    plt.ylabel(ylabel)
    plt.grid()
    plt.savefig(title+"_"+ylabel+".png", bbox_inches = "tight")

def plot_iteration_qdecay(title, ylabel, x1, x2, x3, x4, x5):
    plt.figure()
    temp = max(len(x1), len(x2), len(x3), len(x4), len(x5))
    x_1 = np.linspace(1, temp, len(x1))
    x_2 = np.linspace(1, temp, len(x2))
    x_3 = np.linspace(1, temp, len(x3))
    x_4 = np.linspace(1, temp, len(x4))
    x_5 = np.linspace(1, temp, len(x5))
    # print(title, ylabel)
    # print("1", len(x1))
    # print("2", len(x2))
    # print("3", len(x3))
    # print("4", len(x4))
    # print("5", len(x5))
    y_1 = x1
    y_2 = x2
    y_3 = x3
    y_4 = x4
    y_5 = x5
    plt.plot(x_1, y_1, label="decay=0.99")
    plt.plot(x_2, y_2, label="decay=0.7")
    plt.plot(x_3, y_3, label="decay=0.5")
    plt.plot(x_4, y_4, label="decay=0.3")
    plt.plot(x_5, y_5, label="decay=0.1")
    plt.legend(loc="best")
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel("Iterations")
    plt.ylabel(ylabel)
    plt.grid()
    plt.savefig(title+"_"+ylabel+".png", bbox_inches = "tight")

def value_iteration(title, P, R):
    vi = mdptoolbox.mdp.ValueIteration(P, R, 0.5)
    vi_reward_d1, vi_time_d1, vi_meanv1 = calculate_value(vi)
    vi = mdptoolbox.mdp.ValueIteration(P, R, 0.6)
    vi_reward_d3, vi_time_d3, vi_meanv3  = calculate_value(vi)
    vi = mdptoolbox.mdp.ValueIteration(P, R, 0.7)
    vi_reward_d5, vi_time_d5, vi_meanv5  = calculate_value(vi)
    vi = mdptoolbox.mdp.ValueIteration(P, R, 0.8)
    vi_reward_d7, vi_time_d7, vi_meanv7  = calculate_value(vi)
    vi = mdptoolbox.mdp.ValueIteration(P, R, 0.9)
    vi_reward_d9, vi_time_d9, vi_meanv9  = calculate_value(vi)

    plot_iteration(title, "Reward", vi_reward_d1, vi_reward_d3, vi_reward_d5, vi_reward_d7, vi_reward_d9)
    plot_iteration(title, "Time", vi_time_d1, vi_time_d3, vi_time_d5, vi_time_d7, vi_time_d9)
    plot_iteration(title, "Mean V", vi_meanv1, vi_meanv3, vi_meanv5, vi_meanv7, vi_meanv9)


def policy_iteration(title, P, R):
    pi = mdptoolbox.mdp.PolicyIteration(P, R, 0.5)
    pi_reward_d1, pi_time_d1, pi_meanv1  = calculate_value(pi)
    pi = mdptoolbox.mdp.PolicyIteration(P, R, 0.6)
    pi_reward_d3, pi_time_d3, pi_meanv3  = calculate_value(pi)
    pi = mdptoolbox.mdp.PolicyIteration(P, R, 0.7)
    pi_reward_d5, pi_time_d5, pi_meanv5  = calculate_value(pi)
    pi = mdptoolbox.mdp.PolicyIteration(P, R, 0.8)
    pi_reward_d7, pi_time_d7, pi_meanv7  = calculate_value(pi)
    pi = mdptoolbox.mdp.PolicyIteration(P, R, 0.9)
    pi_reward_d9, pi_time_d9, pi_meanv9  = calculate_value(pi)

    plot_iteration(title, "Reward", pi_reward_d1, pi_reward_d3, pi_reward_d5, pi_reward_d7, pi_reward_d9)
    plot_iteration(title, "Time", pi_time_d1, pi_time_d3, pi_time_d5, pi_time_d7, pi_time_d9)
    plot_iteration(title, "Mean V", pi_meanv1, pi_meanv3, pi_meanv5, pi_meanv7, pi_meanv9)

def q_learner(title, P, R):
    # ql = mdptoolbox.mdp.QLearning(P, R, gamma=0.95, epsilon=1.0, epsilon_decay=0.05, n_iter=1000000, alpha=0.7, alpha_decay=0.05)
    ql = mdptoolbox.mdp.QLearning(P, R, 0.5, epsilon=0.3, alpha=0.7, epsilon_decay=0.99)
    ql_reward_d1, ql_time_d1, ql_meanv1 = calculate_value(ql)
    ql = mdptoolbox.mdp.QLearning(P, R, 0.6, epsilon=0.3, alpha=0.7, epsilon_decay=0.99)
    ql_reward_d3, ql_time_d3, ql_meanv3 = calculate_value(ql)
    ql = mdptoolbox.mdp.QLearning(P, R, 0.7, epsilon=0.3, alpha=0.7, epsilon_decay=0.99)
    ql_reward_d5, ql_time_d5, ql_meanv5 = calculate_value(ql)
    ql = mdptoolbox.mdp.QLearning(P, R, 0.8, epsilon=0.3, alpha=0.7, epsilon_decay=0.99)
    ql_reward_d7, ql_time_d7, ql_meanv7 = calculate_value(ql)
    ql = mdptoolbox.mdp.QLearning(P, R, 0.9, epsilon=0.3, alpha=0.7, epsilon_decay=0.99)
    ql_reward_d9, ql_time_d9, ql_meanv9 = calculate_value(ql)

    # plot_iteration(title, "Reward", ql_reward_d1, ql_reward_d3, ql_reward_d5, ql_reward_d7, ql_reward_d9)
    plot_iteration(title, "Time", ql_time_d1, ql_time_d3, ql_time_d5, ql_time_d7, ql_time_d9)
    plot_iteration(title, "Mean V", ql_meanv1, ql_meanv3, ql_meanv5, ql_meanv7, ql_meanv9)

def q_learner_hp_eps(title, P, R):
    # ql = mdptoolbox.mdp.QLearning(P, R, gamma=0.95, epsilon=1.0, epsilon_decay=0.05, n_iter=1000000, alpha=0.7, alpha_decay=0.05)
    ql = mdptoolbox.mdp.QLearning(P, R, 0.9, epsilon=1.0, alpha=0.7, epsilon_decay=0.99)
    ql_reward_d1, ql_time_d1, ql_meanv1 = calculate_value(ql)
    ql = mdptoolbox.mdp.QLearning(P, R, 0.9, epsilon=0.7, alpha=0.7, epsilon_decay=0.99)
    ql_reward_d3, ql_time_d3, ql_meanv3 = calculate_value(ql)
    ql = mdptoolbox.mdp.QLearning(P, R, 0.9, epsilon=0.5, alpha=0.7, epsilon_decay=0.99)
    ql_reward_d5, ql_time_d5, ql_meanv5 = calculate_value(ql)
    ql = mdptoolbox.mdp.QLearning(P, R, 0.9, epsilon=0.3, alpha=0.7, epsilon_decay=0.99)
    ql_reward_d7, ql_time_d7, ql_meanv7 = calculate_value(ql)
    ql = mdptoolbox.mdp.QLearning(P, R, 0.9, epsilon=0.1, alpha=0.7, epsilon_decay=0.99)
    ql_reward_d9, ql_time_d9, ql_meanv9 = calculate_value(ql)

    # plot_iteration(title, "Reward", ql_reward_d1, ql_reward_d3, ql_reward_d5, ql_reward_d7, ql_reward_d9)
    plot_iteration_qeps(title, "Time", ql_time_d1, ql_time_d3, ql_time_d5, ql_time_d7, ql_time_d9)
    plot_iteration_qeps(title, "Mean V", ql_meanv1, ql_meanv3, ql_meanv5, ql_meanv7, ql_meanv9)

def q_learner_hp_decay(title, P, R):
    # ql = mdptoolbox.mdp.QLearning(P, R, gamma=0.95, epsilon=1.0, epsilon_decay=0.05, n_iter=1000000, alpha=0.7, alpha_decay=0.05)
    ql = mdptoolbox.mdp.QLearning(P, R, 0.9, epsilon=0.3, alpha=0.7, epsilon_decay=0.99)
    ql_reward_d1, ql_time_d1, ql_meanv1 = calculate_value(ql)
    ql = mdptoolbox.mdp.QLearning(P, R, 0.9, epsilon=0.3, alpha=0.7, epsilon_decay=0.7)
    ql_reward_d3, ql_time_d3, ql_meanv3 = calculate_value(ql)
    ql = mdptoolbox.mdp.QLearning(P, R, 0.9, epsilon=0.3, alpha=0.7, epsilon_decay=0.5)
    ql_reward_d5, ql_time_d5, ql_meanv5 = calculate_value(ql)
    ql = mdptoolbox.mdp.QLearning(P, R, 0.9, epsilon=0.3, alpha=0.7, epsilon_decay=0.3)
    ql_reward_d7, ql_time_d7, ql_meanv7 = calculate_value(ql)
    ql = mdptoolbox.mdp.QLearning(P, R, 0.9, epsilon=0.3, alpha=0.7, epsilon_decay=0.1)
    ql_reward_d9, ql_time_d9, ql_meanv9 = calculate_value(ql)

    # plot_iteration(title, "Reward", ql_reward_d1, ql_reward_d3, ql_reward_d5, ql_reward_d7, ql_reward_d9)
    plot_iteration_qdecay(title, "Time", ql_time_d1, ql_time_d3, ql_time_d5, ql_time_d7, ql_time_d9)
    plot_iteration_qdecay(title, "Mean V", ql_meanv1, ql_meanv3, ql_meanv5, ql_meanv7, ql_meanv9)

def main():
    env = gym.make('FrozenLake-v0')
    # print(type(env.P))
    # print(env.P[0][0][0][3]) # Values
    # for key, value in range(env.P):
    #     print(key)
    P = np.zeros([env.action_space.n, env.observation_space.n, env.observation_space.n])
    R = np.zeros([env.observation_space.n, env.action_space.n])
    # print(P)

    for k1, v1 in env.P.items():
        for k2, v2 in v1.items():
            for x in v2:
                i=0
                for var in x:
                    if i==0:
                        prob = var
                    if i==1:
                        newstate = var
                    if i==2:
                        reward = var
                    i = i+1
                P[k2][k1][newstate] += prob
                R[k1][k2] += reward
    # print(P)
    # print(R)

    # Frozen lake problem
    # P, R = mdptoolbox.example.forest(S=100, r1=50, r2=25)
    value_iteration("VI - Frozen Lake", P, R)
    policy_iteration("PI - Frozen Lake", P, R)
    # q_learner_hp_eps("QL - Frozen Lake HP EPS", P, R)
    # q_learner_hp_decay("QL - Frozen Lake HP DECAY", P, R)
    q_learner("QL - Frozen Lake", P, R)

if __name__ == "__main__":
    main()
