import gym
import numpy as np
np.bool8 = np.bool_

def qlearning(*, n_actions: int, n_states: tuple, name: str):
    
    # Training environment without rendering
    env_train = gym.make(name)


    q_table = np.zeros((*n_states, n_actions))

    alpha = 0.5
    epsilon = 1
    epsilon_decay = 0.995
    gamma = 0.9
    min_epsilon = 0.01
    max_steps = 100
    episodes = 10_000

    state, info = env_train.reset()
    for episode in range(episodes):
        for step in range(max_steps):
            if np.random.uniform() < epsilon:
                action = env_train.action_space.sample()
            else:
                action = q_table[state, :].argmax()
            new_state, reward, done, trunc, info = env_train.step(action)
            new_q = q_table[new_state, :].max()
            q_table[state, action] = (1 - alpha) * q_table[state, action] + alpha * (reward + gamma * new_q)
            state = new_state
            if done or trunc:
                state, info = env_train.reset()
                break
        if episode % 1000 == 0:
            print(f"Episode: {episode}")
        epsilon = max(min_epsilon, epsilon * epsilon_decay)

    # Evaluation phase with rendering
    env_test = gym.make(name, render_mode="human")
    state, info = env_test.reset()
    for _ in range(5):
        env_test.render()
        for step in range(max_steps):
            action = q_table[state, :].argmax()
            new_state, reward, done, trunc, info = env_test.step(action)
            state = new_state
            if done or trunc:
                state, info = env_test.reset()
                break