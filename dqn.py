from tensorflow import keras
import tensorflow as tf
import numpy as np
from collections import deque

tf.random.set_seed(42)
np.random.seed(42)

def get_input_shape(l):
    return np.sum([len(_) for _ in l])

# define state as [V_r, V_w, V_p, d, w]
# initial state
V_w = [0] # a single workout with 1 param
V_r = [1]
V_p = [200, 60, 0] # weight(lb), height(in), gender(0 for male 1 for female)
d = [0] # 0-7 corresponding to day of week
w = [0] # week number since starting program
initial_state = [0, 1, 200, 60, 0, 0, 0]

input_shape = [get_input_shape([V_r, V_w, V_p, d, w])]
assert get_input_shape([V_r, V_w, V_p, d, w]) == [7]

# define possible actions as [-2,-1,0,1,2]
actions = [-2, -1, 0, 1, 2]
n_outputs = len(actions)

terminal_weight = V_p[0] - 10 # lose 10 lbs

# define model
model = keras.models.Sequential([
    keras.layers.Dense(32, activation="relu", input_shape=input_shape),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(n_outputs)
])

# define policy
def epsilon_greedy_policy(state, epsilon=0):
    if np.random.rand() < epsilon:
        return actions[np.random.randint(n_outputs)]
    else:
        Q_values = model.predict(state[np.newaxis])
        return np.argmax(Q_values[0])

# init replay memory for dqn
replay_memory = deque(maxlen=2000)

def sample_experiences(batch_size):
    indices = np.random.randint(len(replay_memory), size=batch_size)
    batch = [replay_memory[index] for index in indices]
    states, actions, rewards, next_states, dones = [
        np.array([experience[field_index] for experience in batch])
        for field_index in range(5)]
    return states, actions, rewards, next_states, dones

def calculate_reward(state, action, next_state):
    weight_0 = state[2] # weight
    weight_1 = next_state[2]
    v_r = next_state[0]
    v_w = next_state[1]
    workout_closeness = np.sum(np.square(v_r - v_w))
    return (1/(1+workout_closeness)) * (weight_1 - weight_0)

def simulate_step(state, action):
    # hardcoded for a single workout with 1 param
    action_arr = np.zeros(len(state))
    action_arr[0] = action
    next_state = state + action_arr
    reward = calculate_reward(state, action, next_state)
    done = 1 if next_state[2] <= terminal_weight else 0
    return next_state, reward, done

def run_one_step(state, epsilon):
    action = epsilon_greedy_policy(state, epsilon)
    next_state, reward, done = simulate_step(state, action)
    replay_memory.append((state, action, reward, next_state, done))
    return next_state, reward, done

batch_size = 32
discount_rate = 0.99
optimizer = keras.optimizers.Adam(learning_rate=1e-2)
loss_fn = keras.losses.mean_squared_error

def training_step(batch_size):
    experiences = sample_experiences(batch_size)
    states, actions, rewards, next_states, dones = experiences
    next_Q_values = model.predict(next_states)
    max_next_Q_values = np.max(next_Q_values, axis=1)
    target_Q_values = (rewards +
                       (1 - dones) * discount_rate * max_next_Q_values)
    target_Q_values = target_Q_values.reshape(-1, 1)
    mask = tf.one_hot(actions, n_outputs)
    with tf.GradientTape() as tape:
        all_Q_values = model(states)
        Q_values = tf.reduce_sum(all_Q_values * mask, axis=1, keepdims=True)
        loss = tf.reduce_mean(loss_fn(target_Q_values, Q_values))
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))


rewards = [] 
best_score = 0

for episode in range(600):
    state_0 = initial_state
    for step in range(180):
        epsilon = max(1 - episode / 600, 0.01)
        state_1, reward, done = run_one_step(state_0, epsilon)
        if done:
            break
    rewards.append(step)
    if step >= best_score:
        best_weights = model.get_weights()
        best_score = step
    print("\rEpisode: {}, Steps: {}, eps: {:.3f}".format(episode, step + 1, epsilon), end="")
    if episode > 50:
        training_step(batch_size)

model.set_weights(best_weights)