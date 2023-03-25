import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from workout import Workout
from setup import Setup
import pandas as pd
import random
import matplotlib.pyplot as plt

def create_q_model(num_actions, input_shape):
    inputs = layers.Input(shape=input_shape)
    layer1 = layers.Flatten()(inputs)
    action = layers.Dense(num_actions, activation="linear")(layer1)

    return keras.Model(inputs=inputs, outputs=action)

def reward_function(calories, closeness):
    reward = (calories - 1000 * closeness)/1000
    reward = np.clip(reward, -1, 1)

    return reward

def generate_setup():
    workouts = []
    running = Workout(0, min_changes = [1],
                        max_changes = [2], calorie_formula=None)
    workouts.append(running)

    setup = Setup(workouts, allowed_changes_in_one_step=1)

    return setup

def run_step(model, input, epsilon):
    # Convert input to a batch of size 1
    input_batch = np.expand_dims(input, axis=0)

    # Use the model to predict Q-values for the current state
    q_values = model.predict(input_batch)

    # Select an action based on the epsilon-greedy policy
    if np.random.rand() < epsilon:
        # Choose a random action
        action = np.random.randint(q_values.shape[1])
    else:
        # Choose the action with the highest Q-value
        action = np.argmax(q_values)

    # Return the selected action
    return q_values, action

def run_process():
    setup = generate_setup()

    num_actions = setup.num_actions

    # Specify input shape excluding user specific parameters (for preliminary example)
    # The inputs in this example are past activity of the user, 
    # and the system’s past recommendations (for the last 6 days).
    # In our case, an activity is characterized by a single parameter.
    input_shape = (6, 2, 1,)

    model = create_q_model(num_actions, input_shape)

    # An ad-hoc tool to help generate new recommendations
    # from model predictions. This is highly specific for our application.
    action_map = {0: [-2],
                  1: [-1],
                  2: [1],
                  3: [2]}

    df = pd.read_csv('examples/dailyActivity.csv')

    def reset_input(loc):
        input = np.zeros(input_shape)
        input[-1][0] = df.iloc[loc]['TotalDistance']
        initial_recommendation = np.random.randint(6, size=input_shape[2])
        input[-1][1] = initial_recommendation
        return input
    
    input = reset_input(loc=0)

    user_id = df.iloc[0]['Id']

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer='adam', loss='binary_crossentropy')

    target_model = keras.models.clone_model(model)
    target_model.set_weights(model.get_weights())

    gamma = 0.95
    replay_buffer = []
    replay_buffer_size = 5

    num_iterations = 939

    rewards = {user_id: []}
    closeness_scores = {user_id: []}

    for i in range(1, num_iterations + 1):
        data = df.iloc[i]

        if data['Id'] != user_id:
            # Reset the input and replay buffer when we get to a new user
            input = reset_input(loc=i)
            replay_buffer = []
            user_id = data['Id']
            rewards[user_id] = []
            closeness_scores[user_id] = []

        else:
            q_values, action = run_step(model, input, epsilon=1.0)
            recommendation = max ([0.], input[-1][1] + action_map[action])

            calories = data['Calories']
            closeness = np.linalg.norm(recommendation - data['TotalDistance'])

            closeness_scores[user_id].append(closeness)

            print(f"closeness={closeness}")

            reward = reward_function(calories, closeness)

            rewards[user_id].append(reward)

            print(f"reward={reward}")

            # Store experience in replay buffer
            replay_buffer.append((input, action, reward, recommendation))
            if len(replay_buffer) > replay_buffer_size:
                replay_buffer.pop(0)

            # Calculate the target Q-value
            if len(replay_buffer) == replay_buffer_size:
                # Sample a batch of experiences from the replay buffer
                batch = random.sample(replay_buffer, 1)

                # Calculate the target Q-value for each experience in the batch
                for input_batch, action_batch, reward_batch, recommendation_batch in batch:
                    input_batch = np.expand_dims(input_batch, axis=0)
                    target_q_values = model.predict(input_batch)

                    # Update the Q-value for the selected action based on the reward and the maximum Q-value
                    # for the next state (using the target network)
                    next_input = np.copy(input_batch)
                    next_input[0, :-1, :, :] = input_batch[0, 1:, :, :]
                    next_input[0, -1, 0, 0] = df.iloc[i]['TotalDistance']
                    next_input[0, -1, 1, 0] = recommendation_batch[0]
                    next_q_values = target_model.predict(next_input)
                    max_next_q_value = np.max(next_q_values)
                    target_q_values[0][action_batch] = reward_batch + gamma * max_next_q_value

                    # Train the model on the experience
                    model.fit(input_batch, target_q_values, epochs=1, verbose=0)

    return rewards, closeness_scores

rewards, closeness_scores = run_process()

# Save closeness scores for each user
for user_id, scores in closeness_scores.items():
    np.save(f'user_{user_id}_closeness.npy', scores)

# Plot closeness scores for each user
ig, axs = plt.subplots(nrows=4, ncols=3, figsize=(12, 8))

for i, user_id in enumerate(list(closeness_scores.keys())[-12:]):
    row = i // 3
    col = i % 3
    x = range(1, len(closeness_scores[user_id]) + 1)
    y = closeness_scores[user_id]
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    axs[row, col].plot(x, y)
    axs[row, col].plot(x, p(x), 'r--')
    axs[row, col].plot(range(1, len(closeness_scores[user_id]) + 1), closeness_scores[user_id])
    axs[row, col].set_xlabel('Timesteps')
    axs[row, col].set_ylabel('Closeness')
    axs[row, col].set_title(f'User {user_id}')

plt.tight_layout()
plt.show()