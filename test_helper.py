import numpy as np
from workout import Workout
from setup import Setup

def generate_workouts(n):
    workouts = []
    for i in range(n):
        workouts.append(Workout(i, min_changes = [5, 1, 1, 1],
                        max_changes = [10, 5, 2, 2], calorie_formula=None))
    return workouts

def generate_setup(num_workouts, allowed_changes_in_one_step):
    workouts = generate_workouts(num_workouts)
    setup = Setup(workouts, allowed_changes_in_one_step)
    return setup

def generate_prediction(setup: Setup):
    num_workouts = setup.num_workouts
    allowed_changes_in_one_step = setup.allowed_changes_in_one_step
    
    action = np.zeros((num_workouts, 4))

    num_workouts_to_be_changed = np.random.randint(allowed_changes_in_one_step+1)
    indices_of_workouts_to_be_changed = np.random.randint(num_workouts, 
                                                          size=num_workouts_to_be_changed)
    for i in indices_of_workouts_to_be_changed:
        action[i] += setup.workouts[i].min_changes 

    return action