import numpy as np
from workout import Workout
from setup import Setup

def generate_workouts(n):
    workouts = []
    for i in range(2):
        workouts.append(Workout(i, min_changes = [5, 1, 1, 1],
                        max_changes = [10, 5, 2, 2]))
    return workouts

def generate_setup(num_workouts):
    workouts = generate_workouts(num_workouts)
    setup = Setup(workouts)
    return setup