import numpy as np
from workout import Workout

class Setup:
    def num_possible_changes(self, workout):
        min_changes = workout.min_changes
        max_changes = workout.max_changes
        num_changes = 0
        for i, min_change in enumerate(min_changes):
            max_change = max_changes[i]
            num_changes += 2 * (max_change/min_change)
        return num_changes

    def __init__(self, workouts):
        self.workouts = workouts
        self.num_workouts = len(workouts)
        self.num_actions = np.prod([self.num_possible_changes(workout) 
                                    for workout in self.workouts])

if __name__ == '__main__':
    workouts = []

    for i in range(5):
        workouts.append(Workout(i, min_changes = [5, 1, 1, 1],
                        max_changes = [10, 5, 2, 10]))

    setup = Setup(workouts)

    print(setup.num_actions)