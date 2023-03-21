import numpy as np
from workout import Workout
import itertools

class Setup:
    def num_possible_changes(self, workout):
        min_changes = workout.min_changes
        max_changes = workout.max_changes
        num_changes = 0
        for i, min_change in enumerate(min_changes):
            max_change = max_changes[i]
            num_changes += 2 * (max_change/min_change)
        return num_changes

    def __init__(self, workouts: list[Workout], allowed_changes_in_one_step: int):
        self.workouts = workouts
        self.num_workouts = len(workouts)
        self.num_actions = sum(np.prod(combination) 
                               for combination in itertools.combinations(
            [self.num_possible_changes(workout) 
             for workout in self.workouts], 
             allowed_changes_in_one_step))

if __name__ == '__main__':
    import test

    setup = test.generate_setup(num_workouts=3, allowed_changes_in_one_step=2)
    print(setup.num_actions)