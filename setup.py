import numpy as np
from workout import Workout
import itertools

class Setup:
    def num_possible_changes(self, workout):
        # Calculate the maximum number of possible changes to a workout in one step
        min_changes = workout.min_changes
        max_changes = workout.max_changes
        num_changes = 0
        for i, min_change in enumerate(min_changes):
            max_change = max_changes[i]
            num_changes += 2 * (max_change/min_change)
        return num_changes

    def __init__(self, workouts: list[Workout], allowed_changes_in_one_step: int):
        # Initialize the Setup object with a list of Workout objects and the maximum number of allowed changes in one step
        self.workouts = workouts
        self.num_workouts = len(workouts)
        self.allowed_changes_in_one_step = allowed_changes_in_one_step

        # Calculate the total number of possible actions
        self.num_actions = sum(np.prod(combination) 
                               for combination in itertools.combinations(
            [self.num_possible_changes(workout) 
             for workout in self.workouts], 
             allowed_changes_in_one_step))

if __name__ == '__main__':
    # Generate a sample setup and print the number of possible action sequences
    import test_helper

    setup = test_helper.generate_setup(num_workouts=3, allowed_changes_in_one_step=2)
    print(setup.num_actions)