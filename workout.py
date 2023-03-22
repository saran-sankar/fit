import numpy as np

class Workout:
    def verify_definition(self, min_changes, max_changes):
        return True

    def __init__(self, id, min_changes, max_changes, calorie_formula):
        self.id = id
        if self.verify_definition(min_changes, max_changes):
            self.min_changes = min_changes
            self.max_changes = max_changes
        else:
            raise Exception(f'Error in the definition of workout {id}')
        self.calorie_formula = calorie_formula
    
if __name__ == '__main__':
    import test_helper

    workout_1 = test_helper.generate_workouts(1)[0]
    print(workout_1.min_changes, workout_1.max_changes)
