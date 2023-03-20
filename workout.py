import numpy as np

class Workout:
    def verify_definition(self, min_changes, max_changes):
        return True

    def __init__(self, id, min_changes, max_changes):
        self.id = id
        if self.verify_definition(min_changes, max_changes):
            self.min_changes = min_changes
            self.max_changes = max_changes
        else:
            raise Exception(f'Error in the definition of workout {id}')
    
if __name__ == '__main__':
    workout_1 = Workout(1, min_changes = [5, 1, 1, 0.25],
                        max_changes = [10, 5, 2, 10])
