import numpy as np
from state import State
from setup import Setup
from workout import Workout

def recommendation(setup: Setup, specifications):
    num_workouts = setup.num_workouts

    # Initialize a placeholder array for initial workout parameters, to be modified later
    parameters = np.zeros((num_workouts, 4))
    
    state = State(parameters)
    return state

if __name__ == '__main__':
    import test_helper

    setup = test_helper.generate_setup(num_workouts=5, allowed_changes_in_one_step=3)
    specifications = {}

    rec = recommendation(setup, specifications)
    print(rec.parameters)