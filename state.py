import numpy as np

class State:
    def __init__(self, parameters: np.ndarray):
        self.parameters = parameters

    def transition(self, action):
        new_parameters = self.parameters + action
        return State(new_parameters)
    
if __name__ == '__main__':
    parameters = np.array([[0, 1], [1, 0]])
    initial_state = State(parameters)
    action = np.array([[-1, 0], [-4 , 2]])
    new_state = initial_state.transition(action)
    print(new_state.parameters)
