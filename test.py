import test_helper

if __name__ == '__main__':
    import initial_recommendation

    num_workouts = 3
    allowed_changes_in_one_step = 2

    # To be done manually in production
    setup = test_helper.generate_setup(num_workouts, allowed_changes_in_one_step) 
    specifications = {}

    num_iterations = 10

    state = initial_recommendation.recommendation(setup, specifications)
    state_parameters = state.parameters

    for i in range(num_iterations):
        
        # To be replaced by RL in production
        action = test_helper.generate_prediction(setup)

        state = state.transition(action)

    print(state.parameters)