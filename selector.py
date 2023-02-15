class Selector:
    def __init__(self):
        pass

    def check_precondition(self):
        raise NotImplementedError

    def generate_candidate_goal_list(self, goal_list):
        raise NotImplementedError

    def horizon_select(self, candidate_goal_list):
        raise NotImplementedError