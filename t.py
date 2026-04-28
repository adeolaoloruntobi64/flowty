class FlowGameSolver:
    def __init__(self):
        pass
    
    def get_next(self):
        pass

    def solve_game(self):
        pass
    
    def execute(self):
        pass

class GameInstanceManager:
    NONE = -1
    FLOW_FREE = 0
    FLOW_FREE_BRIDGES = 1
    FLOW_FREE_HEXES = 2
    FLOW_FREE_WARPS = 3
    FLOW_FREE_SHAPES = 5

    def __init__(self):
        self.gametype = GameInstanceManager.NONE
        self.solver = FlowGameSolver()

    def switch_games(self):
        self.gametype += 1
        if self.gametype > 5:
            self.gametype = GameInstanceManager.NONE
            return
    
    def play_game(self):
        print
    
    def play_pack(self):
        pass

    def play_level(self):
        self.solver.solve_game()
        self.solver.get_next()

program = GameInstanceManager()