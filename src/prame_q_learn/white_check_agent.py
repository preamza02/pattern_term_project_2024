from src.setup import append_gym_checker

append_gym_checker()

from checkers.agents import Player

MOVE = tuple[int, int]


class WhiteCheckerAgent(Player):
    def __init__(self) -> None:
        super().__init__(color="white")

    def _calculate_next_move(self, state, legal_move: list[MOVE]) -> tuple[int, int]:
        return

    def next_move(self):
        state = board, self.color, last_moved_piece
        board, turn, last_moved_piece = state
        self.simulator.restore_state(state)
        moves = self.simulator.legal_moves()
        if len(moves) == 1:
            # No other choice
            best_move = moves[0]
        else:
            # More than one legal move
            value, best_move = self.minimax_search(
                state, MinimaxPlayer.loss, MinimaxPlayer.win, self.search_depth, set()
            )
            # print('move', move, 'value', value)
        print(
            "evaluated %i positions in %.2fs (avg %.2f positions/s) with effective branching factor %.2f"
            % (dm, dt, dm / dt, dm ** (1 / self.search_depth))
        )
        self.evaluation_dt += dt
        self.ply += 1
        return best_move
