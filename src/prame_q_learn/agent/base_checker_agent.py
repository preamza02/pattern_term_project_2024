from src.utils import append_gym_checker

append_gym_checker()

import json
import random
from checkers.agents import Player

from ...enum import RESULT_TYPE, CHECKER_COLOR
from ...varible_type import MOVE, WEIGHT


class BaseCheckerAgent(Player):
    def __init__(
        self,
        color: CHECKER_COLOR,
        weight_path: str,
        init_state_score: float = 0,
        explore_rate: float = 0.2,
        alpha: float = 0.9,
        gamma: float = 0.5,
        win_score: float = 100,
        draw_score: float = -5,
        lose_score: float = -50,
    ) -> None:
        super().__init__(color=color.value)
        self._color: CHECKER_COLOR = color
        self._weight_path: str = weight_path
        self._init_state_score: float = init_state_score
        self._explore_rate: float = explore_rate
        self._alpha: float = alpha
        self._gamma: float = gamma
        self._win_score: float = win_score
        self._draw_score: float = draw_score
        self._lose_score: float = lose_score

        self._weight: WEIGHT = self._get_weight()
        self._previos_move_list: list[tuple[str, str]] = []

    def _get_weight(self) -> WEIGHT:
        with open(self._weight_path) as json_file:
            weight: WEIGHT = json.load(json_file)
            json_file.close()
            return weight

    def _dump_weight_json(self) -> None:
        with open(self._weight_path, "w") as json_file:
            json.dump(self._weight, json_file)

    def _update_weight(self, reward: float) -> None:
        reverse_previous_move_list: list[tuple[str, str]] = self._previos_move_list[
            ::-1
        ]
        last_state: str = reverse_previous_move_list[0][0]
        for i in range(len(reverse_previous_move_list)):
            current_state: str = reverse_previous_move_list[i][0]
            selected_move_str: str = reverse_previous_move_list[i][1]
            if i == 0:
                self._weight[current_state][selected_move_str] = reward
                last_state = current_state
            else:
                last_state_max_score: float = max(self._weight[last_state].values())
                this_old_score: float = self._weight[current_state][selected_move_str]
                this_new_score: float = this_old_score + self._alpha * (
                    reward + self._gamma * (last_state_max_score) - this_old_score
                )

                self._weight[current_state][selected_move_str] = this_new_score
        self._dump_weight_json()

    def _convert_board_2_str(self, board: dict) -> str:
        return str(board)

    def _convert_move_2_str(self, move: MOVE) -> str:
        return f"{move[0]}/{move[1]}"

    def _convert_back_str_2_move(self, move_str: str) -> MOVE:
        tmp: list[str] = move_str.split("/")
        return (int(tmp[0]), int(tmp[1]))

    def _calculate_next_move(self, state: str, legal_move: list[MOVE]) -> MOVE:
        decided_move: MOVE
        decided_move_str: str
        # if state not in create new state
        this_explore_rate: float = self._explore_rate
        if state not in self._weight:
            self._weight[state] = {}
            for move in legal_move:
                move_str: str = self._convert_move_2_str(move)
                self._weight[state][move_str] = self._init_state_score
                # to make our model random every time we create new
                this_explore_rate = 0

        move_with_score: dict[str, float] = self._weight[state]
        if random.random() < this_explore_rate:
            decided_move_str = random.choice(list(move_with_score.keys()))
        else:
            decided_move_str: str = max(
                move_with_score, key=lambda k: move_with_score[k]
            )
        decided_move = self._convert_back_str_2_move(decided_move_str)
        self._previos_move_list.append((state, decided_move_str))
        return decided_move

    def next_move(self, board, last_moved_piece) -> MOVE:
        # fix
        state = board, self.color, last_moved_piece
        self.simulator.restore_state(state)
        legal_moves = self.simulator.legal_moves()
        # up to your model
        state_string: str = self._convert_board_2_str(board)
        best_move = legal_moves[0]
        best_move = self._calculate_next_move(state_string, legal_moves)
        return best_move

    def get_result(self, result: RESULT_TYPE) -> None:
        reward: float
        if result == RESULT_TYPE.WIN:
            reward = self._win_score
        elif result == RESULT_TYPE.DRAW:
            reward = self._draw_score
        elif result == RESULT_TYPE.LOSE:
            reward = self._lose_score
        else:
            raise TypeError("This is a type error message")
        self._update_weight(reward)
