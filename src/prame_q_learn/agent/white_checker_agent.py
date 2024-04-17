from src.enum import CHECKER_COLOR
from .base_checker_agent import BaseCheckerAgent


class WhiteCheckerAgent(BaseCheckerAgent):
    def __init__(
        self,
        weight_path: str = "./weight/white_weight.json",
        init_state_score: float = 0,
        explore_rate: float = 0.2,
        alpha: float = 0.9,
        gamma: float = 0.5,
        win_score: float = 100,
        draw_score: float = -5,
        lose_score: float = -50,
    ) -> None:
        super().__init__(
            CHECKER_COLOR.WHITE,
            weight_path,
            init_state_score,
            explore_rate,
            alpha,
            gamma,
            win_score,
            draw_score,
            lose_score,
        )
