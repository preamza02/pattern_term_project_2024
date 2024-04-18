from src.utils import append_gym_checker

append_gym_checker()

from checkers.game import Checkers

# from checkers.agents.baselines import play_a_game
# from checkers.game import Checkers
# from checkers.agents.alpha_beta import (
#     MinimaxPlayer,
#     first_order_adv,
#     material_value_adv,
# )
# from tqdm import tqdm
# from functools import partial
# from src.prame_q_learn.agent.white_checker_agent import WhiteCheckerAgent
# from src.enum import RESULT_TYPE

# from .config_train_white_agent import (
#     MAX_GAME_LEN,
#     N_MATCHS,
#     IS_SHOW_GAME,
#     WEIGHT_PATH,
#     EXPLORE_RATE,
#     ALPHA,
#     GAMMA,
#     WIN_SCORE,
#     DRAW_SCORE,
#     LOSE_SCORE,
# )

# # A few matches against a random player

# n_wins, n_draws, n_losses = 0, 0, 0
# for i in tqdm(range(N_MATCHS)):
#     if IS_SHOW_GAME:
#         print("Game", i + 1)
#     ch = Checkers()
#     black_player = MinimaxPlayer(
#         "black",
#         value_func=partial(first_order_adv, "black", 200, 100, 20, 0),
#         rollout_order_gen=lambda x: sorted(x),
#         search_depth=4,
#         seed=i,
#     )

#     white_player = WhiteCheckerAgent(
#         weight_path=WEIGHT_PATH,
#         explore_rate=EXPLORE_RATE,
#         alpha=ALPHA,
#         gamma=GAMMA,
#         win_score=WIN_SCORE,
#         draw_score=DRAW_SCORE,
#         lose_score=LOSE_SCORE,
#     )

#     # modify this function to put our RL model as white
#     winner = play_a_game(
#         ch,
#         black_player.next_move,
#         white_player.next_move,
#         MAX_GAME_LEN,
#         is_show_detail=IS_SHOW_GAME,
#     )
#     if IS_SHOW_GAME:
#         print(
#             "black player evaluated %i positions in %.2fs (avg %.2f positions/s) effective branching factor %.2f"
#             % (
#                 black_player.n_evaluated_positions,
#                 black_player.evaluation_dt,
#                 black_player.n_evaluated_positions / black_player.evaluation_dt,
#                 (black_player.n_evaluated_positions / black_player.ply)
#                 ** (1 / black_player.search_depth),
#             )
#         )
#         print("black player pruned", black_player.prunes.items())
#         print()
#     result: RESULT_TYPE
#     if winner == "black":
#         n_wins += 1
#         result = RESULT_TYPE.LOSE
#     elif winner is None:
#         n_draws += 1
#         result = RESULT_TYPE.DRAW
#     else:
#         n_losses += 1
#         result = RESULT_TYPE.WIN
#     white_player.get_result(result)
#     print(f"Game : {i+1}/{N_MATCHS} result {result.value}")

# print("black win", n_wins, "draw", n_draws, "loss", n_losses)
