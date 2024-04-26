import os
from pathlib import Path
import sys

pwd = Path(os.getcwd())
if (pwd.name == "notebooks"):
    sys.path.append(str(pwd.parent / "gym-checkers-for-thai"))
else:
    sys.path.append(str(pwd / "gym-checkers-for-thai"))

print(pwd)
sys.path.append(str(pwd))

from checkers.game import Checkers
from checkers.agents.baselines import play_a_game
from checkers.game import Checkers
from checkers.agents.alpha_beta import MinimaxPlayer, first_order_adv, material_value_adv
from tqdm import tqdm
from functools import partial
from src.prame_q_learn.agent.white_checker_agent import WhiteCheckerAgent
from src.enum import RESULT_TYPE
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import random

def rollout_order_gen_random(x):
    random.shuffle(x)
    return x

def encode_board_to_feature_map(board):
    output_maps = torch.zeros((4, 8, 4))
    for k,(side, typ) in enumerate([["black","men"], ["black", "kings"], ["white","men"],["white","kings"]]):
        _indices = list(board[side][typ])
        _2d_indices = [[int(i/4), i % 4] for i in _indices]
        _maps = torch.zeros((1,8,4))
        for i,j in _2d_indices:
            _maps[:,i,j] = torch.tensor([1.0])
        output_maps[k, :, :] = _maps
    return output_maps

def encode_action(action):
    output_maps = torch.zeros((2, 8, 4))
    a = [int(action[0] / 4), action[0] % 4]
    b = [int(action[1] / 4), action[1] % 4]
    output_maps[0, a[0],a[1]] = 1.0
    output_maps[1, b[0],b[1]] = 1.0
    return output_maps


class MultiLayerDenseModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes):
        super(MultiLayerDenseModel, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes
        
        # Define layers
        layers = []
        in_features = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(in_features, hidden_size))
            layers.append(nn.ReLU())  # You can change the activation function if needed
            in_features = hidden_size
        layers.append(nn.Linear(in_features, output_size))
        layers.append(nn.Sigmoid())
        
        # Create the sequential model
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.model(x)

# Example usage:
input_size = 128
output_size = 64
hidden_sizes = [256, 512, 256]  # Example hidden layer sizes
model = MultiLayerDenseModel(input_size, output_size, hidden_sizes)
print(model)

loss_fn = torch.nn.MSELoss(reduction="sum")
optimizer = torch.optim.Adam(model.parameters(), lr = 0.0005)

# A few matches against a random player
max_game_len = 100
n_matches = 10000
n_wins, n_draws, n_losses = 0, 0, 0
is_show_game = False
train_batch_size = 32
device =  "cuda"
offset_seed = 0
save_step = 10
model = model.to(device)
train_seqs = []

for i in tqdm(range(n_matches)):
    if is_show_game:
        print('game', i)
    ch = Checkers()
    black_player = MinimaxPlayer(
        'black',
        # value_func=partial(first_order_adv, 'black', 200, 100, 20, 0),
        value_func=partial(first_order_adv, 'black', 86.0315, 54.568, 87.21072, 25.85066),        
        # The provided legal moves might be ordered differently
        rollout_order_gen=rollout_order_gen_random,
        search_depth=2,
        seed=i+offset_seed)

    white_player = MinimaxPlayer(
                    'white',
                    # value_func=partial(first_order_adv, 'black', 200, 100, 20, 0),
                    value_func=partial(first_order_adv, 'white', 86.0315, 54.568, 87.21072, 25.85066),        
                    # The provided legal moves might be ordered differently
                    rollout_order_gen=rollout_order_gen_random,
                    search_depth=4,
                    seed=i+offset_seed)

    #modify this function to put our RL model as white
    winner = play_a_game(ch, black_player.next_move, white_player.next_move, max_game_len,is_show_detail = is_show_game)

    # Play with a minimax player
    # play_a_game(ch, keyboard_player_move, white_player.next_move)
    if is_show_game:
        print('black player evaluated %i positions in %.2fs (avg %.2f positions/s) effective branching factor %.2f' % (black_player.n_evaluated_positions, black_player.evaluation_dt, black_player.n_evaluated_positions / black_player.evaluation_dt, (black_player.n_evaluated_positions / black_player.ply) ** (1 / black_player.search_depth)))
        print('black player pruned', black_player.prunes.items())
        print()
    result:RESULT_TYPE
    if winner == 'black':
        n_wins += 1
        result = RESULT_TYPE.LOSE
    elif winner is None:
        n_draws += 1
        result = RESULT_TYPE.DRAW
    else:
        n_losses += 1
        result = RESULT_TYPE.WIN

    print(f"round : {i+1}/{n_matches} result {result.value}")

    # training 
    if result.value != "win": continue # dont learn if current episode is not winning

    train_seqs.extend(white_player.board_move_dict)

    if i % 2 != 0 or len(train_seqs) == 0: continue
        
    print("start training")
    encoded_boards = []
    encoded_actions = []
    for board, action in train_seqs:
        encoded_boards.append(encode_board_to_feature_map(board))
        encoded_actions.append(encode_action(action))

    encoded_boards = torch.stack(encoded_boards)
    encoded_actions = torch.stack(encoded_actions)

    dataset = TensorDataset(encoded_boards[:encoded_boards.shape[0]//2], encoded_actions[:encoded_boards.shape[0]//2])
    val_dataset = TensorDataset(encoded_boards[encoded_boards.shape[0]//2:], encoded_actions[encoded_boards.shape[0]//2:])

    train_dataloader = DataLoader(dataset, batch_size = train_batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size = train_batch_size)

    # train loop
    for encoded_board, encoded_action in train_dataloader:
        
        encoded_board = encoded_board.to(device)
        encoded_action = encoded_action.to(device)

        encoded_board = encoded_board.flatten(start_dim = 1)
        encoded_action = encoded_action.flatten(start_dim = 1)
        
        predicted_action = model(encoded_board)

        loss = loss_fn(predicted_action, encoded_action)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print("train loss:", loss)

    # val loop
    with torch.no_grad():
        val_loss = 0
        count = 0
        for encoded_board, encoded_action in val_dataloader:
                
                encoded_board = encoded_board.to(device)
                encoded_action = encoded_action.to(device)
        
                encoded_board = encoded_board.flatten(start_dim = 1)
                encoded_action = encoded_action.flatten(start_dim = 1)
                
                predicted_action = model(encoded_board)
        
                loss = loss_fn(predicted_action, encoded_action)
            
                val_loss += loss.item()
                count += 1
        print("val loss", val_loss/count)

    train_seqs = []

    if i % save_step == 0: torch.save({
        "iteration":i,
        "model":model.state_dict(),
        "optimizer":optimizer.state_dict()
        }, open("/home/boat/pattern/pattern_term_project_2024/boat_weight/densenet/modelDenseSomthing_latest_msesum_depth4vs2_randRollout.pt", "wb"))

print('black win', n_wins, 'draw', n_draws, 'loss', n_losses)