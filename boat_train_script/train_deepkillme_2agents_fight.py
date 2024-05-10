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
from checkers.agents.baselines_deepkillme import play_a_game
from checkers.game import Checkers
from checkers.agents.alpha_beta import MinimaxPlayer, first_order_adv, material_value_adv
from tqdm import tqdm
from functools import partial
from src.prame_q_learn.agent.white_checker_agent import WhiteCheckerAgent
from src.enum import RESULT_TYPE
import torch
from torch import optim
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import random
from src.utils import append_gym_checker
append_gym_checker()
import json
from checkers.agents import Player
import copy
from pathlib import Path
import numpy as np
import random
import math
from torch.utils.tensorboard import SummaryWriter

num_in_feat = 8*4*4
possible_moves = {}
inv_possible_moves = {}
count = 0
for from_row in range(8):
    for from_col in range(4):
        for to_row in range(8):
            for to_col in range(4):
                if from_row == to_row and from_col == to_col: continue
                possible_moves[f"{from_row},{from_col},{to_row},{to_col}"] = count
                inv_possible_moves[count] = [from_row, from_col, to_row, to_col]
                count+=1

class DeepKILLme(nn.Module):
    def __init__(self, in_feat, out_feat):
        super(DeepKILLme, self).__init__()
        # Define layers
        self.fc1 = nn.Linear(in_feat, 256)  # First fully connected layer
        self.fc2 = nn.Linear(256, 256)          # Second fully connected layer
        self.fc3 = nn.Linear(256, 256)          # Third fully connected layer
        self.fc4 = nn.Linear(256, 512)           # Fourth fully connected layer
        self.fc5 = nn.Linear(512, out_feat)  # Final output layer
        self.softmax = nn.Softmax(dim = 1)
        # Define activation function
        self.relu = nn.ReLU()
    
    def masked_softmax(self, vec, mask, dim=1):
        masked_vec = vec * mask.float()
        max_vec = torch.max(masked_vec, dim=dim, keepdim=True)[0]
        exps = torch.exp(masked_vec-max_vec)
        masked_exps = exps * mask.float()
        masked_sums = masked_exps.sum(dim, keepdim=True)
        zeros=(masked_sums == 0)
        masked_sums += zeros.float()
        return masked_exps/masked_sums

    def forward(self, x, legal_moves):
        # Forward pass through the network
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.fc5(x)

        legal_moves_batch = []
        for i in range(len(x)):
            legal_moves_ = legal_moves[i]
            encoded_legal_moves =  torch.stack([encode_action(legal_move_) for legal_move_ in legal_moves_]).to("cuda")
            encoded_legal_moves = encoded_legal_moves.sum(dim=0)

            legal_moves_batch.append(encoded_legal_moves)
        
        legal_moves_batch = torch.stack(legal_moves_batch).to("cuda")

        x = self.masked_softmax(x, legal_moves_batch)
        
        # print("x",x)
        return x

# Instantiate the deeper model
model = DeepKILLme(num_in_feat, len(possible_moves))
print(model)

def encode_board(board):
    output_maps = torch.zeros((4, 8, 4))
    for k,(side, typ) in enumerate([["black","men"], ["black", "kings"], ["white","men"],["white","kings"]]):
        _indices = list(board[side][typ])
        _2d_indices = [[int(i/4), i % 4] for i in _indices]
        _maps = torch.zeros((1,8,4))
        for i,j in _2d_indices:
            _maps[:,i,j] = torch.tensor([1.0])
        output_maps[k, :, :] = _maps
    output_maps = output_maps.flatten()
    return output_maps

def encode_action(action):
    onehot = torch.zeros((len(possible_moves)))
    a = [int(action[0] / 4), action[0] % 4]
    b = [int(action[1] / 4), action[1] % 4]
    index = possible_moves[f"{a[0]},{a[1]},{b[0]},{b[1]}"]
    onehot[index] = torch.tensor([1.0])
    return onehot

def decode_action(encoded_action):
    max_id = encoded_action.argmax(dim=0)

    x = inv_possible_moves[max_id.item()]
    
    max_pos_a = (x[0],x[1])
    max_pos_b = (x[2],x[3])

    pred_move = ((max_pos_a[0] * 4) + max_pos_a[1], (max_pos_b[0] * 4) + max_pos_b[1])
    return pred_move



class DeepKILLmePlayer(Player):
    def __init__(
        self,
        color,
        memory,
    ) -> None:
        super().__init__(color=color)
        self._color = color

        self.policyModel = DeepKILLme(num_in_feat, len(possible_moves)).to("cuda")
        self.targetModel = DeepKILLme(num_in_feat, len(possible_moves)).to("cuda")
        self.targetModel.load_state_dict(self.policyModel.state_dict())
        # extra
        self.modelName = "DeepKILLme" #use to told the env to push reward and next state for us
        self.memory = memory
        self._epsilon = 0.1
        self._illegal_move_penalty = 0

    def next_move(self, board, last_moved_piece):
        # fix
        global global_epsilon
        self._epsilon = global_epsilon
        state = board, self.color, last_moved_piece
        self.simulator.restore_state(state)
        legal_moves = self.simulator.legal_moves()
        # up to your model
        move = None
        reward = None
        # print(board)
        encoded_board = encode_board(board).to("cuda")[None,:]
        encoded_pred_action = self.policyModel(encoded_board, [legal_moves]).to("cuda")
        if random.random() > self._epsilon:
            pred_move = decode_action(encoded_pred_action[0])

            move = pred_move
        else:
            move = random.choice(legal_moves)

        # board move next_board reward legal_move_of_board 
        if self.memory.__len__() > 0:
            self.memory[-1][5] = legal_moves # append previous legal moves
        self.memory.append([copy.deepcopy(board), copy.deepcopy(move), None, reward, legal_moves, None]) #board, action, next board, reward (next board will be fill by env later) (reward will fill by model if illegal move, else fill by model)
        
        return move
    
    def get_model_state(self):
        model_states = {
            "policyModel_state_dict":self.policyModel.state_dict(),
            "targetModel_state_dict":self.targetModel.state_dict(),
        }
        return model_states
    
    def load_model_state(self, model_states):
        self.policyModel.load_state_dict(model_states["policyModel_state_dict"])
        self.targetModel.load_state_dict(model_states["targetModel_state_dict"])

# to adjust
BATCH_SIZE = 512
GAMMA = 0.9
EPS = 0.1
EPS_START = 0.9
EPS_END = 0.05
# TAU = 0.005
TAU_STEP = 200
LR = 1e-4
global_epsilon = 0.9
max_game_len = 100
n_matches = 1000000
EPS_DECAY = 10000
is_show_game = False
explore_rate=0.1
save_steps = 1000
MAX_MEM_CAPACITY = 10000
white_memory = []
black_memory = []

white_player = DeepKILLmePlayer("white", white_memory)
black_player = DeepKILLmePlayer("black", black_memory)

white_optimizer = optim.AdamW(white_player.policyModel.parameters(), lr=LR, amsgrad=True)
black_optimizer = optim.AdamW(black_player.policyModel.parameters(), lr=LR, amsgrad=True)

rand = 1203
n_wins, n_draws, n_losses = 0, 0, 0

save_path = Path("/home/boat/pattern/pattern_term_project_2024/boat_weight/deepkillme_2agents_newRewardSystem_AliveReward_Gamma09_legalMasking")
log_path = save_path / "logs"
writer = SummaryWriter(log_path)

# def rollout_order_gen_random(x):
#     random.shuffle(x)
#     return x

result_to_int_dict = {
    RESULT_TYPE.LOSE.value: 0,
    RESULT_TYPE.DRAW.value: 1,
    RESULT_TYPE.WIN.value: 2
}

def train(player, optimizer, iteration):
    # impl train here
    player.memory = list(filter(lambda x: x[2] is not None and x[5] is not None, player.memory, ))
    print(f"{player._color} player memory size {player.memory.__len__()}")
    if  player.memory.__len__() < BATCH_SIZE: return;
    # clip memory
    if player.memory.__len__() > MAX_MEM_CAPACITY:
        player.memory = player.memory[player.memory.__len__() - MAX_MEM_CAPACITY:]

    # filter none (maybe in case of multiple capture, the condition in baseline.py is not cover this case) TODO: impl multiple capture support

    transitions = random.sample(player.memory, BATCH_SIZE)

    state_batch = []
    next_state_batch = []
    action_batch = []
    reward_batch = []
    legal_moves_batch = []
    legal_moves_next_batch = []
    for board, action, next_board, reward, legal_moves_, legal_moves_next_ in transitions:
        encoded_board = encode_board(board)
        encoded_next_board = encode_board(next_board)
        encoded_action = encode_action(action)
        reward = torch.tensor(reward)

        state_batch.append(encoded_board)
        next_state_batch.append(encoded_next_board)
        action_batch.append(encoded_action)
        reward_batch.append(reward)
        legal_moves_batch.append(legal_moves_)
        legal_moves_next_batch.append(legal_moves_next_)

    state_batch = torch.stack(state_batch).to("cuda")
    next_state_batch = torch.stack(next_state_batch).to("cuda")
    action_batch = torch.stack(action_batch).to("cuda")
    reward_batch = torch.stack(reward_batch).to("cuda")
    
    print("len batch", len(state_batch), len(next_state_batch), len(action_batch), len(reward_batch))

    # forward 
    state_action_values = player.policyModel(state_batch, legal_moves_batch)
    with torch.no_grad():
        next_state_values = player.targetModel(next_state_batch, legal_moves_next_batch).max(1).values
    # print("next_state_values",next_state_values.shape)

    # compute expected Q
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(player.policyModel.parameters(), 100)
    optimizer.step()

    # update target model using policy weight
    target_net_state_dict = player.targetModel.state_dict()
    policy_net_state_dict = player.policyModel.state_dict()
    if i % TAU_STEP == 0:
    # for key in policy_net_state_dict:
        # target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        player.targetModel.load_state_dict(policy_net_state_dict)

    # decay eps
    # global_epsilon = EPS_END + (EPS_START - EPS_END) * \
    #     math.exp(-1. * i / EPS_DECAY)
    # use static eps
    global_epsilon = EPS

    # save model
    if i % save_steps == 0:
        states = {
            "models": player.get_model_state(),
            "iteration": i,
            "optimizer_state_dict": optimizer.state_dict(),
        }
        torch.save(states, open(str(save_path / f"deepkillme_{player._color}_it_{iteration}.pt"), "wb"))
    
    return loss

for i in tqdm(range(n_matches)):
    if is_show_game:
        print('game', i)
    ch = Checkers()
    #modify this function to put our RL model as white
    winner = play_a_game(ch, black_player.next_move, white_player.next_move, max_game_len,is_show_detail = is_show_game, white_player=white_player, black_player=black_player)

    white_loss = train(white_player, white_optimizer, i)
    black_loss = train(black_player, black_optimizer, i)        

    print(f"round : {i+1}/{n_matches} result {winner} wins")
    print("black_loss", black_loss)
    print("white_loss", white_loss)

    white_reward_sum = sum([i[3] for i in white_player.memory])
    black_reward_sum = sum([i[3] for i in black_player.memory])

    print("white reward sum", white_reward_sum)
    print("black reward sum", black_reward_sum)

    if black_loss:
        writer.add_scalar('black_train_loss', black_loss.item(), i)
    if white_loss:
        writer.add_scalar('white_train_loss', white_loss.item(), i)
    # writer.add_scalar('global_epsilon', global_epsilon, i)
    writer.add_scalar('black_reward_sum', black_reward_sum, i)
    writer.add_scalar('white_reward_sum', white_reward_sum, i)
    
    if winner:
        writer.add_text("winner", winner, i)
    else:
        writer.add_text("winner", "draw", i)


print('black win', n_wins, 'draw', n_draws, 'loss', n_losses)