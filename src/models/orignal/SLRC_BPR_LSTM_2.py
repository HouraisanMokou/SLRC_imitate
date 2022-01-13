import torch
import logging
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.distributions as distributions
import torch.nn.functional as F
from torch.utils.data import Dataset as BaseDataset
from torch.nn.utils.rnn import pad_sequence
from torch.distributions import normal, exponential
from typing import NoReturn, List

import os

from common.utils import *
from common.constants import *
from common.reader import Reader
from models.SLRC import SLRC


class SLRC_BPR_LSTM_2(SLRC):
    def _init_hawks_weights(self):
        self.sk = 0.01

        # self.hidden_dim = 32
        # self.linear = nn.Linear(self.hidden_dim * 2, self.hidden_dim * 7, bias=True)
        self.h_0i = nn.Embedding(self.item_num, 3)
        self.c_0i = nn.Embedding(self.item_num, 3)
        self.lstm = nn.LSTM(input_size=1, hidden_size=1, num_layers=3, batch_first=True)

    def _init_cf_weights(self) -> NoReturn:
        self.user_embed = nn.Embedding(self.user_num, self.emb_size)
        self.item_embed = nn.Embedding(self.item_num, self.emb_size)

    def _forward_excit(self, feed_dict):
        """
        :param feed_dict:
        :return:
        """
        items = feed_dict["item_id"]  # the 1st is tested and other is neg
        time_interval = feed_dict["time_intervals"] / TIME_SCALE
        mask = (time_interval > 0).float()

        h0 = self.h_0i(items)
        c0 = self.c_0i(items)
        # h0=torch.permute(h0,(1,0,2))
        # c0 = torch.permute(c0, (1, 0, 2))
        # time_interval=torch.permute(time_interval, (0, 2, 1))
        # hs = self.lstm(time_interval,(h0,c0))
        ti2 = time_interval.view(-1, time_interval.shape[2]).float().contiguous()
        h02 = h0.view(-1, h0.shape[2])
        c02 = c0.view(-1, c0.shape[2])
        ti2 = ti2[:, :, None]
        h02 = h02[:, :, None]
        c02 = c02[:, :, None]
        h02 = torch.permute(h02, (1, 0, 2)).contiguous()
        c02 = torch.permute(c02, (1, 0, 2)).contiguous()
        hs = self.lstm(ti2, (h02, c02))[0]
        hs = hs.squeeze()
        hs = hs.view(-1, time_interval.shape[1], time_interval.shape[2])
        hs = hs * mask
        excit = torch.sum(hs, 2)
        return excit

    def _forward_CF(self, feed_dict):
        users = feed_dict["user_id"]
        items = feed_dict["item_id"]
        u_vector = self.user_embed(users)
        i_vector = self.item_embed(items)

        base = (u_vector[:, None, :] * i_vector).sum(
            -1
        )  # the dim of users and items is not the same
        return base
