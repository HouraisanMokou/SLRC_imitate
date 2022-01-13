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

Layer1 = 16
Layer2 = 32
Layer3 = 16


class MLP_BPR(SLRC):
    def _init_hawks_weights(self):
        self.global_alpha = nn.Parameter(torch.tensor(1.0))
        self.alpha = nn.Embedding(self.item_num, 1)
        self.w1 = nn.Embedding(self.item_num, 1 * Layer1)
        self.b1 = nn.Embedding(self.item_num, Layer1)
        self.w2 = nn.Embedding(self.item_num, Layer1 * Layer2)
        self.b2 = nn.Embedding(self.item_num, Layer2)
        self.w3 = nn.Embedding(self.item_num, Layer2 * Layer3)
        self.b3 = nn.Embedding(self.item_num, Layer3)
        self.w4 = nn.Embedding(self.item_num, Layer3 * 1)
        self.b4 = nn.Embedding(self.item_num, 1)

        self.af = nn.LeakyReLU(1e-2)

    def _init_cf_weights(self) -> NoReturn:
        self.user_embed = nn.Embedding(self.user_num, self.emb_size)
        self.item_embed = nn.Embedding(self.item_num, self.emb_size)

    def _forward_excit(self, feed_dict):
        """
        :param feed_dict:
        :return:
        """
        items = feed_dict["item_id"]  # the 1st is tested and other is neg
        time_interval = (feed_dict["time_intervals"] / TIME_SCALE).float()
        mask = (time_interval > 0).float()

        alpha_b = self.alpha(items)
        alpha = self.global_alpha + alpha_b

        itemNumPerTick = items.shape[1]

        # dts = [time_interval[:, i, :] for i in itemNumPerTick]

        w1s = self.w1(items).reshape(-1, itemNumPerTick, 1, Layer1)
        b1s = self.b1(items).reshape(-1, itemNumPerTick, 1, Layer1)
        w2s = self.w2(items).reshape(-1, itemNumPerTick, Layer1, Layer2)
        b2s = self.b2(items).reshape(-1, itemNumPerTick, 1, Layer2)
        w3s = self.w3(items).reshape(-1, itemNumPerTick, Layer2, Layer3)
        b3s = self.b3(items).reshape(-1, itemNumPerTick, 1, Layer3)
        w4s = self.w4(items).reshape(-1, itemNumPerTick, Layer3, 1)
        b4s = self.b4(items).reshape(-1, itemNumPerTick, 1, 1)

        gamma = self.af(torch.matmul(time_interval[..., None], w1s) + b1s)
        gamma = self.af(torch.matmul(gamma, w2s) + b2s)
        gamma = self.af(torch.matmul(gamma, w3s) + b3s)
        gamma = self.af(torch.matmul(gamma, w4s) + b4s)

        gamma = gamma.reshape(time_interval.shape)

        excit = ((alpha * gamma) * mask).sum(-1)
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
