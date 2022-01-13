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


class MLP_BPR_2(SLRC):
    def _init_hawks_weights(self):
        self.global_alpha = nn.Parameter(torch.tensor(1.0))
        self.alpha = nn.Embedding(self.item_num, 1)
        self.fci1 = nn.Linear(self.emb_size, 64)
        self.fci2 = nn.Linear(64, 32)
        self.fci3 = nn.Linear(32, 16)
        self.fcu1 = nn.Linear(self.emb_size, 64)
        self.fcu2 = nn.Linear(64, 32)
        self.fcu3 = nn.Linear(32, 16)
        self.fct1 = nn.Linear(1, 16)
        self.fct2 = nn.Linear(16, 32)
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 1)

        self.af = nn.LeakyReLU(5e-3)
        # self.pi = nn.Embedding(self.item_num, 1)
        # self.beta = nn.Embedding(self.item_num, 1)
        # self.mu = nn.Embedding(self.item_num, 1)
        # self.sigma = nn.Embedding(self.item_num, 1)

    def _init_cf_weights(self) -> NoReturn:
        self.user_embed = nn.Embedding(self.user_num, self.emb_size)
        self.item_embed = nn.Embedding(self.item_num, self.emb_size)

    def _forward_excit(self, feed_dict):
        """
        :param feed_dict:
        :return:
        """
        users = feed_dict["user_id"]
        items = feed_dict["item_id"]

        itemNumPerTick = items.shape[1]

        u_vector = self.user_embed(users)
        i_vector = self.item_embed(items)

        time_interval = (feed_dict["time_intervals"] / TIME_SCALE).float()

        mask = (time_interval > 0).float()

        alpha_b = self.alpha(items)
        alpha = self.global_alpha + alpha_b

        userFeature = self.af(self.fcu1(u_vector))
        userFeature = self.af(self.fcu2(userFeature))
        userFeature = self.af(self.fcu3(userFeature))

        userFeature = userFeature.reshape(-1, 1, 16).contiguous()
        userFeature = userFeature.repeat([1, itemNumPerTick, 1])

        itemFeature = i_vector.reshape(-1, self.emb_size).contiguous()

        itemFeature = self.af(self.fci1(itemFeature))
        itemFeature = self.af(self.fci2(itemFeature))
        itemFeature = self.af(self.fci3(itemFeature))

        itemFeature = itemFeature.reshape(-1, itemNumPerTick, 16).contiguous()

        Features = torch.cat(
            [userFeature, itemFeature], dim=2
        ).contiguous()  # batch_size * itemNumPerTick * 32

        Features = Features.reshape(-1, itemNumPerTick, 1, 32).contiguous()
        Features = Features.repeat([1, 1, time_interval.shape[2], 1])

        timeFeature = time_interval.reshape(-1, 1)

        timeFeature = self.af(self.fct1(timeFeature))
        timeFeature = self.af(self.fct2(timeFeature))

        timeFeature = timeFeature.reshape(
            -1, itemNumPerTick, Features.shape[2], 32
        ).contiguous()

        Features = torch.cat([Features, timeFeature], dim=3).contiguous()

        Features = Features.reshape(-1, 32 + 32).contiguous()

        gamma = self.af(self.fc1(Features))
        gamma = self.af(self.fc2(gamma))
        gamma = self.af(self.fc3(gamma))

        gamma = gamma.reshape(-1, itemNumPerTick, timeFeature.shape[2]).contiguous()

        excit = ((alpha * gamma) * mask).sum(-1)
        return excit  # [batch_size, itemNumPerTick]

    def _forward_CF(self, feed_dict):
        users = feed_dict["user_id"]
        items = feed_dict["item_id"]
        u_vector = self.user_embed(users)
        i_vector = self.item_embed(items)

        base = (u_vector[:, None, :] * i_vector).sum(
            -1
        )  # the dim of users and items is not the same
        return base
