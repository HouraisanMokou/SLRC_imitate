import torch
import logging
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.distributions as distributions
import torch.nn.functional as F
from torch.utils.data import Dataset as BaseDataset
from torch.nn.utils.rnn import pad_sequence
from typing import NoReturn, List

import os

from src.common.utils import *
from src.common.reader import Reader
from src.models.SLRC import SLRC


class SLRC_BPR(SLRC):
    def _init_hawks_weights(self):
        self.global_alpha = nn.Parameter(torch.tensor(0.))
        self.alpha = nn.Embedding(self.item_num, 1)
        self.pi = nn.Embedding(self.item_num, 1)
        self.beta = nn.Embedding(self.item_num, 1)
        self.mu = nn.Embedding(self.item_num, 1)
        self.sigma = nn.Embedding(self.item_num, 1)

    def _init_cf_weights(self) -> NoReturn:
        self.user_embed = nn.Embedding(self.user_num, self.emb_size)
        self.item_embed = nn.Embedding(self.item_num, self.emb_size)

    def _forward_excit(self, feed_dict):
        """
        :param feed_dict:
        :return:
        """

    def _forward_CF(self, feed_dict):
        users = feed_dict['users']
        items = feed_dict['items']
        u_vector = self.user_embed(users)
        i_vector = self.item_embed(items)

        base = (u_vector[:, :, None] * i_vector).sum(-1)  # the dim of users and items is not the same
        return base
