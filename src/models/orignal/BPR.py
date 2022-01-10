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


class BPR(SLRC):
    def _init_hawks_weights(self):
        pass

    def _init_cf_weights(self) -> NoReturn:
        self.user_embed = nn.Embedding(self.user_num, self.emb_size)
        self.item_embed = nn.Embedding(self.item_num, self.emb_size)

    def _forward_excit(self, feed_dict):
        """
        :param feed_dict:
        :return:
        """
        return 0

    def _forward_CF(self, feed_dict):
        users = feed_dict['user_id']
        items = feed_dict['item_id']
        u_vector = self.user_embed(users)
        i_vector = self.item_embed(items)

        base = (u_vector[:, None, :] * i_vector).sum(-1)  # the dim of users and items is not the same
        return base
