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


class SRC_BPR(SLRC):
    def _init_hawks_weights(self):
        self.global_alpha = nn.Parameter(torch.tensor(1.))
        self.alpha = nn.Embedding(self.item_num, 1)
        self.beta = nn.Embedding(self.item_num, 1)

    def _init_cf_weights(self) -> NoReturn:
        self.user_embed = nn.Embedding(self.user_num, self.emb_size)
        self.item_embed = nn.Embedding(self.item_num, self.emb_size)

    def _forward_excit(self, feed_dict):
        """
        :param feed_dict:
        :return:
        """
        items = feed_dict['item_id']  # the 1st is tested and other is neg
        time_interval = feed_dict['time_intervals']/TIME_SCALE
        mask=(time_interval>0).float()

        alpha_b = self.alpha(items)
        alpha = (self.global_alpha).clamp(min=1e-10,max=10) + alpha_b
        beta = (self.beta(items) + 1).clamp(min=1e-10,max=10)
        exp=exponential.Exponential(beta,validate_args=False)


        dt=time_interval
        gamma1=(exp.log_prob(dt).exp())
        excit1=alpha*gamma1
        excit2=excit1*mask
        excit= (excit2).sum(-1)
        return excit

    def _forward_CF(self, feed_dict):
        users = feed_dict['user_id']
        items = feed_dict['item_id']
        u_vector = self.user_embed(users)
        i_vector = self.item_embed(items)

        base = (u_vector[:, None, :] * i_vector).sum(-1)  # the dim of users and items is not the same
        return base
