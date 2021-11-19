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


class SLRC(nn.Module):
    """
    the father class of SLRC,
    only contain the basic framework, the sons need to implement a few methods
    """

    @staticmethod
    def _weight_inits(m) -> NoReturn:
        """
        init weights after the model is built
        :param m: module
        :return:
        """
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0, std=0.01)
            if m.bias is not None:
                nn.init.normal_(m.bias, mean=0, std=0.01)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0, std=0.01)

    def __init__(self, args, corpus: reader):
        """
        :param args: arguments get in main.py
        :param corpus: reader, with information of data and statistics on data
        """
        self.max_history_length = args.max_history_length

        self.emb_size = args.emb_size
        self.time_scale = args.time_scale
        self.user_num = corpus.n_users
        self.item_num = corpus.n_items

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model_path = os.path.join(args.save_directory, args.model_name, args.model_file_name)
        self.optimizer = None

        self._define_weights()
        self.total_param = count_var(self)
        super(SLRC, self).__init__()

    def _define_weights(self) -> NoReturn:
        """
        the method to init weights
        :return:
        """
        self._init_cf_weights()
        self._init_hawks_weights()

    def _init_cf_weights(self) -> NoReturn:
        """
        init weights related with CF
        would overwrite by the son class of SLRC
        :return:
        """
        pass

    def _init_hawks_weights(self):
        """
        init weights related with Hawks
        may overwrite by the son class of SLRC
        :return:
        """
        pass

    def forward(self, feed_dict):
        """
        :return:
        """
        base = self.forward_excit(feed_dict)
        excitation = self.forward_excit(feed_dict)
        prediction = base + excitation
        return {'prediction': prediction}

    def _forward_excit(self, feed_dict):
        """
        :param feed_dict:
        :return: excitation base of lambda
        """
        return 0

    def _forward_CF(self, feed_dict):
        """
        :param feed_dict:
        :return: base of lambda
        """
        return 0

    def loss(self, out_dict: dict):
        """
        loss function
        :return:
        """
        predictions = out_dict['prediction']
        pos = predictions[:, 0]
        neg = predictions[:, 1]
        loss = -(pos - neg).sigmoid().log().mean()
        return loss
