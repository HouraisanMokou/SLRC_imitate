import copy

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

        self.emb_size = args.emb_size
        self.time_scale = args.time_scale
        self.user_num = corpus.n_users
        self.item_num = corpus.n_items

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model_path = os.path.join(args.save_directory, args.model_name, args.model_file_name)
        self.optimizer = None

        self.logger = args.logger

        super(SLRC, self).__init__()

        self._define_weights()
        self.total_param = count_var(self)

    def _define_weights(self) -> NoReturn:
        """
        the method to init weights
        :return:
        """
        self._init_cf_weights()
        self._init_hawks_weights()
        self.logger.info('parameters is inited')

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
        base = self._forward_CF(feed_dict)
        excitation = self._forward_excit(feed_dict)
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

    def loss(self, out_dict: dict) -> torch.Tensor:
        """
        loss function
        :return:
        """
        predictions = out_dict['prediction']
        pos = predictions[:, 0]
        neg = predictions[:, 1]
        loss = -(pos - neg).sigmoid().log().mean()
        return loss

    def classify_params(self):
        """
        since l2 should not effect on the bias and weight decay effect on the all parameters,
        need to divided weights and bias
        :return:
        """
        result = [{'params': list()}, {'params': list(), 'weight_decay': 0}]
        for name, param in self.named_parameters():
            if param.requires_grad:
                if 'bias' in name:
                    result[1]['params'].append(param)
                else:
                    result[0]['params'].append(param)
        return result


class Dataset(BaseDataset):
    """
    the dataset class, let dataloader to load
    the format:

    {'users':list of int,'items':list of int,'neg_items':list,'time':list of pad_sequence }
    notice that the length of neg is different in train test and val
    in train is 1
    in test or val is longer
    """

    def __init__(self, corpus: Reader, model: nn.Module, phase):
        """
        :param corpus:
        :param model:
        """
        self.corpus = corpus
        self.model = model

        self.phase = phase

        self.data_dict = corpus.dataset_dict[phase]

    def __len__(self):
        """
        overwrite the len()
        :return:
        """
        return len(self.data_dict['user_id'])

    def __getitem__(self, index: int) -> dict:
        """
        overwrite the getitem()
        :param index:
        :return:
        """
        u = self.data_dict['user_id'][index]
        i = self.data_dict['item_id'][index]
        if self.data_dict['neg_items'] is None:
            self.shuffle_neg()
        n = self.data_dict['neg_items'][index]
        ti = self.data_dict['time_interval'][index]
        # if type(n)==str:
        #     n = eval(n)
        tmp=copy.deepcopy(n)
        tmp.insert(0,i)
        i=tmp

        t=[ti]
        for ni in n:
            t.append([0 for _ in range(len(ti))])
        ti=t
        i, n, ti =np.array(i), np.array(n), np.array(ti)
        feed_dict = {
            'user_id': u,
            'item_id': i,
            # 'neg_items': n,
            'time_interval': ti
        }
        return feed_dict

    def _collate_batch(self, feed_dict_list: List[dict]) -> dict:
        """
        to turn several feed_dict to a batched one
        the dataloader would use this
        :param feed_dict_list:
        :return:
        """
        data = [(d['user_id'], d['item_id'], d['time_interval']) for d in
                feed_dict_list]
        data = list(zip(*data))
        users = np.array(data[0])
        items = np.array(data[1])
        time_intervals = data[2]
        time_intervals = torch.tensor(time_intervals)
        p = time_intervals
        feed_dict = {
            'user_id': torch.from_numpy(users),
            'item_id': torch.from_numpy(items),
            'time_intervals': p
        }
        return feed_dict

    def shuffle_neg(self) -> NoReturn:
        """
        do something before each epoch
        prepare neg_item list for the epoch
        :return:
        """

        neg = [list()for _ in range(len(self))]
        for i in range(len(self)):
            tmp = np.random.randint(0, self.corpus.n_items)
            while tmp in self.corpus.user_his[self.data_dict['user_id'][i]].keys():
                tmp = np.random.randint(0, self.corpus.n_items)
            neg[i].append(tmp)
        self.data_dict['neg_items'] = neg
