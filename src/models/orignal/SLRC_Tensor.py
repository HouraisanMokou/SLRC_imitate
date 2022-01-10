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
from torch.distributions import normal, exponential
from typing import NoReturn, List

import os

from common.utils import *
from common.reader import Reader
from models.SLRC import SLRC


class SLRC_Tensor(SLRC):
    def _init_hawks_weights(self):
        self.global_alpha = nn.Parameter(torch.tensor(1.))
        self.alpha = nn.Embedding(self.item_num, 1)
        self.pi = nn.Embedding(self.item_num, 1)
        self.beta = nn.Embedding(self.item_num, 1)
        self.mu = nn.Embedding(self.item_num, 1)
        self.sigma = nn.Embedding(self.item_num, 1)

    def _init_cf_weights(self) -> NoReturn:
        self.bin_num=100
        self.user_embed = nn.Embedding(self.user_num, self.emb_size)
        self.item_embed = nn.Embedding(self.item_num, self.emb_size)
        self.item_time_embed = nn.Embedding(self.bin_num, self.emb_size)
        self.user_time_embed = nn.Embedding(self.bin_num, self.emb_size)

    def _forward_excit(self, feed_dict):
        """
        :param feed_dict:
        :return:
        """
        items = feed_dict['item_id']  # the 1st is tested and other is neg
        time_interval = feed_dict['time_intervals']
        mask=(time_interval>=0).float()

        alpha_b = self.alpha(items)
        alpha = self.global_alpha + alpha_b
        beta = (self.beta(items) + 1).clamp(min=1e-10,max=10)
        pi = self.pi(items) + 0.5
        mu = self.mu(items) + 1  # may fix later
        sigma = (self.sigma(items) + 1).clamp(min=1e-10,max=10)
        exp=exponential.Exponential(beta,validate_args=False)
        norm=normal.Normal(mu,sigma,validate_args=False)

        dt=(time_interval*mask)
        gamma1=pi*(exp.log_prob(dt).exp())
        gamma2=(1-pi)*(norm.log_prob(dt).exp())
        gamma=gamma1+gamma2
        excit= (gamma*alpha*mask).sum(-1)
        return excit

    def _forward_CF(self, feed_dict):
        users = feed_dict['user_id']
        items = feed_dict['item_id']
        time = feed_dict['time_bin']
        u_vector = self.user_embed(users)
        i_vector = self.item_embed(items)
        u_t_vector = self.user_time_embed(time)
        i_t_vector = self.item_time_embed(time)

        s1=(u_vector[:, None, :] * i_vector).sum(-1)
        t2 = (u_vector * u_t_vector).sum(-1)
        s3 = (i_vector * i_t_vector[:,None,:]).sum(-1)
        s2=t2[:,None].repeat(1,s3.size()[1])


        base =s1+s2+s3 # the dim of users and items is not the same
        return base

    class Dataset(SLRC.Dataset):
        def __init__(self, corpus: Reader, model: nn.Module, phase):
            super().__init__(corpus, model, phase)
            self.bin_num=model.bin_num
            self.bin_width=float(self.corpus.max_time-self.corpus.min_time+1)/self.bin_num

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
            time=self.data_dict['time'][index]
            # if type(n)==str:
            #     n = eval(n)
            tmp=copy.deepcopy(n)
            tmp.insert(0,i)
            i=tmp

            # t=[ti]
            # # for ni in n:
            # #     t.append([0 for _ in range(len(ti))])
            # ti=t
            i, n, ti =np.array(i), np.array(n), np.array(ti)
            feed_dict = {
                'user_id': u,
                'item_id': i,
                # 'neg_items': n,
                'time_interval': ti,
                'time_bin':(time-self.corpus.min_time)//self.bin_width
            }
            return feed_dict

        def _collate_batch(self, feed_dict_list: List[dict]) -> dict:
            """
            to turn several feed_dict to a batched one
            the dataloader would use this
            :param feed_dict_list:
            :return:
            """
            data = [(d['user_id'], d['item_id'], d['time_interval'],d['time_bin']) for d in
                    feed_dict_list]
            data = list(zip(*data))
            users = np.array(data[0])
            items = np.array(data[1])
            time_intervals = np.array(data[2])
            time=np.array(data[3]).astype(int)
            time_intervals = torch.from_numpy(time_intervals)
            p = time_intervals
            feed_dict = {
                'user_id': torch.from_numpy(users),
                'item_id': torch.from_numpy(items),
                'time_intervals': p,
                'time_bin':torch.from_numpy(time)
            }
            return feed_dict