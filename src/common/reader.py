import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
import copy
import logging

from typing import NoReturn, List
import pickle
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

from tqdm import tqdm


class Reader():
    """
    read the data and make corpus. turn all data to 3 dataset ('train', 'val', 'test')
    """

    def __init__(self, args):
        self.load_corpus = args.load_corpus
        self.corpus_path = os.path.join(args.corpus_directory, args.dataset_name, args.dataset_name + '.pkl')
        self.dataset_name = args.dataset_name
        self.data_prefix = os.path.join(args.data_directory, args.dataset_name)
        self.sep = args.sep
        self.time_scale = args.time_scale

        self.logger=args.logger

        if not self.load_corpus:
            self._read_data()
            self._save_corpus()
        else:
            self._load_corpus()

    def _read_data(self) -> NoReturn:
        """
        read data from original 4 data files, statics the following things

        self.df_dict=the dict of dict, key is ('train', 'val', 'test'), value is of each file
        self.n_users: int of the total number of users
        self.n_items: int of the total number of items
        self.items_per_user: a list, [u_id]th line is all the items brought by user

        the format of each dataframe:
        columns: ['user', 'item', 'time', 'time_interval']
        'time_interval' is a list, present the time gap of reconsumption

        :return:
        """
        self.df_dict = dict()
        self.n_users = 0
        self.n_items = 0
        self.clicks = set()

        for k in ['train', 'test', 'val']:
            df = pd.read_csv(os.path.join(self.data_prefix, '{}.csv'.format(k)), sep=self.sep)

            self.n_users = max(self.n_users, df['user_id'].max() + 1)
            self.n_items = max(self.n_items, df['item_id'].max() + 1)
            df['time'] = df['time'] / self.time_scale
            for click in zip(df['user_id'], df['item_id'], df['time']):
                self.clicks.add(click)
            self.df_dict[k] = df


        self.logger.info('collecting user history')
        self.items_per_user = [set() for _ in range(self.n_users)]
        self.user_his = [dict() for _ in range(self.n_users)]
        for c in self.clicks:
            self.items_per_user[c[0]].add(c[1])
            if c[1] not in self.user_his[c[0]].keys():
                self.user_his[c[0]][c[1]] = list()
            self.user_his[c[0]][c[1]].append(c[2])
        self.dataset_dict = dict()

        self.logger.info('calculating time intervals')
        self.time_max_length = 0
        for k in ['train', 'test', 'val']:
            df = self.df_dict[k]
            time_interval = [list() for _ in range(len(df))]

            for idx,rec in enumerate(zip(df['user_id'], df['item_id'], df['time'])):
                tmp_his = copy.deepcopy(self.user_his[rec[0]][rec[1]])
                tmp_his = rec[2] - np.array(tmp_his)
                for i in range(len(tmp_his)):
                    if tmp_his[i] > 0:
                        time_interval[idx].append(tmp_his[i])
                self.time_max_length=max(self.time_max_length,len(time_interval[idx]))
            for ti in time_interval:
                while len(ti)<self.time_max_length:
                    ti.append(0)

            df['time_interval'] = time_interval
            if k == 'train':
                tmp = df.to_dict()
                tmp['neg'] = None
                self.dataset_dict[k] = tmp
            else:
                self.dataset_dict[k] = df.to_dict()
                for keys in self.dataset_dict[k].keys():
                    if type(self.dataset_dict[k][keys][0])==str:
                        for _ in range(len(self.dataset_dict[k][keys])):
                            self.dataset_dict[k][keys][_] = eval(self.dataset_dict[k][keys][_])

    def _load_corpus(self) -> NoReturn:
        """
        load corpus from local corpus
        the format is shown before
        can implement this method later
        :return:
        """

    def _save_corpus(self) -> NoReturn:
        """
        save corpus from local corpus
        :return:
        """
        info_dict = dict()
        info_dict['df_dict'] = self.df_dict
        info_dict['n_users'] = self.n_users
        info_dict['n_items'] = self.n_items
        info_dict['clicks'] = self.clicks
        info_dict['items_per_user'] = self.items_per_user
        info_dict['user_his'] = self.user_his
        info_dict['dataset_dict'] = self.dataset_dict
        f = open(self.corpus_path, 'wb')
        pickle.dump(info_dict, f)
        f.close()





# if __name__ == '__main__':
#     class args:
#         load_corpus = False
#         corpus_directory = '../../data'
#         dataset_name = 'debug'
#         data_directory = '../../data'
#         sep = '\t'
#         time_scale = 1#3600 * 7 * 24 * 10
#
#
#     c = Reader(args)
#     d = Dataset(corpus=c, model=None, phase='train')
#     f1 = d.__getitem__(0)
#     f2 = d.__getitem__(1)
#     f3 = d.__getitem__(2)
#     dd = d._collate_batch([f1, f2,f3])
#     print(dd)

