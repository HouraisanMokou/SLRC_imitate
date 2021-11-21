import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
import copy

from torch.utils.data import Dataset as base_dataset
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

        self.items_per_user = [set() for _ in range(self.n_users)]
        self.user_his = [dict() for _ in range(self.n_users)]
        for c in self.clicks:
            self.items_per_user[c[0]].add(c[1])
            if c[1] not in self.user_his[c[0]].keys():
                self.user_his[c[0]][c[1]] = list()
            self.user_his[c[0]][c[1]].append(c[2])
        self.dataset_dict = dict()
        for k in ['train', 'test', 'val']:
            df = self.df_dict[k]
            time_interval = [list() for _ in range(len(df))]
            for idx,rec in enumerate(zip(df['user_id'], df['item_id'], df['time'])):
                tmp_his = copy.deepcopy(self.user_his[rec[0]][rec[1]])
                tmp_his = rec[2] - np.array(tmp_his)
                for i in range(len(tmp_his)):
                    if tmp_his[i] > 0:
                        time_interval[idx].append(tmp_his[i])
            for ti in time_interval:
                ti.sort()
            df['time_interval'] = time_interval
            if k == 'train':
                tmp = df.to_dict()
                tmp['neg'] = None
                self.dataset_dict[k] = tmp
            else:
                self.dataset_dict[k] = df.to_dict()

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


class Dataset(base_dataset):
    """
    the dataset class, let dataloader to load
    the format:

    {'users':list of int,'items':list of int,'neg':list,'time':list of pad_sequence }
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
        if self.data_dict['neg'] is None:
            self.shuffle_neg()
        n = self.data_dict['neg'][index]
        ti = self.data_dict['time_interval'][index]
        n, ti = np.array(n), np.array(ti)
        feed_dict = {
            'user_id': u,
            'item_id': i,
            'neg': n,
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
        data=[(d['user_id'],d['item_id'],d['neg'],d['time_interval'],len(d['time_interval']))for d in feed_dict_list]
        data=list(zip(*data))
        users = np.array(data[0])
        items = np.array(data[1])
        negs = np.array(data[2])
        time_intervals = data[3]
        lengths = data[4]
        lengths = torch.from_numpy(np.array(lengths))
        time_intervals=[torch.tensor(ti).float() for ti in time_intervals]
        p=pad_sequence(time_intervals,batch_first=True)

        feed_dict = {
            'user_id': torch.from_numpy(users),
            'item_id': torch.from_numpy(items),
            'negs': torch.from_numpy(negs),
            'time_intervals': p
        }
        return feed_dict

    def shuffle_neg(self) -> NoReturn:
        """
        do something before each epoch
        prepare neg_item list for the epoch
        :return:
        """

        neg = np.zeros(len(self))
        for i in range(len(self)):
            tmp = np.random.randint(0, self.corpus.n_items)
            while tmp in self.corpus.user_his[self.data_dict['user_id'][i]].keys():
                tmp = np.random.randint(0, self.corpus.n_items)
            neg[i] = tmp
        self.data_dict['neg'] = neg


if __name__ == '__main__':
    class args:
        load_corpus = False
        corpus_directory = '../../data'
        dataset_name = 'debug'
        data_directory = '../../data'
        sep = '\t'
        time_scale = 1#3600 * 7 * 24 * 10


    c = Reader(args)
    d = Dataset(corpus=c, model=None, phase='train')
    f1 = d.__getitem__(0)
    f2 = d.__getitem__(1)
    f3 = d.__getitem__(2)
    dd = d._collate_batch([f1, f2,f3])
    print(dd)

