import os

import numpy as np
import pandas as pd
from torch import nn

from torch.utils.data import dataset as base_dataset
from typing import NoReturn,List

from src.models.SLRC import SLRC


class Reader():
    """
    read the data and make corpus. turn all data to 3 dataset ('train', 'val', 'test')
    """

    def __init__(self, args):
        self.load_corpus = args.load_corpus
        self.corpus_path = os.path.join(args.corpus_directory, args.dataset_name + '.pkl')
        self.dataset_name = args.dataset_name
        self.sep = args.sep

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
        self.n_items=None
        self.n_users=None
        self.df_dict=None
        self.items_per_user=None

    def _load_corpus(self) -> NoReturn:
        """
        load corpus from local corpus
        the format is shown before
        :return:
        """

    def _save_corpus(self) -> NoReturn:
        """
        save corpus from local corpus
        :return:
        """


class Dataset(base_dataset):
    """
    the dataset class, let dataloader to load
    the format:

    {'users':list of int,'items':list of int,'neg':list,'time':list of pad_sequence }
    notice that the length of neg is different in train test and val
    in train is 1
    in test or val is longer
    """

    def __init__(self,corpus:Reader,model:SLRC):
        """
        :param corpus:
        :param model:
        """
        self.corpus=corpus
        self.model=model

        self.data_dict=dict()

    def __len__(self):
        """
        overwrite the len()
        :return:
        """

    def __getitem__(self, index: int) -> dict:
        """
        overwrite the getitem()
        :param index:
        :return:
        """

    def _get_feed_dict(self, index: int) -> dict:
        """
        get ith feed_dict
        :param index:
        :return:
        """

    def _collate_batch(self,feed_dict_list: List[dict])->dict:
        """
        to turn several feed_dict to a batched one
        the dataloader would use this
        :param feed_dict_list:
        :return:
        """

    def actions_before_epoch(self) -> NoReturn:
        """
        do something before each epoch
        prepare neg_item list for the epoch
        :return:
        """
        neg=np.zeros(self.corpus.n_items)
        for i in range(self.corpus.n_items):
            tmp = np.random.random(self.corpus.n_items)
            while tmp not in self.corpus.items_per_user[i]:
                tmp = np.random.random(self.corpus.n_items)
            neg[i]=tmp
        self.data_dict['neg']=neg