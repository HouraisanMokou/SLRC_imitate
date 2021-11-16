import os
import pandas as pd

from torch.utils.data import dataset as base_dataset
from typing import NoReturn,List


class Reader():
    """
    read the data and make corpus. turn all data to 3 dataset ('train', 'val', 'test')
    """

    def __init__(self, args):
        self.load_corpus = args.load_corpus
        self.corpus_path = os.path.join(args.corpus_directory, args.dataset_name + '.pkl')
        self.dataset_name = args.dataset_name
        self.sep = args.sep

        if self.load_corpus:
            self._read_data()
            self._save_corpus()
        else:
            self._load_corpus()


    def _read_data(self) -> NoReturn:
        """
        read data from original 4 data files, statics the following things

        self.df_dict=the dict of dataframe, key is ('train', 'val', 'test'), value is dataframe of each file
        self.n_users: int of the total number of users
        self.n_items: int of the total number of items
        self.neg_items: a matrix, [u_id]th line is the of neg_items list of [u_id]th user

        the format of each dataframe:
        columns: ['user', 'item', 'time', 'time_interval']
        'time_interval' is a list, present the time gap of reconsumption

        :return:
        """

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
    """

    def __init__(self):
        """"""

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