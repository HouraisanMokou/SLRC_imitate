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


class Reader:
    """
    read the data and make corpus. turn all data to 3 dataset ('train', 'val', 'test')
    """

    def __init__(self, args):
        self.load_corpus = args.load_corpus
        self.corpus_path = os.path.join(
            args.corpus_directory, args.dataset_name, args.dataset_name + ".pkl"
        )
        self.dataset_name = args.dataset_name
        self.data_prefix = os.path.join(args.data_directory, args.dataset_name)
        self.sep = args.sep
        self.time_scale = args.time_scale

        self.logger = args.logger

        self.known_test_when_trian = True
        self._read_data()
        self._save_corpus()

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
        self.set_clicks = {"train": set(), "test": set(), "val": set()}

        self.max_time = -np.inf
        self.min_time = np.inf
        for k in ["train", "test", "val"]:
            df = pd.read_csv(
                os.path.join(self.data_prefix, "{}.csv".format(k)), sep=self.sep
            )

            self.n_users = max(self.n_users, df["user_id"].max() + 1)
            self.n_items = max(self.n_items, df["item_id"].max() + 1)
            df["time"] = df["time"] / self.time_scale
            self.max_time = max(df["time"].max(), self.max_time)
            self.min_time = min(df["time"].min(), self.min_time)
            for click in zip(df["user_id"], df["item_id"], df["time"]):
                self.clicks.add(click)
                self.set_clicks[k].add(click)
            self.df_dict[k] = df

        self.time_max_length = 0
        self.logger.info("collecting user history")
        self.items_per_user = [set() for _ in range(self.n_users)]
        self.user_his = [dict() for _ in range(self.n_users)]
        self.user_his_set = {
            "train": [dict() for _ in range(self.n_users)],
            "test": [dict() for _ in range(self.n_users)],
            "val": [dict() for _ in range(self.n_users)],
        }
        for k in ["train", "test", "val"]:
            for c in self.set_clicks[k]:
                self.items_per_user[c[0]].add(c[1])
                if c[1] not in self.user_his_set[k][c[0]].keys():
                    self.user_his_set[k][c[0]][c[1]] = list()
                self.user_his_set[k][c[0]][c[1]].append(c[2])
        for c in self.clicks:
            self.items_per_user[c[0]].add(c[1])
            if c[1] not in self.user_his[c[0]].keys():
                self.user_his[c[0]][c[1]] = list()
            self.user_his[c[0]][c[1]].append(c[2])
            self.time_max_length = max(
                self.time_max_length, len(self.user_his[c[0]][c[1]])
            )
        self.dataset_dict = dict()

        self.logger.info("calculating time intervals")

        time_intervals = dict()
        self.max_consider_length = 10
        self.time_max_length = min(self.time_max_length, self.max_consider_length)
        for k in ["train", "test", "val"]:
            df = self.df_dict[k]
            time_interval = list()

            if k == "train":
                his = (
                    self.user_his
                    if self.known_test_when_trian
                    else self.user_his_set[k]
                )
                for idx, rec in tqdm(
                    enumerate(zip(df["user_id"], df["item_id"], df["time"])),
                    leave=False,
                    desc="time interval building for {}".format(k),
                    mininterval=0.1,
                    ncols=100,
                    total=len(df),
                ):
                    tmp_his = copy.deepcopy(his[rec[0]][rec[1]])
                    tmp_his = rec[2] - np.array(tmp_his)
                    tmp = tmp_his[tmp_his > 0]
                    tmp.sort()
                    tmp = tmp[: self.max_consider_length]
                    time_interval.append(tmp)
                for i in tqdm(
                    range(len(time_interval)),
                    leave=False,
                    desc="pad the time intervals",
                    mininterval=0.01,
                    ncols=100,
                    total=len(time_interval),
                ):
                    ti = time_interval[i]
                    time_interval[i] = [
                        np.pad(ti, (0, self.time_max_length - len(ti))),
                        np.zeros(self.time_max_length),
                    ]
            else:
                his = (
                    self.user_his
                    if self.known_test_when_trian
                    else self.user_his_set[k]
                )
                for idx, rec in tqdm(
                    enumerate(
                        zip(df["user_id"], df["item_id"], df["neg_items"], df["time"])
                    ),
                    leave=False,
                    desc="time interval building for {}".format(k),
                    mininterval=0.1,
                    ncols=100,
                    total=len(df),
                ):
                    tmp_his = copy.deepcopy(his[rec[0]][rec[1]])
                    tmp_his = rec[3] - np.array(tmp_his)
                    tmp = tmp_his[tmp_his > 0]
                    tmp.sort()
                    tmp = tmp[: self.max_consider_length]
                    ti = [tmp]
                    for i in eval(rec[2]):
                        if i not in self.user_his[rec[0]].keys():
                            ti.append([])
                        else:
                            tmp_his = copy.deepcopy(self.user_his[rec[0]][i])
                            tmp_his = rec[3] - np.array(tmp_his)
                            tmp = tmp_his[tmp_his > 0]
                            tmp.sort()
                            tmp = tmp[: self.max_consider_length]
                            ti.append(tmp)
                    time_interval.append(ti)
                for i in tqdm(
                    range(len(time_interval)),
                    desc="pad the time intervals in {} set".format(k),
                    leave=False,
                    mininterval=0.01,
                    ncols=100,
                    total=len(time_interval),
                ):
                    tis = time_interval[i]
                    for j in range(len(tis)):
                        ti = tis[j]
                        tis[j] = np.pad(ti, (0, self.time_max_length - len(ti))).astype(
                            np.float32
                        )
                    time_interval[i] = tis

            time_intervals[k] = time_interval

        self.logger.info("")
        for k in ["train", "test", "val"]:
            df = self.df_dict[k]
            df["time_interval"] = time_intervals[k]
            if k == "train":
                tmp = df.to_dict()
                tmp["neg_items"] = None
                self.dataset_dict[k] = tmp
            else:
                self.dataset_dict[k] = df.to_dict()
                for keys in self.dataset_dict[k].keys():
                    if type(self.dataset_dict[k][keys][0]) == str:
                        for _ in tqdm(
                            range(len(self.dataset_dict[k][keys])),
                            leave=False,
                            desc="convert data form in {} set".format(k),
                            mininterval=0.1,
                            ncols=100,
                            total=len(self.dataset_dict[k][keys]),
                        ):
                            self.dataset_dict[k][keys][_] = eval(
                                self.dataset_dict[k][keys][_]
                            )

    @staticmethod
    def _load_corpus(file_path):
        """
        load corpus from local corpus
        the format is shown before
        can implement this method later
        :return:
        """
        f = open(file_path, "rb")
        return pickle.load(f)

    def _save_corpus(self) -> NoReturn:
        """
        save corpus from local corpus
        :return:
        """
        f = open(self.corpus_path, "wb")
        pickle.dump(self, f)
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
