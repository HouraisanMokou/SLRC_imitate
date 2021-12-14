import random
import time

import numpy as np

from common.constants import *
from tqdm import tqdm
import os
import pandas as pd

from multiprocessing import Pool, RLock, freeze_support

root = "../data/data_order/order.txt"
target = "../data/test_order4"

neg_len = 99

ratioTrain = 0.75
ratioTest = 0.05


def process_order_files(src: str):
    """
    process the src file into data in 3 files: train.csv, test.csv, val.csv
    the formats of each files:

    train.csv: user_id \t item_id \t time
    test.csv: user_id \t item_id \t time \t neg_items
    val.csv: user_id \t item_id \t time \t neg_items

    neg_items is a list of random items not include ground truth items (amount 20, for example)

    as for train, the negative items would be get by reader and be picked by dataset each epoch
    as for test and val the negative items is would use list here.
    This is to because the meaning of negative item is little different in train and test or val


    :param src: the file to process
    :return: no turn
    """

    _time = time.time()
    df = pd.read_csv(src, header=None, sep="\t")

    use_count = df[0].value_counts()

    df = df[df[0].isin(use_count[use_count > (2 * use_count.std())].index)]
    # NOTE: use_count > (2 * use_count.std()) might not make sense.

    item_count = df[1].value_counts()
    df = df[df[1].isin(item_count[item_count > 2].index)]

    df = df.reset_index(drop=True)
    df = df.head(820000)
    # print(len(df))
    # print(df[0].value_counts())
    # print(df[1].value_counts())

    # idx = df[[0, 1]].value_counts()
    # idx = set(idx[idx > 1].index)

    clicksUsers = []
    clicksItems = []
    clicksTimes = []

    # tmp_clicks = list()
    for c_idx, click in tqdm(
        enumerate(zip(df[0], df[1], df[3])),
        leave=False,
        desc="first pass",
        ncols=100,
        mininterval=0.1,
    ):
        u, i, t = click

        clicksUsers.append(u)
        clicksItems.append(i)
        clicksTimes.append(t)

        # tmp_clicks.append(click)

        # if (u, i) in idx:
        #     tmp_clicks.append(click)

    # change the index of u and i

    del df, use_count, item_count

    clicks = []
    Users = list(set(clicksUsers))
    n_user = len(Users)
    Users = dict(zip(Users, range(n_user)))
    Items = list(set(clicksItems))
    n_item = len(Items)
    Items = dict(zip(Items, range(n_item)))

    # users = dict()
    # items = dict()
    # n_user = 0
    # n_item = 0

    user_purchase_dict = [dict() for _ in range(n_user)]

    for click in tqdm(
        zip(clicksUsers, clicksItems, clicksTimes),
        leave=False,
        desc="second pass",
        ncols=100,
        mininterval=0.1,
    ):
        u, i, t = click
        uid = Users[u]
        iid = Items[i]
        clicks.append((uid, iid, t))

        if t not in user_purchase_dict[uid].keys():
            user_purchase_dict[uid][t] = set()

        user_purchase_dict[uid][t].add(iid)

    # for c_idx, click in tqdm(
    #     enumerate(tmp_clicks),
    #     leave=False,
    #     desc="second pass",
    #     total=len(df),
    #     ncols=100,
    #     mininterval=0.1,
    # ):
    #     u, i, t = click
    #     if u not in users:
    #         users[u] = n_user
    #         n_user += 1
    #     uid = users[u]
    #     if i not in items:
    #         items[i] = n_item
    #         n_item += 1
    #     iid = items[i]
    #     clicks.append((uid, iid, t))
    #     if uid not in user_purchase_dict.keys():
    #         user_purchase_dict[uid] = dict()
    #     if t not in user_purchase_dict[uid].keys():
    #         user_purchase_dict[uid][t] = set()
    #     user_purchase_dict[uid][t].add(iid)

    clicks = np.array(clicks)
    random.shuffle(clicks)
    train_size = int(np.round(len(clicks) * 0.75))
    test_size = int(np.round(len(clicks) * 0.05))
    # print(clicks)

    dataset = {
        "train": clicks[:train_size],
        "test": clicks[train_size : train_size + test_size],
        "val": clicks[train_size + test_size :],
    }

    dfs = {
        "train": pd.DataFrame(dataset["train"], columns=["user_id", "item_id", "time"]),
        "test": pd.DataFrame(dataset["test"], columns=["user_id", "item_id", "time"]),
        "val": pd.DataFrame(dataset["val"], columns=["user_id", "item_id", "time"]),
    }

    if not os.path.exists(target):
        os.makedirs(target)
    # else:
    #     raise

    for k in dfs.keys():
        df = dfs[k]
        file = "{}/{}.csv".format(target, k)
        if k == "train":
            df.to_csv(file, sep="\t", index=False)
        else:
            neg_items = shuffle_neg(dataset[k], n_item, user_purchase_dict)
            df["neg_items"] = neg_items
            df.to_csv(file, sep="\t", index=False)

    _time = time.time() - _time
    print("time used: {}".format(_time))


# def parallel_func(args):
#     click, total_pool, user_purchase_dict = args
#     neg = list(total_pool.difference(user_purchase_dict[click[0]][click[2]]))
#     # neg = np.random.choice(neg, replace=False, size=neg_len)
#     np.random.shuffle(neg)
#     neg = neg[:neg_len]


def shuffle_neg(dataset: list, n_item: int, user_purchase_dict: dict) -> list:
    """
    :param df:
    :return:
    """
    negs = list()
    total_pool = set(range(n_item))

    args = [(click, total_pool, user_purchase_dict) for click in dataset]

    # with Pool(4) as p:
    #     negs = p.map(parallel_func, args)

    for click in tqdm(
        dataset,
        leave=False,
        desc="find neg",
        ncols=100,
        total=len(dataset),
        mininterval=0.01,
    ):
        # pool=total_pool.difference(user_purchase_dict[click[0]][click[2]])
        #
        # neg=list(neg)

        neg = list(total_pool.difference(user_purchase_dict[click[0]][click[2]]))
        # neg = np.random.choice(neg, replace=False, size=neg_len)
        np.random.shuffle(neg)
        neg = neg[:neg_len]

        # for idx in range(neg_len):
        #     neg_item=random.randint(0,n_item-1)
        #     while neg_item not in user_purchase_dict[click[0]]:
        #         neg_item = random.randint(0, n_item - 1)
        #     neg.append(neg_item)
        negs.append(neg)

    return negs


if __name__ == "__main__":
    process_order_files(root)
    # df=pd.DataFrame([str([1,2,3]).replace(' ',''),str([1,2,3]).replace(' ','')])
    # print(df)
