from common.constants import *
import os
import pandas as pd

root = '../data/data_order/order.txt'


def process_files(src: str):
    """
    process the src file into data in 4 files: train.csv, test.csv, val.csv
    the formats of each files:

    train.csv: user_id \t item_id \t time
    test.csv: user_id \t gt_item_id \t time \t neg_items
    cal.csv: user_id \t gt_item_id \t time \t neg_items

    gt_item_id is the list of items user buy at this timestamp.
    neg_items is a list of random items not include ground truth items (amount 20, for example)

        pay attention that there maybe users buy different items at same timestamp

    as for train, the negative items would be get by reader and be picked by dataset each epoch
    as for test and val the negative items is would use list here.
    This is to because the meaning of negative item is little different in train and test or val


    :param src: the file to process
    :return: no turn
    """
    pass
