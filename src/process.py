from common.constants import *
import os
import pandas as pd

root = '../data/data_order/order.txt'


def process_files(src: str):
    """
    process the src file into data in 4 files: train.csv, test.csv, val.csv
    the formats of each files:

    train.csv: user_id \t item_id \t time
    test.csv: user_id \t item_id \t time \t neg_items
    cal.csv: user_id \t item_id \t time \t neg_items

    neg_items is a list of little part of random items this user never consume or in the candidate list of customer,
    but not buy, according to the original dataset

    as for train, the negative items would be static by reader and be picked by dataset each epoch
    as for test and val the negative items is would use list here.
    This is to because the meaning of negative item is little different in train and test or val

    :param src: the file to process
    :return: no turn
    """
    pass
