from common.constants import *
import os
import pandas as pd

def process_files(src:str):
    """
    process the src file into data in four files: train.csv, test.csv, val.csv, item_meta.csv
    the formats of each files:

    train.csv: user_id \t item_id
    test.csv: user_id \t item_id \t neg_items
    dev.csv: user_id \t item_id \t neg_items
    item_meta.csv: user_id \t item_id \t reconsumption

    neg_items is a list of all items this user never consume
    reconsumption is a list, the value of ith index represents the time gap from i-1th to ith consumption

    :param src: the file to process
    :return: no turn
    """
root='../data\data_order\order.txt'


