import os
import sys
import time
import argparse
import numpy as np
import joblib

"""
to control the program,
control the paths and start to train or test
shell in '../script' would run this file  
"""

def get_args():
    parser = argparse.ArgumentParser(description='Run the models')
    parser.add_argument('gpu',type=str,default='0',help='the gpu used')

    parser.add_argument('--data_directory', type=str, default='../data', help='original data directory')
    parser.add_argument('--save_directory', type=str, default='../result', help='Save data directory')

    parser.add_argument('--load_model', type=bool, default=False, help='whether to load model already exist')
    parser.add_argument('--model_name', type=str, default='Grocery_and_Gourmet_Food', help='the name of model')
    parser.add_argument('--dataset_name', type=str, default='Grocery_and_Gourmet_Food', help='the name of data set')
    # the path of data set should be data_directory/dataset_name
    # the model would be saved to save_directory/model_name

    parser.add_argument('--corpus_path', type=str, default='../data', help='the path of corpus')
    parser.add_argument('--load_corpus', type=bool, default=False, help='whether to load corpus already exist')
    # the path of corpus should be corpus_path/dataset_name+'.pkl'
    # if load_corpus is False the path is the save path
    # else, the path is path to load

    parser.add_argument('--stage', type=str, default='train', help='train/test')
    parser.add_argument('--logging_directory',
                        type=str, default='../result',help='the directory the log would be saved to')
    parser.add_argument('--random_state',type=int,default=2021,help='the random seed')

if __name__=='__main__':
    get_args()