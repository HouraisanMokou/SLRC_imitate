import os
import sys
import time
import argparse
import numpy as np
import joblib

"""
to control the program,
control the paths and start to train or test
shell (*.sh) in '../script' would run this file  
"""


def get_args():
    """
    get arguments from sys
    :return:
    """

    # arguments for IO/ controlling
    parser = argparse.ArgumentParser(description='Run the models')
    parser.add_argument('gpu', type=str, default='0', help='set CUDA_VISIBLE_DEVICES')

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
                        type=str, default='../result', help='the directory the log would be saved to')
    parser.add_argument('--random_state', type=int, default=2021, help='the random seed')

    # arguments for reader
    parser.add_argument('sep',default='\t',help='sep of data set file')

    # arguments for runner
    parser.add_argument('--epoch', type=int, default=200,
                        help='number of epochs')
    parser.add_argument('--test_epoch', type=int, default=-1,
                        help='test with a period of some epoch (-1 means no test)')
    parser.add_argument('--stop',type=int,default=5,help='stop cnt when accuracy down continuously')
    parser.add_argument('--lr', type=float, default=1e-3,help='learning rate')
    parser.add_argument('--l2', type=float, default=0,help='l2 regularization in optimizer')
    parser.add_argument('--batch_size', type=int, default=256,help='batch size while training/ validating')
    parser.add_argument('--eval_batch_size', type=int, default=256,help='batch size while testing')

    args=parser.parse_known_args()
    return args


def make_corpus(args):
    """
    to make corpus by calling the common.reader
    if load_corpus is True, load corpus from path
    otherwise, load data from data directory and save to corpus path
    :param args: arguments.
    relate with: corpus_path, load_corpus, sep, data_directory, dataset_name
    :return:
    """
    pass

def train(args):
    """
    call the runner to set the weights (initial or from local model), build the model,
    move the model and data to gpu, and then start train, finally save the model
    :param args: arguments.
    relate with: gpu, save_directory, load_model, model_name, epoch,
                 test_epoch, stop, lr, l2, batch_size
    :return:
    """
    pass

def test(args):
    """
    call the runner to set the weights from local model, build the model,
    move the model and data to gpu, and then start test, logging the result
    :param args: arguments.
    relate with: gpu, save_directory, load_model, model_name, epoch,
                 test_epoch, stop, lr, l2, eval_batch_size
    :return:
    """
    pass

def main(args):
    """
    The main controlling method. Set random seed, prepare hardware and logging, read data and then make corpus.
    Finally, start train or test.
    :param args: the arguments gets from get_args
    :return: no return
    """
    pass


if __name__ == '__main__':
    args=get_args()
    main(args)
