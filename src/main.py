import logging
import os
import random
import sys
import time
import argparse
import numpy as np
import joblib
import torch.random

from src.common.runner import Runner
from src.common.reader import Reader
from src.models.SLRC import SLRC
from src.models.orignal.SLRC_BPR import SLRC_BPR
from src.models.orignal.SLRC_NCF import SLRC_NCF
from src.models.orignal.SLRC_Tensor import SLRC_Tensor
from src.models.orignal.BPR import BPR
from src.models.orignal.SRC_BPR import SRC_BPR
from src.models.orignal.LRC_BPR import LRC_BPR

"""
to control the program,
control the paths and start to train or test
shell (*.sh) in 'script' would run this file  
"""


def get_args():
    """
    get arguments from sys
    :return:
    """

    # arguments for IO/ controlling
    parser = argparse.ArgumentParser(description='Run the models')

    parser.add_argument('--gpu', type=str, default='0', help='set CUDA_VISIBLE_DEVICES')

    parser.add_argument('--data_directory', type=str, default='../data', help='original data directory')
    parser.add_argument('--save_directory', type=str, default='../../result', help='Save data directory')

    parser.add_argument('--load_model', type=bool, default=False, help='whether to load model already exist')
    parser.add_argument('--model_name', type=str, default='SLRC_BPR', help='the name of model')
    parser.add_argument('--dataset_name', type=str, default='Grocery_and_Gourmet_Food', help='the name of data set')
    # the path of data set should be data_directory/dataset_name
    # the model would be saved to save_directory/model_name

    parser.add_argument('--corpus_directory', type=str, default='../data', help='the path of directory of corpus')
    parser.add_argument('--load_corpus', type=bool, default=False, help='whether to load corpus already exist')
    # the name of corpus would be corpus_directory/dataset_name+'.pkl'
    # if load_corpus is False the path is the save path
    # else, the path is path to load

    parser.add_argument('--stage', type=str, default='train', help='train/test')
    parser.add_argument('--logging_directory',
                        type=str, default='../result', help='the directory the log would be saved to')
    parser.add_argument('--random_state', type=int, default=2021, help='the random seed')

    parser.add_argument('--logging_file_name', type=str, default='', help='the file name of logging')
    parser.add_argument('--model_file_name', type=str, default='', help='the file name of saved model')
    # the value would change to the format talked in main() if default

    # arguments for reader
    parser.add_argument('--sep', default='\t', help='sep of data set file')
    parser.add_argument('--time_scale', type=int, default=24 * 7 * 10, help='the time scale')

    # arguments for runner
    parser.add_argument('--epoch', type=int, default=200,
                        help='number of epochs')
    parser.add_argument('--test_epoch', type=int, default=-1,
                        help='test with a period of some epoch (-1 means no test)')
    parser.add_argument('--stop', type=int, default=5, help='stop cnt when accuracy down continuously')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--l2', type=float, default=0, help='l2 regularization in optimizer')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size while training/ validating')
    parser.add_argument('--eval_batch_size', type=int, default=256, help='batch size while testing')
    parser.add_argument('--emb_size', type=int, default=64, help='the length of embedding vector')
    parser.add_argument('--num_workers', type=int, default=0, help='workers of io used in loading data')
    parser.add_argument('--pin_memory', type=int, default=1, help='the length of embedding vector')
    args, unknown = parser.parse_known_args()
    return args


def run(args):
    """
    call the runner to set the weights (initial or from local model), build the model,
    move the model and data to gpu, and then start train or test, finally save the model
    :param args: arguments.
    relate with: gpu, save_directory, load_model, model_name, epoch, test_epoch,
                 stop, lr, l2, batch_size, eval_batch_size, max_history_length
    :return:
    """

    logger = args.logger
    logger.info('setting up runner')
    runner = Runner(args)
    logger.info('')

    if args.stage == 'train':
        logger.info('start to train')
        runner.train()
    else:
        if (args.load_model):
            logger.info('start to test')
            runner.evaluate(args.datasets['test'])
        else:
            logger.info('break down as for no model to load')


def main(args):
    """
    The main controlling method. Set random seed, prepare hardware and logging, read data and then make corpus.
    Finally, start train or test.
    the name of saved model and logging file has format if :
     [model_name]_[dataset_name]_[CF_method]_[lr]_[l2].pt
    :param args: the arguments gets from get_args
    :return: no return
    """
    # set random seed
    random.seed(args.random_state)
    np.random.seed(args.random_state)
    torch.manual_seed(args.random_state)
    torch.cuda.manual_seed(args.random_state)
    torch.backends.cudnn.deterministic = True

    # prepare hardware
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    # prepare logging
    if args.logging_file_name == '':
        args.logging_file_name = os.path.join(args.logging_directory, '{}@{}_{}.txt'.format(
            args.model_name, args.dataset_name, time.strftime('%Y.%m.%d', time.localtime())))
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)
    file_handler = logging.FileHandler(args.logging_file_name)
    file_handler.setLevel(logging.INFO)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console)
    logger.info('{}: start to logging\n'.format(time.strftime('%Y.%m.%d_%H:%M:%S', time.localtime())))
    setattr(args, 'logger', logger)

    # read data
    logger.info('reading data')
    if not args.load_corpus:
        reader = Reader(args)
    else:
        reader = Reader._load_corpus(os.path.join(args.corpus_directory, args.dataset_name, args.dataset_name + '.pkl'))
    logger.info('')

    # build model
    logger.info('building model')
    model_class = eval(args.model_name)
    model = model_class(args, reader)
    model.apply(model._weight_inits)
    setattr(args, 'model', model)
    logger.info(model)

    # build dataset
    logger.info('')
    datasets = dict()
    for k in ['train', 'test', 'val']:
        datasets[k] = model_class.Dataset(reader, model, k)
    setattr(args, 'datasets', datasets)

    run(args)


if __name__ == '__main__':
    args = get_args()

    # debug mode
    debug_on = True
    if debug_on:
        args.dataset_name = 'test_order4'  # 'Grocery_and_Gourmet_Food'
        args.emb_size = 100
        args.batch_size = 256
        args.epoch = 200
        args.l2 = 1e-4
        args.lr=5e-5
        args.test_epoch = 1
        args.stop=5
        args.model_name = 'SLRC_BPR'
        args.load_corpus=True

    main(args)
