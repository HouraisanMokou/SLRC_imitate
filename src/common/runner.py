import numpy as np

from src.models.SLRC import SLRC
from reader import Dataset
from typing import NoReturn,List

class Runner():
    """
    to control how the net is trained and tested.
    """
    def __init__(self,args):
        self.epoch=args.epoch
        self.test_epoch = args.test_epoch
        self.stop=args.stop
        self.l2=args.l2
        self.lr=args.lr
        self.batch_size=args.batch_size
        self.eval_batch_size = args.eval_batch_size
        self.emb_size=args.emb_size

    def evaluate(self,dataset:Dataset)->np.ndarray:
        """
        evaluate on val or test set
        :param dataset: the dataset to be test (the test set or val set)
        :return: a 2D-ndarray with cols are top 5 and top 10 ndcg, acc, precision, recall
        """

    def predict(self,dataset:Dataset)->np.ndarray:
        """
        predict on val or test
        :param dataset: the dataset to be test
        :return: a 2D-ndarray with each row is that [\lambda_{item}, \lambda_{neg_item}]
        """

    def fit(self,dataset:Dataset)->float:
        """
        train part in a epoch
        return the loss of this epoch
        :param dataset:
        :return:
        """

    def _set_optimizer(self,model:SLRC):
        """
        set up the optimizer for model
        :param model: target model
        :return:
        """

    def train(self)->NoReturn:
        """
        start to train for many epoches , check with a period of several, test after train
        :return:
        """

    def eval_termination(self, val_ndcg: List[float]) -> bool:
        """
        to stop early if ndcg decreasing
        :param val_ndcg: the ndcg of val, can get by evaluate()
        :return:whether to shut down the process
        """


