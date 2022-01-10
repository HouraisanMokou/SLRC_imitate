import torch.nn as nn
import torch.utils.data.dataset as dataset
from typing import NoReturn

import common.reader as reader

"""
Auxiliary Methods of Models
"""
def load_model(m:nn.Module,model_path:str)->NoReturn:
    """
    load model from local file
    :param m: the target module
    :param model_path: the path of model
    :return:
    """
    pass

def save_model(m:nn.Module,model_path:str)->NoReturn:
    """
    save model to local file
    :param m: the target module
    :param model_path: the path of model
    :return:
    """
    pass

def count_var(m:nn.Module)->int:
    """
    count number of the variables
    :param m: the target module
    :return: the number
    """
    pass


def actions_before_train(model:nn.Module,data:dataset)->NoReturn:  # e.g., re-initial some special parameters
    """
    do something before train if need
    :param model:
    :return:
    """
    pass

def actions_after_train(model:nn.Module,data:dataset)->NoReturn:
    """
    do something after train if need
    :param model:
    :return:
    """
    pass

"""
Auxiliary for reader and dataset
"""
def save_corpus(corpus:reader)->NoReturn:
    """
    save the corpus to a *.pkl
    :param corpus:
    :return:
    """

"""
Auxiliary for runner
"""
def print_res(self, runner, data: dataset) -> str:
    """
    construct the res for runner before and after training
    :param self:
    :param data:
    :return:
    """

def ndcg_cal(self,gt:list,pred:dict):
    """
    calculate ndcg
    :param self:
    :param gt:
    :param pred:
    :return:
    """