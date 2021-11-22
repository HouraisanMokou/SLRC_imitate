import logging
import time

import numpy as np
import torch.optim

from tqdm import tqdm

from src.models.SLRC import SLRC
from src.models.SLRC import Dataset
from typing import NoReturn,List
from torch.utils.data import DataLoader
from torch import nn

class Runner():
    """
    to control how the net is trained and tested.
    """
    def __init__(self,args):
        self.datasets=args.datasets
        self.epoch=args.epoch
        self.test_epoch = args.test_epoch
        self.stop=args.stop
        self.l2=args.l2
        self.lr=args.lr
        self.batch_size=args.batch_size
        self.eval_batch_size = args.eval_batch_size
        self.emb_size=args.emb_size
        self.num_workers=args.num_workers
        self.pin_memory=args.pin_memory

        self.logger=args.logger

        self.device=torch.device('cuda:0'if torch.cuda.is_available() else 'cpu')
        self.logger.info('register device: {}'.format(self.device))


    def evaluate(self,dataset:Dataset)->float:
        """
        evaluate on val or test set
        :param: dataset: test or val
        :return: a 2D-ndarray with cols are top 5 and top 10 ndcg, acc, precision, recall
        """
        model =dataset.model
        dl=DataLoader(dataset, batch_size=self.batch_size, shuffle=True, collate_fn=dataset._collate_batch,
                      num_workers=self.num_workers,pin_memory=self.pin_memory)
        with torch.no_grad():
            for batch in tqdm(dl, leave=False,
                              desc='epoch {:<4d}'.format(self.cur_epoch if self.cur_epoch is not None else '')
                    , ncols=100, mininterval=1):
                # to gpu
                for _ in batch:
                    if type(batch[_]) is torch.Tensor:
                        batch[_] = batch[_].to(self.device)

                out = model(batch)

                # for debug
                time.sleep(1)





    def predict(self,dataset:Dataset)->np.ndarray:
        """
        predict on val or test
        :param dataset: the dataset to be test
        :return: a 2D-ndarray with each row is that [\lambda_{item}, \lambda_{neg_item}]
        """

    def fit(self,dataset:Dataset)->float:
        """
        train part in a epoch
        return the used time
        :param dataset:
        :return:
        """
        t1=time.time()
        model=dataset.model
        if model.optimizer ==None:
            self._set_optimizer(model)

        dataset.shuffle_neg()

        dl=DataLoader(dataset, batch_size=self.batch_size, shuffle=True, collate_fn=dataset._collate_batch,
                      num_workers=self.num_workers,pin_memory=self.pin_memory)

        for batch in tqdm(dl,leave=False,desc='epoch {:<4d}'.format(self.cur_epoch if self.cur_epoch is not  None else '')
                          ,ncols=100,mininterval=1):
            # train in a single batch

            #to gpu
            for _ in batch:
                if type(batch[_]) is torch.Tensor:
                    batch[_]=batch[_].to(self.device)

            out = model(batch)
            loss =model.loss(out)
            loss.backward()
            model.optimizer.step()

            # for debug
            time.sleep(0.05)

        return time.time()-t1





    def _set_optimizer(self,model:nn.Module):
        """
        set up the optimizer for model
        :param model: target model
        :return:
        """
        model.optimizer=torch.optim.Adam(model.classify_params(),lr=self.lr,weight_decay=self.l2)
        logging.info('Optimizer has set')

    def train(self)->NoReturn:
        """
        control the train
        :return:
        """
        dataset = self.datasets['train']
        self.cur_epoch=0

        model=dataset.model
        model.to(self.device)

        time_list=[]
        for epoch in range(self.epoch):
            self.cur_epoch+=1
            used_time=self.fit(dataset)
            time_list.append(used_time)

            if self.test_epoch!=-1 and self.cur_epoch % self.test_epoch ==0:
                # test on val
                self.evaluate(self.datasets['val'])

        self.logger.info('average time used in training: {:<3f}'.format(np.mean(np.array(time_list))))

    def eval_termination(self, val_ndcg: List[float]) -> bool:
        """
        to stop early if ndcg decreasing
        :param val_ndcg: the ndcg of val, can get by evaluate()
        :return:whether to shut down the process
        """


