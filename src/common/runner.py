import copy
import logging
import time

import numpy as np
import torch.optim

from tqdm import tqdm

from src.models.SLRC import SLRC
from typing import NoReturn, List
from torch.utils.data import DataLoader
from torch import nn


class Runner():
    """
    to control how the net is trained and tested.
    """

    def __init__(self, args):
        self.datasets = args.datasets
        self.epoch = args.epoch
        self.test_epoch = args.test_epoch
        self.stop = args.stop
        self.l2 = args.l2
        self.lr = args.lr
        self.batch_size = args.batch_size
        self.eval_batch_size = args.eval_batch_size
        self.emb_size = args.emb_size
        self.num_workers = args.num_workers
        self.pin_memory = args.pin_memory

        self.logger = args.logger

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # self.device = torch.device('cpu')
        self.logger.info('register device: {}'.format(self.device))

    def evaluate(self, dataset: SLRC.Dataset) -> dict:
        """
        evaluate on val or test set
        :param: dataset: test or val
        :return: a 2D-ndarray with cols are top 5 and top 10 ndcg, acc, precision, recall
        """
        prediction = self.predict(dataset)
        pos = np.argwhere(prediction == 0)[:, 1] + 1
        hit = (pos <= 5)
        acc = hit.mean()
        ndcg = (hit / np.log2(pos + 1)).mean()
        return {'acc': acc, 'ndcg': ndcg}

    def predict(self, dataset: SLRC.Dataset) -> np.ndarray:
        """
        predict on val or test
        :param dataset: the dataset to be test
        :return: a 2D-ndarray with each row is that [\lambda_{item}, \lambda_{neg_item}]
        """
        model = dataset.model
        dl = DataLoader(dataset, batch_size=2*self.batch_size, shuffle=False, collate_fn=dataset._collate_batch,
                        num_workers=self.num_workers, pin_memory=self.pin_memory)
        predict_list = None

        with torch.no_grad():
            for batch in tqdm(dl, leave=False,
                              desc='epoch {:<4d}'.format(self.cur_epoch if self.cur_epoch is not None else '')
                    , ncols=100, mininterval=1):
                # to gpu

                for _ in batch:
                    if type(batch[_]) is torch.Tensor:
                        batch[_] = batch[_].to(self.device)

                out = model(batch)
                preds = out['prediction'].cpu().numpy()

                predictions = np.argsort(-preds, axis=1)
                if predict_list is None:
                    predict_list = predictions
                else:
                    predict_list = np.vstack((predict_list, predictions))
                # for debug
                # time.sleep(1)
        return predict_list

    def fit(self, dataset: SLRC.Dataset) -> (float, float):
        """
        train part in a epoch
        return the used time
        :param dataset:
        :return:
        """
        t1 = time.time()
        model = dataset.model
        if model.optimizer == None:
            self._set_optimizer(model)

        dataset.shuffle_neg()

        dl = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, collate_fn=dataset._collate_batch,
                        num_workers=self.num_workers, pin_memory=self.pin_memory)

        loss_list = list()
        for batch in tqdm(dl, leave=False,
                          desc='epoch {:<4d}'.format(self.cur_epoch if self.cur_epoch is not None else '')
                , ncols=100, mininterval=1):
            # train in a single batch

            # to gpu
            for _ in batch:
                if type(batch[_]) is torch.Tensor:
                    batch[_] = batch[_].to(self.device)

            out = model(batch)
            loss = model.loss(out)
            model.optimizer.zero_grad()
            loss.backward()
            model.optimizer.step()

            loss_list.append(loss.detach().cpu().data.numpy())
            # for debug
            # time.sleep(0.05)

        return time.time() - t1, np.mean(loss_list)

    def _set_optimizer(self, model: nn.Module):
        """
        set up the optimizer for model
        :param model: target model
        :return:
        """
        # model.optimizer = torch.optim.Adam(model.classify_params(), lr=self.lr, weight_decay=self.l2)
        model.optimizer = torch.optim.Adagrad(model.classify_params(), lr=self.lr, weight_decay=self.l2)
        logging.info('Optimizer has set')

    def train(self) -> NoReturn:
        """
        control the train
        :return:
        """
        dataset = self.datasets['train']
        self.cur_epoch = 0

        model = dataset.model
        model.to(self.device)

        time_list = []
        eval_list = []
        loss_list = []
        for epoch in range(self.epoch):
            try:
                self.cur_epoch += 1
                used_time, epoch_loss = self.fit(dataset)
                time_list.append(used_time)
                loss_list.append(epoch_loss)
                if self.test_epoch != -1 and self.cur_epoch % self.test_epoch == 0:
                    # test on val
                    val_res = self.evaluate(self.datasets['val'])
                    self.logger.info(
                        'at epoch{}: (HR: {}, NDCG: {}) loss: {} used time: {} '.format(self.cur_epoch, val_res['acc'],
                                                                                        val_res['ndcg'], epoch_loss,
                                                                                        used_time))
                    eval_list.append(val_res)

                if epoch_loss<1e-4:
                    break
                if self.eval_termination(eval_list):
                    break
            except:
                self.print_result(time_list, loss_list, eval_list)
                raise KeyboardInterrupt

        self.logger.info('average time used in each epoch in training: {:.3f}s'.format(np.mean(np.array(time_list))))
        self.logger.info('stop at epoch: {:<4d}'.format(self.cur_epoch))

        self.logger.info('evaluate on test set')
        test_res = self.evaluate(self.datasets['test'])
        self.logger.info('average ndcg in test set: {:.3f}'.format(test_res['ndcg']))
        self.logger.info('acc in test set: {:.3f}'.format(test_res['acc']))
        self.logger.info('detailed list as below')
        self.print_result(time_list,loss_list,eval_list)

    def print_result(self,time_list,loss_list,eval_list):
        self.logger.info('*' * 80)
        self.logger.info('time list')
        self.logger.info(time_list)
        self.logger.info('*' * 80)
        self.logger.info('loss list')
        self.logger.info(loss_list)
        self.logger.info('*' * 80)
        hr_list, ndcg_list = [], []
        for e in eval_list:
            hr_list.append(e['acc'])
            ndcg_list.append(e['ndcg'])
        self.logger.info('HR list')
        self.logger.info(hr_list)
        self.logger.info('*' * 80)
        self.logger.info('NDCG list')
        self.logger.info(ndcg_list)
    def eval_termination(self, eval_list: List[dict]) -> bool:
        """
        to stop early if ndcg decreasing
        :param val_ndcg: the ndcg of val, can get by evaluate()
        :return:whether to shut down the process
        """
        last = -np.inf
        for i in range(len(eval_list)):
            if i == self.stop:
                return True
            pos = len(eval_list) - 1 - i
            ndcg = eval_list[pos]['ndcg']
            if  2e-4< last-ndcg:
                return False
            last = ndcg

        return False
