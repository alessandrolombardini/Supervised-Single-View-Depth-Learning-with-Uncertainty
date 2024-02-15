import numpy as np
import pandas as pd
import math
from torch.utils.tensorboard import SummaryWriter
from torch_uncertainty.metrics import AUSE

from model import *
from loss import Loss
from util import make_optimizer, compute_psnr, compute_ause, summary


class Operator:
    def __init__(self, config, ckeck_point):
        self.config = config
        self.epochs = config.epochs
        self.uncertainty = config.uncertainty
        self.ckpt = ckeck_point
        self.tensorboard = config.tensorboard
        if self.tensorboard:
            self.summary_writer = SummaryWriter(self.ckpt.log_dir, 300)
            print("Tensorboard is activated.")
            print("Run tensorboard --logdir={}".format(self.ckpt.log_dir))

        # set model, criterion, optimizer
        self.model = Model(config)
        summary(self.model, config_file=self.ckpt.config_file)

        # set criterion, optimizer
        self.criterion = Loss(config)
        self.optimizer = make_optimizer(config, self.model)

        # load ckpt, model, optimizer
        if self.ckpt.exp_load is not None or not config.is_train:
            print("Loading model... ")
            self.load(self.ckpt)
            print(self.ckpt.last_epoch, self.ckpt.global_step)



    def train(self, data_loader):
        last_epoch = self.ckpt.last_epoch
        train_batch_num = len(data_loader['train'])


        for epoch in range(last_epoch, self.epochs):
            self.model.train()

            for batch_idx, batch_data in enumerate(data_loader['train']):

                batch_input, batch_label = batch_data
                batch_input = batch_input.to(self.config.device)
                batch_label = batch_label.to(self.config.device)
                
                # forward
                batch_results = self.model(batch_input)
                loss = self.criterion(batch_results, batch_input)

                # backward
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                print('Epoch: {:03d}/{:03d}, Iter: {:03d}/{:03d}, Loss: {:5f}'
                    .format(epoch, self.config.epochs,
                            batch_idx, train_batch_num,
                            loss.item()))

                # use tensorboard
                if self.tensorboard:
                    current_global_step = self.ckpt.step()
                    self.summary_writer.add_scalar('train/loss',
                                                   loss, current_global_step)
                    if batch_idx % 100:
                        self.summary_writer.add_images("train/input_img",
                                                    batch_input,
                                                    current_global_step)
                        self.summary_writer.add_images("train/mean_img",
                                                    torch.clamp(batch_results['mean'], 0., 1.),
                                                    current_global_step)

            # use tensorboard
            if self.tensorboard:
                #print(self.optimizer.get_lr(), epoch)
                self.summary_writer.add_scalar('epoch_lr',
                                               self.optimizer.get_lr(), epoch)

            # test model & save model
            self.optimizer.schedule()
            self.save(self.ckpt, epoch)
            self.test(data_loader)

        self.summary_writer.close()



    def test(self, data_loader):
        with torch.no_grad():
            self.model.eval()

            total_psnr = 0.
            psnrs = []
            test_batch_num = len(data_loader['test'])

            if self.uncertainty != 'normal':
                auses = []
                total_ause = 0.

            for batch_idx, batch_data in enumerate(data_loader['test']):
                
                batch_input, batch_label = batch_data
                batch_input = batch_input.to(self.config.device)
                batch_label = batch_label.to(self.config.device)

                # Forward
                batch_results = self.model(batch_input)

                # Metrices 
                ## Calculate PSNR
                current_psnr = compute_psnr(batch_results['mean'], batch_input)
                psnrs.append(current_psnr)
                total_psnr += current_psnr
                ## AUSE
                #if self.uncertainty != "normal":
                #    current_ause = compute_ause(batch_input, batch_results)
                #    auses.append(current_ause)
                #    total_ause += current_ause

                # use tensorboard
                if self.tensorboard:
                    step = self.ckpt.do_step_test()
                    if batch_idx % 100:
                        self.summary_writer.add_images("test/input_img",
                                                    batch_input, 
                                                    step)
                        self.summary_writer.add_images("test/mean_img",
                                                    torch.clamp(batch_results['mean'], 0., 1.),
                                                    step)
                        self.summary_writer.add_images("test/var_img",
                                                    batch_results['var'] / torch.max(batch_results['var']),
                                                    step)
                
                #if self.uncertainty != 'normal':
                #    print("Test: Iter: {:03d}/{:03d}, AUSE {:5f}, PSNR {:5f}".format(
                #        batch_idx, test_batch_num, current_ause, current_psnr))
                #else:
                #    print("Test: Iter: {:03d}/{:03d}, PSNR {:5f}".format(
                #        batch_idx, test_batch_num, current_psnr))
                    
            #print("[FINAL] Test: Iter: {:03d}/{:03d}, AUSE {:5f}, PSNR {:5f}".format(
            #    batch_idx, test_batch_num, total_ause/len(auses), total_psnr/len(psnrs)))
            print("[FINAL] Test: Iter: {:03d}/{:03d}, PSNR {:5f}".format(
                batch_idx, test_batch_num, total_psnr/len(psnrs)))
        

            # use tensorboard
            if self.tensorboard:
                self.summary_writer.add_scalar('test/mean_psnr',
                                            total_psnr/len(psnrs), self.ckpt.last_epoch)
                #if not self.uncertainty == 'normal':
                #    self.summary_writer.add_scalar('test/mean_ause',
                #                                   total_ause/len(auses), 
                #                                   self.ckpt.last_epoch)

    def load(self, ckpt):
        ckpt.load() # load ckpt
        self.model.load(ckpt) # load model
        self.optimizer.load(ckpt) # load optimizer

    def save(self, ckpt, epoch):
        ckpt.save(epoch) # save ckpt: global_step, last_epoch
        self.model.save(ckpt, epoch) # save model: weight
        self.optimizer.save(ckpt) # save optimizer:


