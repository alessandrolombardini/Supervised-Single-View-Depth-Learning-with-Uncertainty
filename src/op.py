from torch.utils.tensorboard import SummaryWriter
from models import *
from losses import Loss
from utils.optimizer import make_optimizer
from utils.metrics import compute_psnr, compute_rmse, compute_ause, compute_auce
from utils.summary import summary
from tqdm import tqdm


class Operator:
    def __init__(self, config, check_point):
        self.config = config
        self.epochs = config.epochs
        self.uncertainty = config.uncertainty
        self.ckpt = check_point
        self.tensorboard = config.tensorboard
        if self.tensorboard:
            self.summary_writer = SummaryWriter(self.ckpt.log_dir, 300)
            print("Tensorboard is activated.")
            print("Run tensorboard --logdir={}".format(self.ckpt.log_dir))
        # Set model, criterion, optimizer
        self.model = Model(config)
        self.criterion = Loss(config)
        self.optimizer = make_optimizer(config, self.model)
        # Summary
        summary(self.model, config_file=self.ckpt.config_file)
        # Load ckpt, model, optimizer
        if self.ckpt.exp_load is not None or not config.is_train:
            print("Loading model... ")
            self.load(self.ckpt)
            print(self.ckpt.last_epoch, self.ckpt.global_step)



    def train(self, data_loader):
        last_epoch = self.ckpt.last_epoch
        best_psnr = None
        best_rmse = None
        best_ause = None
        for epoch in range(last_epoch, self.epochs):
            self.model.train()
            for _, batch_data in tqdm(enumerate(data_loader['train'])):
                batch_input = batch_data['image'].to(self.config.device)
                batch_label = batch_data['depth'].to(self.config.device)
                # Permute to have channels at the beginning
                batch_input = batch_input.permute(0, 3, 1, 2) 
                batch_label = batch_label.permute(0, 3, 1, 2)  
                # Forward
                batch_results = self.model(batch_input)
                loss = self.criterion(batch_results, batch_label)
                # Backward
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                break
            # Feedback
            print('[Epoch: {:03d}/{:03d}] Loss: {:5f}'.format(epoch, self.config.epochs, loss.item()))
            if self.tensorboard:
                self.summary_writer.add_scalar('train/loss',
                                                loss, 
                                                epoch)
                self.summary_writer.add_images("train/input_img",
                                                batch_input,
                                                epoch)
                self.summary_writer.add_images("train/mean_img",
                                                torch.clamp(batch_results['mean'], 0., 1.),
                                                epoch)
                self.summary_writer.add_scalar('train/lr',
                                               self.optimizer.get_lr(), epoch)
            # Test model & save model
            self.optimizer.schedule()
            self.save(self.ckpt, epoch)
            if (epoch + 1) % 5 == 0: 
                self.test(data_loader, 'train', epoch)
                if self.uncertainty != "normal":
                    psnr, rmse, ause = self.test(data_loader, 'test', epoch)
                else:
                    psnr, rmse = self.test(data_loader, 'test', epoch)

                if best_ause is None or ause < best_ause:
                    best_psnr = psnr
                    best_rmse = rmse
                    if self.uncertainty != "normal":
                        best_ause = ause

        if self.uncertainty != "normal":
            print("Best PSNR: {:5f}, Best RMSE: {:5f}, Best AUSE: {:5f}".format(best_psnr, best_rmse, best_ause))
        else:
            print("Best PSNR: {:5f}, Best RMSE: {:5f}".format(best_psnr, best_rmse))
        self.summary_writer.close()



    def test(self, data_loader, label, epoch):
        psnrs = []
        rmses = []
        auses = []
        auces = []
        total_psnr = 0.
        total_rmse = 0. 
        total_ause = 0.
        total_auce = 0.
        with torch.no_grad():
            self.model.eval()
            for _, batch_data in enumerate(data_loader[label]):
                batch_input = batch_data['image'].to(self.config.device)
                batch_label = batch_data['depth'].to(self.config.device)
                # Permute to have channels at the beginning
                batch_input = batch_input.permute(0, 3, 1, 2) 
                batch_label = batch_label.permute(0, 3, 1, 2)  
                # Forward
                batch_results = self.model(batch_input)
                # Metrices 
                ## PSNR
                current_psnr = compute_psnr(batch_input, batch_results)
                psnrs.append(current_psnr)
                total_psnr += current_psnr
                # RMSE
                current_rmse = compute_rmse(batch_input, batch_results)
                rmses.append(current_rmse)
                total_rmse += current_rmse
                if self.uncertainty != "normal":
                    # AUSE  
                    current_ause = compute_ause(batch_input, batch_results)
                    auses.append(current_ause)
                    total_ause += current_ause
                    # AUCE
                    #current_auce = compute_auce(batch_input, batch_results)
                    #auces.append(current_auce)
                    #total_auce += current_auce
                break
            # Feedback      
            if self.uncertainty != "normal":
                #print('[Epoch: {:03d}/{:03d}][{}] PSNR {:5f}, RMSE {:5f}, AUSE {:5f}, AUCE {:5f}'
                #    .format(epoch, self.config.epochs, label.upper(),
                #            total_psnr/len(psnrs),
                #            total_rmse/len(rmses),
                #            total_ause/len(auses) if self.uncertainty != "normal" else 'x',
                #            total_auce/len(auces) if self.uncertainty != "normal" else 'x'))
                print('[Epoch: {:03d}/{:03d}][{}] PSNR {:5f}, RMSE {:5f}, AUSE {:5f}'
                    .format(epoch, self.config.epochs, label.upper(),
                            total_psnr/len(psnrs),
                            total_rmse/len(rmses),
                            total_ause/len(auses) if self.uncertainty != "normal" else 'x'))
            else:
                print('[Epoch: {:03d}/{:03d}][{}] PSNR {:5f}, RMSE {:5f}'
                    .format(epoch, self.config.epochs, label.upper(),
                            total_psnr/len(psnrs),
                            total_rmse/len(rmses)))
                
            if self.tensorboard:
                if label == 'test':
                    self.summary_writer.add_images("eval/test/input_img",
                                                    batch_input, 
                                                    epoch)
                    if self.uncertainty != 'normal':
                        self.summary_writer.add_images("eval/test/mean_img",
                                                        torch.clamp(batch_results['mean'], 0., 1.),
                                                        epoch)
                        self.summary_writer.add_images("eval/test/var_img",
                                                    torch.clamp(batch_results['var'], 0., 1.), #?
                                                    epoch)
                self.summary_writer.add_scalar('eval/{}/mean_psnr'.format(label),
                                                total_psnr/len(psnrs), 
                                                epoch)
                self.summary_writer.add_scalar('eval/{}/mean_rmse'.format(label),
                                                total_rmse/len(rmses), 
                                                epoch)
                if self.uncertainty != 'normal':
                    self.summary_writer.add_scalar('eval/{}/mean_ause'.format(label),
                                                    total_ause/len(auses), 
                                                    epoch)
                    #self.summary_writer.add_scalar('eval/{}/mean_auce'.format(label),
                    #                                total_auce/len(auces), 
                    #                                epoch)

        if self.uncertainty != "normal":
            return total_psnr/len(psnrs), total_rmse/len(rmses), total_ause/len(auses) #, total_auce/len(auces)
        else:
            return total_psnr/len(psnrs), total_rmse/len(rmses)


    def load(self, ckpt):
        ckpt.load()               # load ckpt
        self.model.load(ckpt)     # load model
        self.optimizer.load(ckpt) # load optimizer



    def save(self, ckpt, epoch):
        ckpt.save(epoch)             # save ckpt: global_step, last_epoch
        self.model.save(ckpt, epoch) # save model: weight
        self.optimizer.save(ckpt)    # save optimizer:


