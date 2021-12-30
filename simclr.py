import logging
import os
from shutil import make_archive
import sys

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torchvision.utils import make_grid
from utils import save_config_file, accuracy, save_checkpoint

torch.manual_seed(0)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class SimCLR(object):

    def __init__(self, *args, **kwargs):
        self.args = kwargs['args']
        self.model = kwargs['model'].to(self.args.device)
        self.optimizer = kwargs['optimizer']
        self.scheduler = kwargs['scheduler']

        folder_name = "runs/[" + str(self.args.dataset_name) + "]_BS=" + str(self.args.bs) + "_LR=" + str(self.args.lr) + "_eps=" + str(self.args.eps)

        self.writer = SummaryWriter(log_dir=folder_name)
        logging.basicConfig(filename=os.path.join(self.writer.log_dir, 'training.log'), level=logging.DEBUG)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.args.device)

    def info_nce_loss(self, features, targ):

        labels = torch.cat([torch.arange(self.args.bs) for i in range(self.args.n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.args.device)

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)
        # assert similarity_matrix.shape == (
        #     self.args.n_views * self.args.bs, self.args.n_views * self.args.bs)
        # assert similarity_matrix.shape == labels.shape

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.args.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.args.device)

        logits = logits / self.args.temperature

        loss = self.criterion(logits, labels)
        return loss, logits, labels

    def train(self, train_loader):

        scaler = GradScaler(enabled=self.args.fp16_precision)

        # save config file
        save_config_file(self.writer.log_dir, self.args)
        logging.info(f"Start SimCLR training for {self.args.epochs} epochs.")
        logging.info(f"Training with gpu: {self.args.disable_cuda}.")

        attack_kwargs = {
            'eps': self.args.eps,
            'step_size': 1,
            'iterations': 3,
            'custom_loss': self.info_nce_loss,
            #'bypass' : 0
        }

        # Speed up training if eps=0
        if (self.args.eps == 0):
            make_adv = False
        else:
            make_adv = True

        n_iter = 0

        for epoch_counter in range(self.args.epochs):
            
            losses = AverageMeter()
            top1 = AverageMeter()
            top5 = AverageMeter()

            for images, target in tqdm(train_loader):
                images = torch.cat(images, dim=0)

                images = images.to(self.args.device)
                target = target.to(self.args.device)

                with autocast(enabled=self.args.fp16_precision):
                    features, adv_img = self.model(images, target, make_adv, **attack_kwargs)
                    loss, logits, labels = self.info_nce_loss(features, target)

                #print(logits[0])

                prec1, prec5 = accuracy(logits, labels, topk=(1, 5))
                prec1, prec5 = prec1[0], prec5[0]
                
                losses.update(loss.item(), images.size(0))
                top1.update(prec1, images.size(0))
                top5.update(prec5, images.size(0))

                self.optimizer.zero_grad()

                scaler.scale(loss).backward()

                scaler.step(self.optimizer)
                scaler.update()

                if n_iter % self.args.log_every_n_steps == 0:

                    top1_iter, top5_iter = accuracy(logits, labels, topk=(1, 5))
                    self.writer.add_scalar('top1 (iterations)', top1_iter[0], global_step=n_iter)
                    self.writer.add_scalar('top5 (iterations)', top5_iter[0], global_step=n_iter)
                    #self.writer.add_scalar('learning_rate', self.scheduler.get_lr()[0], global_step=n_iter)

                    # Print some examples in tensorboard
                    nat0_image = make_grid(images[:15, ...])
                    adv0_image = make_grid(adv_img[:15, ...])
                    self.writer.add_image('Input', nat0_image, epoch_counter)
                    self.writer.add_image('Input - Adversarial', adv0_image, epoch_counter)
                    #self.writer.add_scalar('learning_rate', self.scheduler.get_lr()[0], global_step=n_iter)
                n_iter += 1

            self.writer.add_scalar('loss (iterations)', loss, global_step=n_iter)
            
            if self.writer is not None:
                descs = ['loss (epochs)', 'top1 (epochs)', 'top5 (epochs)']
                vals = [losses, top1, top5]
                for d, v in zip(descs, vals):
                    self.writer.add_scalar(str(d), v.avg, epoch_counter)

            # warmup for the first 10 epochs
            #if epoch_counter >= 10:
            #    self.scheduler.step()

            logging.debug(f"Epoch: {epoch_counter}\tLoss: {loss}\tTop1 accuracy: {top1_iter[0]}")

        logging.info("Training has finished.")
        # save model checkpoints
        checkpoint_name = 'checkpoint_{:04d}.pth.tar'.format(self.args.epochs)
        save_checkpoint({
            'epoch': self.args.epochs,
            'arch': self.args.arch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, is_best=False, filename=os.path.join(self.writer.log_dir, checkpoint_name))
        logging.info(f"Model checkpoint and metadata has been saved at {self.writer.log_dir}.")
