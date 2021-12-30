from models.make_adv import make_adv
from models.resnet_simclr import ResNetSimCLR
import torch
import torch.backends.cudnn as cudnn
from torchvision import models
import argparse
import numpy as np
import os
import yaml
import matplotlib.pyplot as plt
import torchvision

from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.tensorboard import SummaryWriter
import logging

from utils import save_config_file, accuracy, save_checkpoint

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch SimCLR')
parser.add_argument('-data', metavar='DIR', default='/datasets/',
                    help='path to dataset', dest='dataset_path')
parser.add_argument('-dataset-name', default='cifar10',
                    help='dataset name', choices=['stl10', 'cifar10','cifar100'])
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=12, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=250, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel', dest='bs')
parser.add_argument('--lr', '--learning-rate', default=8e-5, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')

parser.add_argument('--checkpoint', default='../runs/[cifar]_BS=256_LR=2e-4_eps=0.5/checkpoint_0500.pth.tar',
                    help='Checkpoint to resume model for fine-tuning.', dest='checkpoint')
parser.add_argument('--pretrained-dataset', default='cifar10',
                    help='Name of dataset used in checkpoint model', dest='ftDataset')
parser.add_argument('--eps', '--eps', default=0.5, type=float,
                    metavar='EPS', help='WARNING! Only to point out eps of pretrained model in filename', dest='eps')


parser.add_argument('--disable-cuda', action='store_true',
                    help='Disable CUDA')
parser.add_argument('--fp16-precision', action='store_true',
                    help='Whether or not to use 16-bit precision GPU training.')

parser.add_argument('--out_dim', default=128, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('--log-every-n-steps', default=100, type=int,
                    help='Log every n steps')
parser.add_argument('--temperature', default=0.07, type=float,
                    help='softmax temperature (default: 0.07)')
parser.add_argument('--n-views', default=2, type=int, metavar='N',
                    help='Number of views for contrastive learning training.')
parser.add_argument('--gpu-index', default=2, type=int, help='Gpu index.')

def get_cifar10_data_loaders(path, download, shuffle=False, batch_size=256):
    train_dataset = datasets.CIFAR10(path, train=True, download=download,
                                    transform=transforms.ToTensor())

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                num_workers=0, drop_last=False, shuffle=shuffle)
  
    test_dataset = datasets.CIFAR10(path, train=False, download=download,
                                    transform=transforms.ToTensor())

    test_loader = DataLoader(test_dataset, batch_size=2*batch_size,
                                num_workers=10, drop_last=False, shuffle=shuffle)
    return train_loader, test_loader

def get_cifar100_data_loaders(path, download, shuffle=False, batch_size=256):
    train_dataset = datasets.CIFAR100(path, train=True, download=download,
                                    transform=transforms.ToTensor())

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                num_workers=0, drop_last=False, shuffle=shuffle)
  
    test_dataset = datasets.CIFAR100(path, train=False, download=download,
                                    transform=transforms.ToTensor())

    test_loader = DataLoader(test_dataset, batch_size=2*batch_size,
                                num_workers=10, drop_last=False, shuffle=shuffle)
    return train_loader, test_loader


def get_stl10_data_loaders(path, download, shuffle=False, batch_size=256):
    train_dataset = datasets.STL10(path, split='train', download=download,
                                    transform=transforms.ToTensor())

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                num_workers=0, drop_last=False, shuffle=shuffle)
  
    test_dataset = datasets.STL10('./data', split='test', download=download,
                                    transform=transforms.ToTensor())

    test_loader = DataLoader(test_dataset, batch_size=2*batch_size,
                                num_workers=10, drop_last=False, shuffle=shuffle)
    return train_loader, test_loader



def main():
    args = parser.parse_args()
    assert args.n_views == 2, "Only two view training is supported. Please use --n-views 2."
    # check if gpu training is available
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
        cudnn.deterministic = True
        cudnn.benchmark = True
    else:
        args.device = torch.device('cpu')
        args.gpu_index = -1

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using device:", device)

    # Initialize writer for Tensorboard
    folder_name = "runs/FT/["+ str(args.ftDataset) + "-" + str(args.dataset_name) +  "]_BS=" + str(args.bs) + "_LR=" + str(args.lr) + "_ADVModelEps=" + str(args.eps)

    writer = SummaryWriter(log_dir=folder_name)
    logging.basicConfig(filename=os.path.join(writer.log_dir, 'training.log'), level=logging.DEBUG)

    if args.dataset_name == 'cifar10':
        if args.arch == 'resnet18':
            model = torchvision.models.resnet18(pretrained=False, num_classes=10).to(device)
            #model = ResNetEval(base_model=args.arch, out_dim=10).to(device)
        elif args.arch == 'resnet50':
            model = torchvision.models.resnet50(pretrained=False, num_classes=10).to(device)
    elif args.dataset_name == 'cifar100':
        if args.arch == 'resnet18':
            model = torchvision.models.resnet18(pretrained=False, num_classes=100).to(device)
        elif args.arch == 'resnet50':
            model = torchvision.models.resnet50(pretrained=False, num_classes=100).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    state_dict = checkpoint['state_dict']

    for k in list(state_dict.keys()):
        if k.startswith('backbone.'):
            if k.startswith('backbone') and not k.startswith('backbone.fc'):
                # remove prefix
                state_dict[k[len("backbone."):]] = state_dict[k]
        del state_dict[k]
    
    log = model.load_state_dict(state_dict, strict=False)
    assert log.missing_keys == ['fc.weight', 'fc.bias']


    if args.dataset_name == 'cifar10':
        train_loader, test_loader = get_cifar10_data_loaders(args.dataset_path, download=True, batch_size=args.bs)
    elif args.dataset_name == 'cifar100':
        train_loader, test_loader = get_cifar100_data_loaders(args.dataset_path, download=True, batch_size=args.bs)
    elif args.dataset_name == 'stl10':
        train_loader, test_loader = get_stl10_data_loaders(args.dataset_path, download=True, batch_size=args.bs)
    print("Dataset:", args.dataset_name)

    # freeze all layers but the last fc
    for name, param in model.named_parameters():
        if name not in ['fc.weight', 'fc.bias']:
            param.requires_grad = False
    
    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    assert len(parameters) == 2  # fc.weight, fc.bias

    print(args)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = torch.nn.CrossEntropyLoss().to(device)

    
    # save config file
    save_config_file(writer.log_dir, args)

    n_iter_train = 0
    n_iter_test = 0
    logging.info(f"Start fine tuning for {args.epochs} epochs.")
    logging.info(f"Pre-trained model loaded: {args.checkpoint}.")
    logging.info(f"Training with gpu: {args.disable_cuda}.")

    attack_kwargs = {
            'eps': args.eps,
            'step_size': 1,
            'iterations': 3
            #'bypass' : 0
        }

    epochs = 100
    for epoch in range(epochs):
        top1_train_accuracy = 0
        for counter, (x_batch, y_batch) in enumerate(train_loader):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            adv_img = make_adv(model, x_batch, y_batch, **attack_kwargs)

            logits = model(adv_img)
            loss = criterion(logits, y_batch)
            top1 = accuracy(logits, y_batch, topk=(1,))
            top1_train_accuracy += top1[0]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if n_iter_train % args.log_every_n_steps == 0:
                writer.add_scalar('loss', loss, global_step=n_iter_train)
                writer.add_scalar('acc/FT/TRAIN_acc-Top1', top1_train_accuracy.item()/(counter+1), global_step=n_iter_train)
            
            n_iter_train += 1


        top1_train_accuracy /= (counter + 1)
        top1_accuracy = 0
        top5_accuracy = 0

        for counter, (x_batch, y_batch) in enumerate(test_loader):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            adv_img = make_adv(model, x_batch, y_batch, **attack_kwargs)

            logits = model(x_batch)

            top1, top5 = accuracy(logits, y_batch, topk=(1,5))
            top1_accuracy += top1[0]
            top5_accuracy += top5[0]

            if n_iter_test % args.log_every_n_steps == 0:
                writer.add_scalar('acc/FT/TEST_acc-Top1', top1_accuracy.item()/(counter+1), global_step=n_iter_test)
                writer.add_scalar('acc/FT/TEST_acc-Top5', top5_accuracy.item()/(counter+1), global_step=n_iter_test)
            
            n_iter_test += 1
    
        top1_accuracy /= (counter + 1)
        top5_accuracy /= (counter + 1)

        print(f"Epoch {epoch}\tTop1 Train accuracy {top1_train_accuracy.item()}\tTop1 Test accuracy: {top1_accuracy.item()}\tTop5 test acc: {top5_accuracy.item()}")
        logging.debug(f"Epoch {epoch}\tTop1 Train accuracy {top1_train_accuracy.item()}\tTop1 Test accuracy: {top1_accuracy.item()}\tTop5 test acc: {top5_accuracy.item()}")
    
    logging.info("Evaluation has finished.\n")

    # save model checkpoints
    checkpoint_name = 'checkpoint_{:04d}.pth.tar'.format(args.epochs)
    save_checkpoint({
        'epoch': args.epochs,
        'arch': args.arch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, is_best=False, filename=os.path.join(writer.log_dir, checkpoint_name))
    logging.info(f"Model checkpoint and metadata has been saved at {writer.log_dir}.")


if __name__ == "__main__":
    main()