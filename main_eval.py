from AdversarialSSL.models.resnet_simclr import ResNetSimCLR
from models.resnet_eval import ResNetEval
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
from torchvision.utils import make_grid
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
parser.add_argument('--epochs', default=500, type=int, metavar='N',
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

parser.add_argument('--checkpoint', default='../runs/FT/[cifar100-cifar10]_BS=256_LR=8e-05_ADVModelEps=0.5/checkpoint_0500.pth.tar',
                    help='Checkpoint to resume model for fine-tuning.', dest='checkpoint')
parser.add_argument('--pretrained-dataset', default='cifar10',
                    help='Name of dataset used in checkpoint model', dest='ftDataset')
parser.add_argument('--eps', '--eps', default=0.2, type=float,
                    metavar='EPS', help='WARNING! Only to point out eps of pretrained model in filename', dest='eps')
parser.add_argument('--range', '--range', default=1, type=int,
                    metavar='RANGE', help='...', dest='range')


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

    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                                num_workers=10, drop_last=False, shuffle=shuffle)
    return train_loader, test_loader

def get_cifar100_data_loaders(path, download, shuffle=False, batch_size=256):
    train_dataset = datasets.CIFAR100(path, train=True, download=download,
                                    transform=transforms.ToTensor())

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                num_workers=0, drop_last=False, shuffle=shuffle)
  
    test_dataset = datasets.CIFAR100(path, train=False, download=download,
                                    transform=transforms.ToTensor())

    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                                num_workers=10, drop_last=False, shuffle=shuffle)
    return train_loader, test_loader


def get_stl10_data_loaders(path, download, shuffle=False, batch_size=256):
    train_dataset = datasets.STL10(path, split='train', download=download,
                                    transform=transforms.ToTensor())

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                num_workers=0, drop_last=False, shuffle=shuffle)
  
    test_dataset = datasets.STL10('./data', split='test', download=download,
                                    transform=transforms.ToTensor())

    test_loader = DataLoader(test_dataset, batch_size=batch_size,
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
    print(f"Using device: {device}\tIndex: {torch.cuda.device_count()}")

    # Initialize writer for Tensorboard
    folder_name = "runs/EVAL/["+ str(args.ftDataset) + "-" + str(args.dataset_name) +  "]_BS=" + str(args.bs) + "_TestEps=" + str(args.eps)

    writer = SummaryWriter(log_dir=folder_name)
    logging.basicConfig(filename=os.path.join(writer.log_dir, 'training.log'), level=logging.DEBUG)

    if args.dataset_name == 'cifar10':
        model = ResNetEval(base_model=args.arch, out_dim=10).to(device)
    elif args.dataset_name == 'cifar100':
        model = ResNetEval(base_model=args.arch, out_dim=100).to(device)
    

    checkpoint = torch.load(args.checkpoint, map_location=device)
    state_dict = checkpoint['state_dict']

    for k in list(state_dict.keys()):
        if k.startswith('backbone.'):
            if k.startswith('backbone') and not k.startswith('backbone.fc'):
                # remove prefix
                state_dict[k[len("backbone."):]] = state_dict[k]
        del state_dict[k]

    log = model.load_state_dict(state_dict, strict=False)

    if args.dataset_name == 'cifar10':
        train_loader, test_loader = get_cifar10_data_loaders(args.dataset_path, download=True, batch_size=args.bs, shuffle=True)
        num_classes = 10
    elif args.dataset_name == 'cifar100':
        train_loader, test_loader = get_cifar100_data_loaders(args.dataset_path, download=True, batch_size=args.bs, shuffle=True)
        num_classes = 100
    elif args.dataset_name == 'stl10':
        train_loader, test_loader = get_stl10_data_loaders(args.dataset_path, download=True, batch_size=args.bs, shuffle=True)
    print("Dataset:", args.dataset_name)


    # freeze all layers but the last fc
    for name, param in model.named_parameters():
        if name not in ['fc.weight', 'fc.bias']:
            param.requires_grad = False
    print(args)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # save config file
    save_config_file(writer.log_dir, args)

    n_iter_test = 0
    logging.info(f"Training with gpu: {args.disable_cuda}.")

    attack_kwargs = {
            'eps': args.eps,
            'step_size': 1,
            'iterations': 3
            #'bypass' : 0
        }

    nat_mis = torch.zeros([args.range, 3, 32, 32])
    adv_mis = torch.zeros([args.range, 3, 32, 32])
    missed_class = torch.zeros([10])
    correct_class = torch.zeros([10])

    model.eval()

    epochs = 1
    for epoch in range(epochs):
        top1_accuracy = 0
        top5_accuracy = 0

        for counter, (x_batch, y_batch) in enumerate(test_loader):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            logits, adv_img = model(x_batch, y_batch, make_adv=True, **attack_kwargs)
            top1, top5 = accuracy(logits, y_batch, topk=(1,5))
            top1_accuracy += top1[0]
            top5_accuracy += top5[0]

            predictions = torch.argmax(logits, dim=1)
            for sampleno in range(x_batch.size(0)):
                if(y_batch[sampleno]!=predictions[sampleno]):
                    
                    # Store first i misclassification in tensorboard
                    for i in range(args.range):
                        if torch.equal(nat_mis[i, ...], torch.zeros([3, 32, 32])):
                            nat_mis[i] = x_batch[sampleno]
                            adv_mis[i] = adv_img[sampleno]
                            print(f"|Step {counter}|:\tReal Label: {y_batch[sampleno]}\tPredicted: {predictions[sampleno]}")
                            break
                    
                    missed_class[y_batch[sampleno]] += 1    
                else:
                    correct_class[y_batch[sampleno]] += 1

            writer.add_scalar('acc/FT/TEST_acc-Top1', top1_accuracy.item()/(counter+1), global_step=n_iter_test)
            writer.add_scalar('acc/FT/TEST_acc-Top5', top5_accuracy.item()/(counter+1), global_step=n_iter_test)
            
            # Print missclasified image and respective adversarial version
            # nat_image = make_grid(nat_mis[:args.range, ...])
            # adv_image = make_grid(adv_mis[:args.range, ...])
            writer.add_image('Misclassified Natural Image', nat_mis[i] , counter)
            writer.add_image('Misclassified Adversarial Image', adv_mis[i], counter)

            # Reset images visualization tensors
            nat_mis = torch.zeros([args.range, 3, 32, 32])
            adv_mis = torch.zeros([args.range, 3, 32, 32])

            n_iter_test += 1
    
        top1_accuracy /= (counter + 1)
        top5_accuracy /= (counter + 1)

        print(f"Missed Classes: {missed_class.t()}")
        print(f"Correct Classes: {correct_class.t()}")

        # Store loss at the end of the epoch
        print(f"Top1 Test accuracy: {top1_accuracy.item()}\tTop5 test acc: {top5_accuracy.item()}")
        logging.debug(f"Top1 Test accuracy: {top1_accuracy.item()}\tTop5 test acc: {top5_accuracy.item()}")
    
    logging.info("Training has finished.")


if __name__ == "__main__":
    main()