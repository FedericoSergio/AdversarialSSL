from pyexpat import features
from numpy.lib.npyio import savetxt
from models.make_adv import make_adv
import torch
import torch.backends.cudnn as cudnn
from torchvision import models
import argparse
import numpy as np
import os
import yaml
import matplotlib.pyplot as plt
import torchvision
import sklearn.metrics as mtcs
from sklearn import decomposition
from sklearn import manifold
import skimage.io
import pandas as pd
from torchvision.models.feature_extraction import get_graph_node_names

from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import datasets
from torchvision.utils import make_grid, save_image
from torch.utils.tensorboard import SummaryWriter
import logging
from torchvision.models.feature_extraction import get_graph_node_names, create_feature_extractor

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

# parser.add_argument('--checkpoint', default='../runs/FT/[cifar10-cifar10]_BS=256_LR=8e-05_ADVModelEps=0/checkpoint_0500.pth.tar',
#                     help='Checkpoint to resume model for fine-tuning.', dest='checkpoint')
parser.add_argument('--pretrained-dataset', default='cifar100',
                    help='Name of dataset used in checkpoint model', dest='ftDataset')
parser.add_argument('--eps', '--eps', default=5, type=float,
                    metavar='EPS', help='eps to apply to test images', dest='eps')
parser.add_argument('--range', '--range', default=16, type=int,
                    metavar='RANGE', help='...', dest='range')

parser.add_argument('--perplexity', default=35, type=int,
                    metavar='perplexity', help='...', dest='perplexity')

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
    folder_name = "runs/EVAL/["+ str(args.ftDataset) + "-" + str(args.dataset_name) + "]_perplexity=" + str(args.perplexity)

    # Create folder for images
    img_path = "/home/sefe/AdversarialSSL/" + folder_name + "/images/"
    try:
        os.makedirs(os.path.dirname(img_path))
    except FileExistsError:
        print('Path is not correctly defined.')

    writer = SummaryWriter(log_dir=folder_name)
    logging.basicConfig(filename=os.path.join(writer.log_dir, 'training.log'), level=logging.DEBUG)

    if args.dataset_name == 'cifar10':
        model_2 = torchvision.models.resnet18(pretrained=False, num_classes=10).to(device)
        model_1 = torchvision.models.resnet18(pretrained=False, num_classes=10).to(device)
    elif args.dataset_name == 'cifar100':
        model_2 = torchvision.models.resnet18(pretrained=False, num_classes=100).to(device)
        model_1 = torchvision.models.resnet18(pretrained=False, num_classes=100).to(device)

    ckpt_1 = torch.load('../runs/FT/[cifar100-cifar10]_BS=256_LR=8e-05_FTeps=0/checkpoint_0100.pth.tar', map_location=device)
    state_dict_1 = ckpt_1['state_dict']

    ckpt_2 = torch.load('../runs/FT/[cifar100-cifar10]_BS=256_LR=8e-05_FTeps=1/checkpoint_0100.pth.tar', map_location=device)
    state_dict_2 = ckpt_2['state_dict']

    return_nodes = {
        'x' : 'inputs',
        'maxpool' : 'maxpool',
        'layer4.1.relu_1' : 'Layer 4',
        'avgpool' : 'avgpool',
        'fc' : 'Linear Layer',
    }

    #---------------LOADING SIMCLR WEIGHTS---------------#
    # for k in list(state_dict_1.keys()):
    #     if k.startswith('backbone.'):
    #         if k.startswith('backbone'): #and not k.startswith('backbone.fc')
    #             # remove prefix
    #             state_dict_1[k[len("backbone."):]] = state_dict_1[k]
    #     del state_dict_1[k]
    
    log = model_1.load_state_dict(state_dict_1, strict=False)
    #assert log.missing_keys == ['fc.weight', 'fc.bias']

    nodes, _ = get_graph_node_names(model_1)

    feature_extractor_1 = create_feature_extractor(model_1, return_nodes=return_nodes).to(device)
    #----------------------------------------------------#

    #---------------LOADING LINEAR WEIGHTS---------------#
    log = model_2.load_state_dict(state_dict_2, strict=True)

    nodes, _ = get_graph_node_names(model_2)

    feature_extractor_2 = create_feature_extractor(model_2, return_nodes=return_nodes).to(device)
    #----------------------------------------------------#

    model_2.eval()
    model_1.eval()

    if (torch.equal(feature_extractor_1(torch.zeros(2, 3, 32, 32).to(device))['avgpool'], feature_extractor_2(torch.zeros(2, 3, 32, 32).to(device))['avgpool'])):
        print('ALCUNI LAYER SONO IDENTICI..')

    if args.dataset_name == 'cifar10':
        train_loader, test_loader = get_cifar10_data_loaders(args.dataset_path, download=True, batch_size=args.bs, shuffle=True)
        num_classes = 10
        classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    elif args.dataset_name == 'cifar100':
        train_loader, test_loader = get_cifar100_data_loaders(args.dataset_path, download=True, batch_size=args.bs, shuffle=True)
        num_classes = 100
        classes = [
                'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
                'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
                'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
                'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
                'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
                'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
                'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
                'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
                'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
                'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
                'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
                'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
                'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
                'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman',
                'worm'
            ]
    elif args.dataset_name == 'stl10':
        train_loader, test_loader = get_stl10_data_loaders(args.dataset_path, download=True, batch_size=args.bs, shuffle=True)
    print("Dataset:", args.dataset_name)
    
    # save config file
    save_config_file(writer.log_dir, args)

    logging.info(f"Training with gpu: {args.disable_cuda}.")

    attack_kwargs = {
            'eps': args.eps,
            'step_size': 1,
            'iterations': 20,
            #'bypass' : 0
        }

    f_task_1 = torch.zeros(10000)
    f_task_2 = torch.ones(10000)
    f_storer_1 = [(), (), (), (), ()]
    f_storer_2 = [(), (), (), (), ()]

    for counter, (x_batch, y_batch) in enumerate(test_loader):
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        adv_img = make_adv(model_2, x_batch, y_batch, **attack_kwargs)

        with torch.no_grad():
            logits = model_2(adv_img)

            # Store data for feature evaluation
            out_1 = feature_extractor_1(adv_img)
            
            feature_1_l1 = out_1['inputs'].reshape(out_1['inputs'].size(0), -1)
            f_storer_1[0] = f_storer_1[0] + (feature_1_l1,)
            feature_1_l2 = out_1['maxpool'].reshape(out_1['maxpool'].size(0), -1)
            f_storer_1[1] = f_storer_1[1] + (feature_1_l2,)
            feature_1_l3 = out_1['Layer 4'].reshape(out_1['Layer 4'].size(0), -1)
            f_storer_1[2] = f_storer_1[2] + (feature_1_l3,)
            feature_1_l4 = out_1['avgpool'].reshape(out_1['avgpool'].size(0), -1)
            f_storer_1[3] = f_storer_1[3] + (feature_1_l4,)
            feature_1_fc = out_1['Linear Layer'].reshape(out_1['Linear Layer'].size(0), -1)
            f_storer_1[4] = f_storer_1[4] + (feature_1_fc,)

            out_2 = feature_extractor_2(adv_img)

            feature_2_l1 = out_2['inputs'].reshape(out_2['inputs'].size(0), -1)
            f_storer_2[0] = f_storer_2[0] + (feature_2_l1,)
            feature_2_l2 = out_2['maxpool'].reshape(out_2['maxpool'].size(0), -1)
            f_storer_2[1] = f_storer_2[1] + (feature_2_l2,)
            feature_2_l3 = out_2['Layer 4'].reshape(out_2['Layer 4'].size(0), -1)
            f_storer_2[2] = f_storer_2[2] + (feature_2_l3,)
            feature_2_l4 = out_2['avgpool'].reshape(out_2['avgpool'].size(0), -1)
            f_storer_2[3] = f_storer_2[3] + (feature_2_l4,)
            feature_2_fc = out_2['Linear Layer'].reshape(out_2['Linear Layer'].size(), -1)
            f_storer_2[4] = f_storer_2[4] + (feature_2_fc,)

    for i in range(len(f_storer_1)):
        f_storer_1[i] = torch.cat(f_storer_1[i], dim=0).cpu()
        f_storer_2[i] = torch.cat(f_storer_2[i], dim=0).cpu()

    # store t-SNE
    tasks = ['simclr', 'linear']
    for i in range(len(f_storer_1)):
    
        print(f_storer_1[i].size())
        print(f_storer_2[i].size())
        output_tsne_data_1 = get_tsne(f_storer_1[i], perplexity=args.perplexity)
        output_tsne_data_2 = get_tsne(f_storer_2[i], perplexity=args.perplexity)
        plot_representations(output_tsne_data_1, f_task_1, classes=tasks, img_path=img_path, filename='t-SNE1_'+list(return_nodes.values())[i] + '.png', plotname='t-SNE in '+list(return_nodes.values())[i]+' of [' +str(args.ftDataset) + '-' + str(args.dataset_name) + ']')
        plot_representations(output_tsne_data_2, f_task_2, classes=tasks, img_path=img_path, filename='t-SNE2_'+list(return_nodes.values())[i] + '.png', plotname='t-SNE in '+list(return_nodes.values())[i]+' of [' +str(args.ftDataset) + '-' + str(args.dataset_name) + ']')

    logging.info("Test has finished.")

def plot_intensity_hist(img_path, filename, plot_name):
    fig, ax = plt.subplots()
    im = skimage.io.imread(fname=(img_path + filename))
    # tuple to select colors of each channel line
    colors = ("red", "green", "blue")
    channel_ids = (0, 1, 2)

    # create the histogram plot, with three lines, one for
    # each color
    plt.xlim([0, 256])
    for channel_id, c in zip(channel_ids, colors):
        histogram, bin_edges = np.histogram(
            im[:, :, channel_id], bins=256, range=(0, 256)
        )
        plt.plot(bin_edges[0:-1], histogram, color=c)
    plt.title("RGB Intensity")
    plt.ylabel("Pixel Intensity")
    plt.xlabel("Pixel Number")
    plt.grid()
    #plt.savefig(os.path.join(img_path, '{}.png'.format('Image Intensity')))
    fig.savefig(img_path + plot_name)
    plt.close(fig)

def plot_conf_matrix(img_path, y_true, y_pred, classes):
    # Make confusion matrix with sklearn
    fig, ax = plt.subplots(figsize=(8, 5))
    cm = mtcs.ConfusionMatrixDisplay(mtcs.confusion_matrix(y_true, y_pred), display_labels=classes)
    cm.plot(ax=ax)
    #plt.savefig(os.path.join(img_path, '{}.png'.format('confusion_matrix')))
    fig.savefig(img_path + 'confusion_matrix.png')
    plt.close(fig)

def get_pca(data, n_components = 2):
    pca = decomposition.PCA()
    pca.n_components = n_components
    pca_data = pca.fit_transform(data)
    return pca_data

def plot_representations(data, labels, img_path, filename, plotname, classes=None, n_images=None):
            
    if n_images is not None:
        data = data[:n_images]
        labels = labels[:n_images]
                
    fig = plt.figure(figsize = (15, 15))
    plt.grid()
    ax = fig.add_subplot(111)

    scatter = ax.scatter(data[:, 0], data[:, 1], c = labels, cmap = 'hsv')
    y = np.unique(labels)
    handles = [plt.Line2D([],[],marker="o", ls="", 
                        color=scatter.cmap(scatter.norm(yi))) for yi in y]
    plt.legend(handles, classes)
    ax.set_title(str(plotname))
    fig.savefig(img_path + filename)

def get_tsne(data, n_components = 2, perplexity = 30, n_images = None):
    
    if n_images is not None:
        data = data[:n_images]
        
    tsne = manifold.TSNE(n_components = n_components, perplexity = perplexity, n_iter = 1500, early_exaggeration=48, random_state = 0)
    tsne_data = tsne.fit_transform(data)
    return tsne_data

if __name__ == "__main__":
    main()
