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

from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import datasets
from torchvision.utils import make_grid, save_image
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

# parser.add_argument('--checkpoint', default='../runs/FT/[cifar10-cifar10]_BS=256_LR=8e-05_ADVModelEps=0/checkpoint_0500.pth.tar',
#                     help='Checkpoint to resume model for fine-tuning.', dest='checkpoint')
parser.add_argument('--pretrained-dataset', default='cifar10',
                    help='Name of dataset used in checkpoint model', dest='ftDataset')
# parser.add_argument('--eps', '--eps', default=3, type=float,
#                     metavar='EPS', help='eps to apply to test images', dest='eps')
parser.add_argument('--range', '--range', default=16, type=int,
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

    print(args)

    checkpoint_list = ['../runs/FT/[cifar10-cifar10]_BS=256_LR=8e-05_ADVModelEps=0/checkpoint_0500.pth.tar',
                        '../runs/FT/[cifar10-cifar10]_BS=256_LR=8e-05_ADVModelEps=0.1/checkpoint_0500.pth.tar',
                        '../runs/FT/[cifar10-cifar10]_BS=256_LR=8e-05_ADVModelEps=0.2/checkpoint_0500.pth.tar',
                        '../runs/FT/[cifar10-cifar10]_BS=256_LR=8e-05_ADVModelEps=0.3/checkpoint_0500.pth.tar',
                        '../runs/FT/[cifar10-cifar10]_BS=256_LR=8e-05_ADVModelEps=0.4/checkpoint_0500.pth.tar',
                        '../runs/FT/[cifar10-cifar10]_BS=256_LR=8e-05_ADVModelEps=0.5/checkpoint_0500.pth.tar' 
                        ]

    eps_list = [0, 0.1, 0.2, 0.3, 0.4, 0.5]

    header = ['eps=0', 'eps=0.1', 'eps=0.2', 'eps=0.3', 'eps=0.4', 'eps=0.5']
    acc_table = np.array(header)

    for checkpoint in range(len(checkpoint_list)):

        acc_row = np.zeros(len(eps_list))

        for eps in range(len(eps_list)):

            # Initialize writer for Tensorboard
            folder_name = "runs/EVAL/["+ str(args.ftDataset) + "-" + str(args.dataset_name) +  "]_BS=" + str(args.bs) + "_eps=" + str(eps_list[eps])

            # Create folder for images
            img_path = "/home/sefe/AdversarialSSL/" + folder_name + "/images/" 
            os.makedirs(os.path.dirname(img_path))

            writer = SummaryWriter(log_dir=folder_name)
            logging.basicConfig(filename=os.path.join(writer.log_dir, 'training.log'), level=logging.DEBUG)

            if args.dataset_name == 'cifar10':
                model = torchvision.models.resnet18(pretrained=False, num_classes=10).to(device)
            elif args.dataset_name == 'cifar100':
                model = torchvision.models.resnet18(pretrained=False, num_classes=100).to(device)
            

            ckpt = torch.load(checkpoint_list[checkpoint], map_location=device)
            state_dict = ckpt['state_dict']

            # # If ckpt is not fine tuned
            # if not ckpt.startswith('../runs/FT'):
            #     for k in list(state_dict.keys()):
            #         if k.startswith('backbone.'):
            #             if k.startswith('backbone') and not k.startswith('backbone.fc'):
            #                 # remove prefix
            #                 state_dict[k[len("backbone."):]] = state_dict[k]
            #         del state_dict[k]

            # print("Model's state_dict:")
            # for param_tensor in state_dict:
            #     print(param_tensor, "\t", state_dict[param_tensor].size())
                
            log = model.load_state_dict(state_dict, strict=True)
            model.eval()

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

            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            
            # save config file
            save_config_file(writer.log_dir, args)

            logging.info(f"Training with gpu: {args.disable_cuda}.")

            attack_kwargs = {
                    'eps': eps_list[eps],
                    'step_size': 1,
                    'iterations': 3
                    #'bypass' : 0
                }

            nat_mis = torch.zeros([args.range, 3, 32, 32])
            adv_mis = torch.zeros([args.range, 3, 32, 32])
            missed_class = torch.zeros([num_classes]).to(torch.int)
            correct_class = torch.zeros([num_classes]).to(torch.int)
            confusion_matrix_t = torch.zeros([num_classes, num_classes]).to(torch.int)
            y_true = torch.zeros(10000)
            y_pred = torch.zeros(10000)
            logits_storer = []

            model.eval()
            
            top1_accuracy = 0
            top5_accuracy = 0
            
            k = 0
            for counter, (x_batch, y_batch) in enumerate(test_loader):
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)

                adv_img = make_adv(model, x_batch, y_batch, **attack_kwargs)

                with torch.no_grad():
                    logits = model(adv_img)

                    # Store data for evaluation
                    logits_storer.append(logits.cpu())

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
                                #print(f"|Step {counter}|:\tReal Label: {classes[y_batch[sampleno]]}\tPredicted: {classes[predictions[sampleno]]}")
                                break
                        missed_class[y_batch[sampleno]] += 1    
                    else:
                        correct_class[y_batch[sampleno]] += 1
                    
                    # MATRIX([true][predicted])
                    confusion_matrix_t[y_batch[sampleno]][predictions[sampleno]] += 1
                    y_true[k] = y_batch[sampleno]
                    y_pred[k] = predictions[sampleno]
                    k += 1
                
                # Print missclasified images and respective adversarial versions
                nat_grid = make_grid(nat_mis[:args.range, ...])
                adv_grid = make_grid(adv_mis[:args.range, ...])
                writer.add_image('Misclassified Natural Image', nat_grid , counter)
                writer.add_image('Misclassified Adversarial Image', adv_grid, counter)

                if counter == 0:
                    # Store examples in folder
                    save_image(nat_grid, os.path.join(img_path, '{}.png'.format('nat_misclassifications')))
                    save_image(adv_grid, os.path.join(img_path, '{}.png'.format('adv_misclassifications')))
                    save_image(nat_mis[0], os.path.join(img_path, '{}.png'.format('nat_ex1')))
                    save_image(adv_mis[0], os.path.join(img_path, '{}.png'.format('adv_ex1')))
                    #save_image(nat_mis[1], os.path.join(img_path, '{}.png'.format('nat_ex2')))
                    #save_image(adv_mis[1], os.path.join(img_path, '{}.png'.format('adv_ex2')))

                # # Reset images visualization tensors
                # nat_mis = torch.zeros([args.range, 3, 32, 32])
                # adv_mis = torch.zeros([args.range, 3, 32, 32])

            top1_accuracy /= (counter + 1)
            top5_accuracy /= (counter + 1)
            
            logits_storer = torch.cat(logits_storer, dim=0)

            # Evaluation metrics
            precision, recall, fscore, _ = mtcs.precision_recall_fscore_support(y_true, y_pred, average='macro')
            mcc_coef = mtcs.matthews_corrcoef(y_true, y_pred)

            plot_intensity_hist(img_path, filename='nat_ex1.png', plot_name='nat_ex1_intensity_plot.png')
            plot_intensity_hist(img_path, filename='adv_ex1.png', plot_name='adv_ex1_intensity_plot.png')
            #plot_intensity_hist(img_path, filename='nat_ex2.png', plot_name='nat_ex2_plot.png')
            #plot_intensity_hist(img_path, filename='adv_ex2.png', plot_name='adv_ex2_plot.png')
            
            if (args.dataset_name == 'cifar10'):
                # store confusion matrix
                plot_conf_matrix(img_path, y_true, y_pred, classes)

                # store PCA 
                output_pca_data = get_pca(logits_storer)
                plot_representations(output_pca_data, y_true, img_path, filename='PCA.png')

                # store t-SNE
                output_tsne_data = get_tsne(logits_storer)
                plot_representations(output_tsne_data, y_true, img_path, filename='t-SNE.png')

            # separator = ' \\ '
            # np.savetxt("confusion_matrix.csv", confusion_matrix.to(torch.int), delimiter=' & ', newline=separator, fmt='%d')

            np.append(acc_row, top1_accuracy.item())

            print(confusion_matrix_t)
            print(f"Missed Classes: {missed_class.t()}")
            print(f"Correct Classes: {correct_class.t()}")

            # Store accuracies and other data at the end of the epoch
            print(f"Top1 Test accuracy: {top1_accuracy.item()} | Top5 test acc: {top5_accuracy.item()}")
            logging.debug(f"Top1 Test accuracy: {top1_accuracy.item()} | Top5 test acc: {top5_accuracy.item()}")

            print(f"Precision: {precision} | Recall: {recall} | F1_score: {fscore} | MCC: {mcc_coef}")
            logging.debug(f"Precision: {precision} | Recall: {recall} | F1_score: {fscore} | MCC: {mcc_coef}")

            print(f"Most correct label: {classes[torch.argmax(correct_class)]} | Most missed label: {classes[torch.argmax(missed_class)]}")
            logging.debug(f"Most correct label: {classes[torch.argmax(correct_class)]} | Most missed label: {classes[torch.argmax(missed_class)]}")

            # if args.dataset_name == 'cifar10': 
            #     miscassifications = missed_class.t().numpy()

            #     plt.bar(classes, miscassifications)
            #     plt.title('Prediction Errors')
            #     plt.xlabel('Missed Classes')
            #     plt.ylabel('# of Incorrect Predictions')

                
            #     plt.savefig(os.path.join(folder_name, '{}.png'.format('wrong_predictions')))
            #elif args.dataset_name == 'cifar100':
                # TODO

        np.vstack(acc_table, acc_row)
    
    separator = ' \\ '
    np.savetxt("exp1.csv", acc_table, delimiter=' & ', newline=separator, fmt='%d')
    print(acc_table)
    
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

def plot_representations(data, labels, img_path, filename, n_images = None ):
            
    if n_images is not None:
        data = data[:n_images]
        labels = labels[:n_images]
                
    fig = plt.figure(figsize = (15, 15))
    ax = fig.add_subplot(111)
    ax.scatter(data[:, 0], data[:, 1], c = labels, cmap = 'hsv')
    ax.set_title(str(filename))
    fig.savefig(img_path + filename)

def get_tsne(data, n_components = 2, n_images = None):
    
    if n_images is not None:
        data = data[:n_images]
        
    tsne = manifold.TSNE(n_components = n_components, random_state = 0)
    tsne_data = tsne.fit_transform(data)
    return tsne_data

if __name__ == "__main__":
    main()
