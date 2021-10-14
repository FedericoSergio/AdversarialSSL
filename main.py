"""
The main file, which exposes the robustness command-line tool, detailed in
:doc:`this walkthrough <../example_usa /cli_usage>`.
"""

from argparse import ArgumentParser
from datetime import date, datetime
import os
import git
import torch as ch

import cox
import cox.utils
import cox.store

try:
    from .model_utils import make_and_restore_model
    from .datasets import DATASETS
    from .train import train_model, eval_model
    from .tools import constants, helpers
    from . import defaults
    from .defaults import check_and_fill_args
except:
    raise ValueError("Make sure to run with python -m (see README.md)")


parser = ArgumentParser()
parser = defaults.add_args_to_parser(defaults.CONFIG_ARGS, parser)
parser = defaults.add_args_to_parser(defaults.MODEL_LOADER_ARGS, parser)
parser = defaults.add_args_to_parser(defaults.TRAINING_ARGS, parser)
parser = defaults.add_args_to_parser(defaults.PGD_ARGS, parser)

def main(args, store=None):
    '''Given arguments from `setup_args` and a store from `setup_store`,
    trains as a model. Check out the argparse object in this file for
    argument options.
    '''
    
    # MAKE DATASET AND LOADERS
    data_path = os.path.expandvars(args.data)
    dataset = DATASETS[args.dataset](data_path)

    train_loader, val_loader = dataset.make_loaders(args.workers,
                    args.batch_size, data_aug=bool(args.data_aug))

    train_loader = helpers.DataPrefetcher(train_loader)
    val_loader = helpers.DataPrefetcher(val_loader)
    loaders = (train_loader, val_loader)

    # MAKE MODEL
    model, checkpoint = make_and_restore_model(arch=args.arch,
            dataset=dataset, resume_path=args.resume)
    if 'module' in dir(model): model = model.module

    # Projection Head
    dim_mlp = model.model.linear.in_features
    model.model.linear = ch.nn.Sequential(ch.nn.Linear(dim_mlp, dim_mlp), ch.nn.ReLU(), model.model.linear)

    print(args)
    if args.eval_only:
        return eval_model(args, model, val_loader, store=store)
 
    # Defining custom loss     
    def custom_train_loss(logits, targ):
        labels = ch.cat([ch.arange(args.batch_size) for i in range(args.n_views)], dim=0)

        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.cuda()

        features = ch.nn.functional.normalize(logits, dim=1)

        similarity_matrix = ch.matmul(features, features.T)
        # assert similarity_matrix.shape == (
        #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
        # assert similarity_matrix.shape == labels.shape
            

        # discard the main diagonal from both: labels and similarities matrix
        mask = ch.eye(labels.shape[0], dtype=ch.bool).cuda()
        labels = labels[~mask].view(labels.shape[0], -1)
        #print(similarity_matrix.shape[0])
        #print(labels.shape[0])
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        #print(similarity_matrix.shape)
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = ch.cat([positives, negatives], dim=1)
        labels = ch.zeros(logits.shape[0], dtype=ch.long).cuda()

        logits = logits / args.temperature

        loss_fun = ch.nn.CrossEntropyLoss()
        loss = loss_fun(logits, labels)
        
        return loss, logits

    custom_adv_loss = custom_train_loss

    args.custom_train_loss = custom_train_loss
    args.custom_adv_loss = custom_adv_loss

    if not args.resume_optimizer: checkpoint = None
    model = train_model(args, model, loaders, store=store,
                                    checkpoint=checkpoint)
    return model

def setup_args(args):
    '''
    Fill the args object with reasonable defaults from
    :mod:`robustness.defaults`, and also perform a sanity check to make sure no
    args are missing.
    '''
    # override non-None values with optional config_path
    if args.config_path:
        args = cox.utils.override_json(args, args.config_path)

    ds_class = DATASETS[args.dataset]
    args = check_and_fill_args(args, defaults.CONFIG_ARGS, ds_class)

    if not args.eval_only:
        args = check_and_fill_args(args, defaults.TRAINING_ARGS, ds_class)

    if args.adv_train or args.adv_eval:
        args = check_and_fill_args(args, defaults.PGD_ARGS, ds_class)

    args = check_and_fill_args(args, defaults.MODEL_LOADER_ARGS, ds_class)
    if args.eval_only: assert args.resume is not None, \
            "Must provide a resume path if only evaluating"
    return args

def setup_store_with_metadata(args):
    '''
    Sets up a store for training according to the arguments object. See the
    argparse object above for options.
    '''

    if (args.exp_name == None):
        if (args.adv_train == 1):
            args.exp_name = "[" + str(args.dataset) + "]_BS=" + str(args.batch_size) + "_LR=" + str(args.lr) + "_eps=" + str(args.eps) #+ "_bypass=" + str(args.bypass)
        else:
            args.exp_name = "[" + str(args.dataset) + "]_BS=" + str(args.batch_size) + "_LR=" + str(args.lr)
        #args.exp_name = date.today().strftime("%B_%d_%Y" + datetime.now().strftime("-%H:%M:%S-") + str(args.dataset))

    # Create the store
    store = cox.store.Store(args.out_dir, args.exp_name)
    args_dict = args.__dict__
    schema = cox.store.schema_from_dict(args_dict)
    store.add_table('metadata', schema)
    store['metadata'].append_row(args_dict)

    return store

if __name__ == "__main__":
    args = parser.parse_args()
    args = cox.utils.Parameters(args.__dict__)

    args = setup_args(args)
    store = setup_store_with_metadata(args)

    final_model = main(args, store=store)