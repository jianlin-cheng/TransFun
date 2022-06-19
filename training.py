import math
import os
import numpy as np
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torchviz import make_dot

import Constants
import params
import wandb

from Dataset.Dataset import load_dataset
from Sampler.ImbalancedDatasetSampler import ImbalancedDatasetSampler
from models.gnn import GCN  # myGCN  # ,# GAT GCN,

import argparse
import torch
import time
from torch_geometric.loader import DataLoader
import pandas as pd
from collections import Counter
from preprocessing.utils import pickle_save, pickle_load, save_ckp, load_ckp, class_distribution_counter

import warnings

warnings.filterwarnings("ignore", category=UserWarning)

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["WANDB_API_KEY"] = "b155b6571149501f01b9790e27f6ddac80ae09b3"
os.environ["WANDB_MODE"] = "online"

parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden1', type=int, default=1000, help='Number of hidden units.')
parser.add_argument('--hidden2', type=int, default=1000, help='Number of hidden units.')
parser.add_argument('--hidden3', type=int, default=1000, help='Number of hidden units.')
parser.add_argument('--train_batch', type=int, default=10, help='Training batch size.')
parser.add_argument('--valid_batch', type=int, default=5, help='Validation batch size.')
parser.add_argument('--dropout', type=float, default=0., help='Dropout rate (1 - keep probability).')
parser.add_argument('--seq', type=float, default=0.9, help='Sequence Identity (Sequence Identity).')
parser.add_argument("--ont", default='biological_process', type=str, help='Ontology under consideration')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
if args.cuda:
    device = 'cuda'

kwargs = {
    'seq_id': args.seq,
    'ont': args.ont,
    'session': 'train'
}

if args.ont == 'molecular_function':
    ont_kwargs = params.mol_kwargs
elif args.ont == 'cellular_component':
    ont_kwargs = params.cc_kwargs
elif args.ont == 'biological_process':
    ont_kwargs = params.bio_kwargs

# wandb.init(project="transfun_{}".format(args.ont), entity='frimpz',
#            name="{}_{}".format(args.seq, ont_kwargs['edge_type']))

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


def create_class_weights(cnter):
    class_weight_path = Constants.ROOT + "{}/{}/class_weights".format(kwargs['seq_id'], kwargs['ont'])
    if os.path.exists(class_weight_path + ".pickle"):
        print("Loading class weights")
        class_weights = pickle_load(class_weight_path)
    else:
        print("Generating class weights")
        go_terms = pickle_load(Constants.ROOT + "/go_terms")
        terms = go_terms['GO-terms-{}'.format(args.ont)]  # [:600]
        class_weights = [cnter[i] for i in terms]
        pickle_save(class_weights, class_weight_path)

    total = sum(class_weights)  # /100
    # class_weights = torch.tensor([total - i for i in class_weights], dtype=torch.float).to(device)
    class_weights = torch.tensor([total / i for i in class_weights], dtype=torch.float).to(device)

    return class_weights


#########################################################
# Creating training data #
#########################################################

class_weights = create_class_weights(class_distribution_counter(**kwargs))

dataset = load_dataset(root=Constants.ROOT, **kwargs)
# ct = 0
# for i in dataset:
#     for k in i:
#         if k[0] == 'sequence_letters' or k[0] == 'protein':
#             pass
#         else:
#             print(k[0])
#             print(k[1].shape)
#     ct = ct +1
#     if ct >10:
#         break
train_dataloader = DataLoader(dataset,
                              batch_size=2,
                              drop_last=True
                              # sampler=ImbalancedDatasetSampler(dataset, **kwargs, device=device))
                              , shuffle=True,
                              exclude_keys=['dist_3', 'dist_3_edge_attr',
                                            'cbrt', 'dist_4', 'dist_6',
                                            'dist_10', 'dist_12', 'sqrt_edge_attr',
                                            'cbrt_edge_attr', 'dist_4_edge_attr'])

for i in train_dataloader:
    print(i.batch)
    exit()


kwargs['session'] = 'valid'
val_dataset = load_dataset(root=Constants.ROOT, **kwargs)

valid_dataloader = DataLoader(val_dataset,
                              batch_size=args.valid_batch,
                              drop_last=True,
                              shuffle=True)

print('========================================')
print(f'# training proteins: {len(dataset)}')
print(f'# validation proteins: {len(val_dataset)}')
print('========================================')

num_class = len(pickle_load(Constants.ROOT + 'go_terms')[f'GO-terms-{args.ont}'])

current_epoch = 1
min_val_loss = np.Inf

model = GCN(input_features=dataset.num_features, **ont_kwargs)

model.to(device)
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.01)
criterion = torch.nn.BCELoss(reduction='none')


# def draw_architecture():
#     batch = next(iter(train_dataloader)).to(device)
#     output = model(batch)
#     make_dot(output, params=dict(model.named_parameters())).render("rnn_lstm_torchviz", format="png")
# draw_architecture()
#
# exit()

# def print_architecture():
#     print(model)
# print_architecture()
#
# exit()

def train(start_epoch, min_val_loss, model, optimizer, criterion, data_loader):
    min_val_loss = min_val_loss

    for epoch in range(start_epoch, args.epochs):
        print(" ---------- Epoch {} ----------".format(epoch))
        # initialize variables to monitor training and validation loss
        epoch_loss, epoch_precision, epoch_recall, epoch_accuracy, epoch_f1 = 0.0, 0.0, 0.0, 0.0, 0.0
        val_loss, val_precision, val_recall, val_accuracy, val_f1 = 0.0, 0.0, 0.0, 0.0, 0.0

        t = time.time()
        with torch.autograd.set_detect_anomaly(True):

            ###################
            # train the model #
            ###################
            model.train()
            for data in data_loader['valid']:
                print(data)
                exit()
                optimizer.zero_grad()
                output = model(data.to(device))

                loss = criterion(output, getattr(data, args.ont))
                # loss = loss.mean()
                loss = (loss * class_weights).mean()

                loss.backward()
                optimizer.step()

                epoch_loss += loss.data.item()
                epoch_accuracy += accuracy_score(getattr(data, args.ont).cpu(), output.cpu() > 0.5)
                epoch_precision += precision_score(getattr(data, args.ont).cpu(), output.cpu() > 0.5,
                                                   average="samples")
                epoch_recall += recall_score(getattr(data, args.ont).cpu(), output.cpu() > 0.5,
                                             average="samples")
                epoch_f1 += f1_score(getattr(data, args.ont).cpu(), output.cpu() > 0.5, average="samples")

            epoch_accuracy = epoch_accuracy / len(loaders['train'])
            epoch_precision = epoch_precision / len(loaders['train'])
            epoch_recall = epoch_recall / len(loaders['train'])
            epoch_f1 = epoch_f1 / len(loaders['train'])

            ###################
            # Validate the model #
            ###################

            model.eval()
            for data in data_loader['valid']:
                output = model(data.to(device))

                _val_loss = criterion(output, getattr(data, args.ont))
                _val_loss = (_val_loss * class_weights).mean()
                # _val_loss = _val_loss.mean()
                val_loss += _val_loss.data.item()
                val_accuracy += accuracy_score(getattr(data, args.ont).cpu(), output.cpu() > 0.5)
                val_precision += precision_score(getattr(data, args.ont).cpu(), output.cpu() > 0.5,
                                                 average="samples")
                val_recall += recall_score(getattr(data, args.ont).cpu(), output.cpu() > 0.5,
                                           average="samples")
                val_f1 += f1_score(getattr(data, args.ont).cpu(), output.cpu() > 0.5, average="samples")

            val_loss = val_loss / len(loaders['valid'])
            val_accuracy = val_accuracy / len(loaders['valid'])
            val_precision = val_precision / len(loaders['valid'])
            val_recall = val_recall / len(loaders['valid'])
            val_f1 = val_f1 / len(loaders['valid'])

            print('Epoch: {:04d}'.format(epoch),
                  'train_loss: {:.4f}'.format(epoch_loss),
                  'train_acc: {:.4f}'.format(epoch_accuracy),
                  'precision: {:.4f}'.format(epoch_precision),
                  'recall: {:.4f}'.format(epoch_recall),
                  'f1: {:.4f}'.format(epoch_f1),
                  'val_acc: {:.4f}'.format(val_accuracy),
                  'val_loss: {:.4f}'.format(val_loss),
                  'val_precision: {:.4f}'.format(val_precision),
                  'val_recall: {:.4f}'.format(val_recall),
                  'val_f1: {:.4f}'.format(val_f1),
                  'time: {:.4f}s'.format(time.time() - t))

            wandb.log({"train_acc": epoch_accuracy,
                       "train_loss": epoch_loss,
                       "precision": epoch_precision,
                       "recall": epoch_recall,
                       "f1": epoch_f1,
                       "val_acc": val_accuracy,
                       "val_loss": val_loss,
                       "val_precision": val_precision,
                       "val_recall": val_recall,
                       "val_f1": val_f1})

            checkpoint = {
                'epoch': epoch,
                'valid_loss_min': val_loss,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }

            # save checkpoint
            save_ckp(checkpoint, False, ckp_pth,
                     ckp_dir + "best_model.pt")

            if val_loss <= min_val_loss:
                print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'. \
                      format(min_val_loss, val_loss))

                # save checkpoint as best model
                save_ckp(checkpoint, True, ckp_pth,
                     ckp_dir + "best_model.pt")
                min_val_loss = val_loss

    return model


loaders = {
    'train': train_dataloader,
    'valid': valid_dataloader
}

ckp_dir = Constants.ROOT + '{}/{}/model_checkpoint/{}/'.format(args.seq, args.ont, ont_kwargs['edge_type'])
ckp_pth = ckp_dir + "current_checkpoint.pt"
if os.path.exists(ckp_pth):
    print("Loading model checkpoint")
    model, optimizer, current_epoch, min_val_loss = load_ckp(ckp_pth, model, optimizer)
else:
    if not os.path.exists(ckp_dir):
        os.makedirs(ckp_dir)

print("Training model on epoch {}, with minimum validation loss as {}".format(current_epoch, min_val_loss))

wandb.config = {
    "learning_rate": args.lr,
    "epochs": current_epoch,
    "batch_size": args.train_batch
}

trained_model = train(current_epoch, min_val_loss,
                      model=model, optimizer=optimizer,
                      criterion=criterion, data_loader=loaders)

# exit()
#
#
# def test(loader):
#     model.eval()
#     correct = 0
#     for data in train_dataloader:
#         out = model(data)
#         pred = out.argmax(dim=1)
#         correct += int((pred == data.molecular_function).sum())  # Check against ground-truth labels.
#     return correct / len(loader.dataset)  # Derive ratio of correct predictions.
