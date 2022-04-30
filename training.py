import os

import numpy as np
import torch.optim as optim
import torchmetrics
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

import Constants
import wandb

from Dataset.Dataset import load_dataset
from models.gnn import GCN
import argparse
import torch
from torch.utils.tensorboard import SummaryWriter
import time
from torch.utils.data.sampler import SubsetRandomSampler
from torch_geometric.loader import DataLoader
import pandas as pd
from collections import Counter

from preprocessing.utils import pickle_save, pickle_load, save_ckp

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

os.environ["WANDB_API_KEY"] = "b155b6571149501f01b9790e27f6ddac80ae09b3"
os.environ["WANDB_MODE"] = "online"
wandb.init(project="frimpong")


parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden1', type=int, default=512, help='Number of hidden units.')
parser.add_argument('--hidden2', type=int, default=64, help='Number of hidden units.')
parser.add_argument('--hidden3', type=int, default=32, help='Number of hidden units.')
parser.add_argument('--nclass', type=int, default=8, help='Number of clusters.')
parser.add_argument('--dropout', type=float, default=0., help='Dropout rate (1 - keep probability).')

device = 'cpu'
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
if args.cuda:
    device = 'cuda'

# writer = SummaryWriter()
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


kwargs = {
    'seq_id': 0.3,
    'ont': 'molecular_function',
    'session': 'train'
}

# data = pickle_load(Constants.ROOT + "{}/{}/{}".format(kwargs['seq_id'], kwargs['ont'], kwargs['session']))

# all_proteins = []
# for i in data:
#     all_proteins.extend(data[i])
#
# print(len(all_proteins))
#
# annot = pd.read_csv(Constants.ROOT + 'annot.tsv', delimiter='\t')
# annot = annot.where(pd.notnull(annot), None)
# annot = annot[annot['Protein'].isin(all_proteins)]
# annot = pd.Series(annot[kwargs['ont']].values, index=annot['Protein']).to_dict()
#
# terms = []
# for i in annot:
#     terms.extend(annot[i].split(","))
#
# counter = Counter(terms)
#
# print(counter, len(counter))
#
#
# exit()
dataset = load_dataset(root=Constants.ROOT, **kwargs)

class_weight_path = Constants.ROOT + "{}/{}/class_weights".format(kwargs['seq_id'], kwargs['ont'])
if os.path.exists(class_weight_path+".pickle"):
    print("Loading class weights")
    class_weights = pickle_load(class_weight_path)
else:
    print("Generating class weights")
    lab = []
    for i in dataset:
        if kwargs['ont'] == 'molecular_function':
            lab.append(i.molecular_function)
        elif kwargs['ont'] == 'biological_process':
            lab.append(i.biological_process)
        elif kwargs['ont'] == 'cellular_component':
            lab.append(i.cellular_component)
        elif kwargs['ont'] == 'all':
            lab.append(i.all)

    result = torch.sum(torch.stack(lab), dim=0)
    result = result.to(torch.int)
    class_weights = {i: result[0][i].item() for i in range(result.size(1))}
    pickle_save(class_weights, class_weight_path)


class_weights = list(class_weights.values())
total = sum(class_weights)
# print(total)
# total = 680 * len(dataset)

class_weights = [total/i for i in class_weights]
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
# weights = 1 / (weights / torch.min(weights))
train_dataloader = DataLoader(dataset, batch_size=40, drop_last=False, shuffle=True)


kwargs = {
    'seq_id': 0.3,
    'ont': 'molecular_function',
    'session': 'valid'
}
val_dataset = load_dataset(root='/data/pycharm/TransFunData/data/', **kwargs)
valid_dataloader = DataLoader(val_dataset, batch_size=20, drop_last=False, shuffle=True)


# print(f'Dataset: {dataset}:')
# print(f'Dataset: {dataset[0]}:')
# print('====================')
# print(f'Number of graphs: {len(dataset)}')
# print(f'Number of features: {dataset.num_features}')

args.nclass = dataset[0].molecular_function.shape[1]
model = GCN(input_features=dataset.num_features, hidden_channels_1=args.hidden1, hidden_channels_2=args.hidden2,
            hidden_channels_3=args.hidden3, num_classes=args.nclass)
model.to(device)


optimizer = optim.Adam(model.parameters(), lr=args.lr)#, weight_decay=args.weight_decay)
acc = torchmetrics.Accuracy().to(device)
prec = torchmetrics.Precision()
recall = torchmetrics.Recall()
f1score = torchmetrics.F1()

# criterion = torch.nn.CrossEntropyLoss()
criterion = torch.nn.BCEWithLogitsLoss(reduction='none')
criterion = torch.nn.BCELoss(reduction='none')

valid_loss_min = np.Inf


def train(start_epoch, valid_loss_min_input):

    valid_loss_min = valid_loss_min_input

    for epoch in range(start_epoch, args.epochs):

        # initialize variables to monitor training and validation loss
        epoch_loss, epoch_precision, epoch_recall, epoch_accuracy, epoch_f1, train_batches = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        val_loss, val_precision, val_recall, val_accuracy, val_f1, val_batches = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        t = time.time()
        with torch.autograd.set_detect_anomaly(True):

            ###################
            # train the model #
            ###################
            model.train()
            for batch_idx, data in enumerate(train_dataloader):

                print(batch_idx, data)
                continue

                optimizer.zero_grad()
                output = model(data.to(device))

                loss = criterion(output, data.molecular_function)
                loss = (loss * class_weights).mean()

                loss.backward()
                optimizer.step()

                epoch_loss += loss.data.item()
                epoch_accuracy += accuracy_score(data.molecular_function.cpu(), output.cpu() > 0.5)
                epoch_precision += precision_score(data.molecular_function.cpu(), output.cpu() > 0.5, average="samples")
                epoch_recall += recall_score(data.molecular_function.cpu(), output.cpu() > 0.5, average="samples")
                epoch_f1 += f1_score(data.molecular_function.cpu(), output.cpu() > 0.5, average="samples")
                mini_batches += 1

            epoch_loss = epoch_loss/mini_batches
            epoch_accuracy = epoch_accuracy / mini_batches
            epoch_precision = epoch_precision / mini_batches
            epoch_recall = epoch_recall / mini_batches
            epoch_f1 = epoch_f1 / mini_batches

            print('Epoch: {:04d}'.format(epoch),
                  'train_loss: {:.4f}'.format(epoch_loss),
                  'train_acc: {:.4f}'.format(epoch_accuracy),
                  'precision: {:.4f}'.format(epoch_precision),
                  'recall: {:.4f}'.format(epoch_recall),
                  'f1: {:.4f}'.format(epoch_f1),
                  'time: {:.4f}s'.format(time.time() - t))

            wandb.log({"train_acc": epoch_accuracy,
                       "train_loss": epoch_loss,
                       "precision": epoch_precision,
                       "recall": epoch_recall,
                       "f1": epoch_f1})

            print(" ---------- EVALUATE ON VALIDATION SET ----------")
            mini_batches = 0.0
            model.eval()
            for data in valid_dataloader:

                output = model(data.to(device))

                _val_loss = criterion(output, data.molecular_function)
                _val_loss = (_val_loss * class_weights).mean()

                val_loss += _val_loss.data.item()
                val_accuracy += accuracy_score(data.molecular_function.cpu(), output.cpu() > 0.5)
                val_precision += precision_score(data.molecular_function.cpu(), output.cpu() > 0.5, average="samples")
                val_recall += recall_score(data.molecular_function.cpu(), output.cpu() > 0.5, average="samples")
                val_f1 += f1_score(data.molecular_function.cpu(), output.cpu() > 0.5, average="samples")
                mini_batches += 1

            val_loss = val_loss / mini_batches
            val_accuracy = val_accuracy / mini_batches
            val_precision = val_precision / mini_batches
            val_recall = val_recall / mini_batches
            val_f1 = val_f1 / mini_batches

            print('Epoch: {:04d}'.format(epoch),
                  'val_acc: {:.4f}'.format(val_accuracy),
                  'val_loss: {:.4f}'.format(val_loss),
                  'val_precision: {:.4f}'.format(val_precision),
                  'val_recall: {:.4f}'.format(val_recall),
                  'val_f1: {:.4f}'.format(val_f1),
                  'time: {:.4f}s'.format(time.time() - t))

            wandb.log({"val_acc": val_accuracy, "val_loss": val_loss,
                       "val_precision": val_precision, "val_recall": val_recall,
                       "val_f1": val_f1})

            checkpoint = {
                'epoch': epoch + 1,
                'valid_loss_min': val_loss,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }

            # # save checkpoint
            save_ckp(checkpoint, False, Constants.ROOT + 'model_checkpoint',
                     Constants.ROOT + 'best_model')

            ## TODO: save the model if validation loss has decreased
            if val_loss <= valid_loss_min:
                print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min, valid_loss))
                # save checkpoint as best model
                save_ckp(checkpoint, True, Constants.ROOT + 'model_checkpoint',
                         Constants.ROOT + 'best_model')
                valid_loss_min = val_loss


def test(loader):
    model.eval()

    correct = 0
    for data in train_dataloader:
        out = model(data)
        pred = out.argmax(dim=1)
        correct += int((pred == data.molecular_function).sum())  # Check against ground-truth labels.
    return correct / len(loader.dataset)  # Derive ratio of correct predictions.

train(1, np.Inf)