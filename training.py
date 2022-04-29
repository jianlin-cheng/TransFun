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

from preprocessing.utils import pickle_save, pickle_load

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

os.environ["WANDB_API_KEY"] = "b155b6571149501f01b9790e27f6ddac80ae09b3"
os.environ["WANDB_MODE"] = "online"
# wandb.init(project="frimpong")


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
    'seq_id': 0.9,
    'ont': 'molecular_function',
    'session': 'valid'
}
val_dataset = load_dataset(root='/data/pycharm/TransFunData/data/', **kwargs)
valid_dataloader = DataLoader(val_dataset, batch_size=40, drop_last=False, shuffle=True)


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


def train(epoch):
    t = time.time()
    with torch.autograd.set_detect_anomaly(True):
        train_loss, train_acc = 0.0, 0.0
        model.train()
        count = 0.
        total = 0.
        correct = 0.
        for data in train_dataloader:
            optimizer.zero_grad()

            output = model(data.to(device))

            # print(torch.sum(output, dim=0), torch.sum(output, dim=0).shape)
            # print(torch.sum(output, dim=1), torch.sum(output, dim=1).shape)
            #
            # exit()


            loss = criterion(output, data.molecular_function)
            loss = (loss * class_weights).mean()
            # writer.add_scalar("Loss/train", loss, epoch)

            loss.backward()
            optimizer.step()
            # optimizer.zero_grad()

            train_loss = loss.data.item()
            accuracy = accuracy_score(data.molecular_function.cpu(), output.cpu() > 0.5)
            precision = precision_score(data.molecular_function.cpu(), output.cpu() > 0.5, average="samples")
            recall = recall_score(data.molecular_function.cpu(), output.cpu() > 0.5, average="samples")
            f1 = f1_score(data.molecular_function.cpu(), output.cpu() > 0.5, average="samples")

            # pred = (output > 0.5).float()
            # correct += (pred == data.molecular_function).float().sum()
            #
            # count += 1
            # total += data.molecular_function.size(1) * data.molecular_function.size(0)

            print('Epoch: {:04d}'.format(epoch),
                  'train_loss: {:.4f}'.format(train_loss),
                  'train_acc: {:.4f}'.format(accuracy),
                  'precision: {:.4f}'.format(precision),
                  'recall: {:.4f}'.format(recall),
                  'f1: {:.4f}'.format(f1),
                  'time: {:.4f}s'.format(time.time() - t))

        # pass
        # train_acc = 100 * correct / total
        # train_loss = train_loss / count
        #
            # wandb.log({"train_acc": accuracy, "train_loss": train_loss,
            #            "precision": precision, "recall": recall,
            #            "f1": f1})
        #
        print("#####     Validating     #####")
        # --- EVALUATE ON VALIDATION SET -------------------------------------
        model.eval()
        val_loss, val_acc = 0.0, 0.0
        count = 0
        total = 0.
        correct = 0.
        for data in valid_dataloader:
            output = model(data.to(device))
            val_loss = criterion(output, data.molecular_function)
            val_loss = (val_loss * class_weights).mean()

            val_loss = val_loss.data.item()
            val_acc = accuracy_score(data.molecular_function.cpu(), output.cpu() > 0.5)
            val_precision = precision_score(data.molecular_function.cpu(), output.cpu() > 0.5, average="samples")
            val_recall = recall_score(data.molecular_function.cpu(), output.cpu() > 0.5, average="samples")
            val_f1 = f1_score(data.molecular_function.cpu(), output.cpu() > 0.5, average="samples")
        #
        print('Epoch: {:04d}'.format(epoch),
              'val_acc: {:.4f}'.format(val_acc),
              'val_loss: {:.4f}'.format(val_loss),
              'val_precision: {:.4f}'.format(val_precision),
              'val_recall: {:.4f}'.format(val_recall),
              'val_f1: {:.4f}'.format(val_f1),
              'time: {:.4f}s'.format(time.time() - t))

        # wandb.log({"val_acc": val_acc, "val_loss": val_loss,
        #            "val_precision": val_precision, "val_recall": val_recall,
        #            "val_f1": val_f1})


def test(loader):
    model.eval()

    correct = 0
    for data in train_dataloader:
        out = model(data)
        pred = out.argmax(dim=1)
        correct += int((pred == data.molecular_function).sum())  # Check against ground-truth labels.
    return correct / len(loader.dataset)  # Derive ratio of correct predictions.


for epoch in range(1, args.epochs):
    train(epoch)

# writer.flush()
# writer.close()