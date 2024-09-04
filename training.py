import math
import os
import numpy as np
import torch.optim as optim
from torchsummary import summary

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import Constants
import params

from Dataset.Dataset import load_dataset
from models.gnn import GCN
import argparse
import torch
import time
from torch_geometric.loader import DataLoader
from preprocessing.utils import pickle_save, pickle_load, save_ckp, load_ckp, class_distribution_counter, \
    draw_architecture, compute_roc

import warnings

warnings.filterwarnings("ignore", category=UserWarning)


parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=70, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
parser.add_argument('--train_batch', type=int, default=10, help='Training batch size.')
parser.add_argument('--valid_batch', type=int, default=10, help='Validation batch size.')
parser.add_argument('--seq', type=float, default=0.9, help='Sequence Identity (Sequence Identity).')
parser.add_argument("--ont", default='molecular_function', type=str, help='Ontology under consideration')

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

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

num_class = len(pickle_load(Constants.ROOT + 'go_terms')[f'GO-terms-{args.ont}'])


def create_class_weights(cnter):
    class_weight_path = Constants.ROOT + "{}/{}/class_weights".format(kwargs['seq_id'], kwargs['ont'])
    if os.path.exists(class_weight_path + ".pickle"):
        print("Loading class weights")
        class_weights = pickle_load(class_weight_path)
    else:
        print("Generating class weights")
        go_terms = pickle_load(Constants.ROOT + "/go_terms")
        terms = go_terms['GO-terms-{}'.format(args.ont)]
        class_weights = [cnter[i] for i in terms]

    total = sum(class_weights)
    _max = max(class_weights)
    class_weights = torch.tensor([total / i for i in class_weights], dtype=torch.float).to(device)

    return class_weights
class_weights = create_class_weights(class_distribution_counter(**kwargs))


dataset = load_dataset(root=Constants.ROOT, **kwargs)
labels = pickle_load(Constants.ROOT + "{}_labels".format(args.ont))

edge_types = list(params.edge_types)

train_dataloader = DataLoader(dataset,
                              batch_size=args.train_batch,
                              drop_last=True,
                              exclude_keys=edge_types,
                              shuffle=True)

kwargs['session'] = 'validation'
val_dataset = load_dataset(root=Constants.ROOT, **kwargs)
valid_dataloader = DataLoader(val_dataset,
                              batch_size=args.valid_batch,
                              drop_last=False,
                              shuffle=False,
                              exclude_keys=edge_types)

print('========================================')
print(f'# training proteins: {len(dataset)}')
print(f'# validation proteins: {len(val_dataset)}')
print(f'# Number of classes: {num_class}')
# print(f'# Max class weights: {torch.max(class_weights)}')
# print(f'# Min class weights: {torch.min(class_weights)}')
print('========================================')

current_epoch = 1
min_val_loss = np.Inf

inpu = next(iter(train_dataloader))
model = GCN(**ont_kwargs)


model.to(device)
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=ont_kwargs['wd'])
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
criterion = torch.nn.BCELoss(reduction='none')

labels = pickle_load(Constants.ROOT + "{}_labels".format(args.ont))


def train(start_epoch, min_val_loss, model, optimizer, criterion, data_loader):
    min_val_loss = min_val_loss

    for epoch in range(start_epoch, args.epochs):
        print(" ---------- Epoch {} ----------".format(epoch))
        # initialize variables to monitor training and validation loss
        epoch_loss, epoch_precision, epoch_recall, epoch_accuracy, epoch_f1 = 0.0, 0.0, 0.0, 0.0, 0.0
        val_loss, val_precision, val_recall, val_accuracy, val_f1 = 0.0, 0.0, 0.0, 0.0, 0.0

        t = time.time()

        with torch.autograd.set_detect_anomaly(True):
            lr_scheduler.step()
            ###################
            # train the model #
            ###################
            model.train()
            for pos, data in enumerate(data_loader['train']):
                labs = []
                for la in data['atoms'].protein:
                    labs.append(torch.tensor(labels[la], dtype=torch.float32).view(1, -1))

                labs = torch.cat(labs, dim=0)
                cnts = torch.sum(labs, dim=0)
                total = torch.sum(cnts)

                optimizer.zero_grad()

                output = model(data.to(device))
                loss = criterion(output, labs.to(device))
                loss = (loss * class_weights).mean()


                pom = output.detach().cpu().numpy()
                bins = np.arange(0.0, 1.1, 0.1)
                digitized = np.digitize(pom, bins) - 1
                counts = np.bincount(digitized.flatten(), minlength=len(bins))

                # Display the counts for each range
                for i, count in enumerate(counts):
                    lower_bound = round(bins[i], 1)
                    upper_bound = round(bins[i+1], 1) if i < len(bins) - 1 else 1.0
                    print(f"Range {lower_bound:.1f} - {upper_bound:.1f}: {count} elements")


                exit()

                loss.backward()
                optimizer.step()
                epoch_loss += loss.data.item()

                out_cpu_5 = output.cpu() > 0.5
                epoch_accuracy += accuracy_score(y_true=labs, y_pred=out_cpu_5)
                epoch_precision += precision_score(y_true=labs, y_pred=out_cpu_5, average="samples")
                epoch_recall += recall_score(y_true=labs, y_pred=out_cpu_5, average="samples")
                epoch_f1 += f1_score(y_true=labs, y_pred=out_cpu_5, average="samples")
                

            epoch_accuracy = epoch_accuracy / len(loaders['train'])
            epoch_precision = epoch_precision / len(loaders['train'])
            epoch_recall = epoch_recall / len(loaders['train'])
            epoch_f1 = epoch_f1 / len(loaders['train'])

        ###################
        # Validate the model #
        ###################
        with torch.no_grad():
            model.eval()
            for data in data_loader['valid']:

                labs = []
                for la in data['atoms'].protein:
                    labs.append(torch.tensor(labels[la], dtype=torch.float32).view(1, -1))
                labs = torch.cat(labs)

                output = model(data.to(device))

                _val_loss = criterion(output, labs.to(device))
                _val_loss = (_val_loss * class_weights).mean()
                val_loss += _val_loss.data.item()

                val_accuracy += accuracy_score(labs, output.cpu() > 0.5)
                val_precision += precision_score(labs, output.cpu() > 0.5, average="samples")
                val_recall += recall_score(labs, output.cpu() > 0.5, average="samples")
                val_f1 += f1_score(labs, output.cpu() > 0.5, average="samples")

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

            

            checkpoint = {
                'epoch': epoch,
                'valid_loss_min': val_loss,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }

            # save checkpoint
            # save_ckp(checkpoint, False, ckp_pth,
            #          ckp_dir + "best_model.pt")
            #
            # if val_loss <= min_val_loss:
            #     print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'. \
            #           format(min_val_loss, val_loss))
            #
            #     # save checkpoint as best model
            #     save_ckp(checkpoint, True, ckp_pth,
            #              ckp_dir + "best_model.pt")
            #     min_val_loss = val_loss

    return model


loaders = {
    'train': train_dataloader,
    'valid': valid_dataloader
}


ckp_dir = Constants.ROOT + 'checkpoints/model_checkpoint/{}/'.format("cur")
ckp_pth = ckp_dir + "current_checkpoint.pt"
#ckp_pth = ""

ckp_dir = "/home/fbqc9/PycharmProjects/TFUNClone/TransFun/data/"
ckp_pth = ckp_dir + "molecular_function.pt"

print(ckp_pth)

if os.path.exists(ckp_pth):
    print("Loading model checkpoint @ {}".format(ckp_pth))
    model, optimizer, current_epoch, min_val_loss = load_ckp(ckp_pth, model, optimizer, device="cuda:0")
else:
    if not os.path.exists(ckp_dir):
        os.makedirs(ckp_dir)

print("Training model on epoch {}, with minimum validation loss as {}".format(current_epoch, min_val_loss))

config = {
    "learning_rate": args.lr,
    "epochs": current_epoch,
    "batch_size": args.train_batch,
    "valid_size": args.valid_batch,
    "weight_decay": ont_kwargs['wd']
}



trained_model = train(current_epoch, min_val_loss,
                      model=model, optimizer=optimizer,
                      criterion=criterion, data_loader=loaders)
