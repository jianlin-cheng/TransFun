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
import time
from torch_geometric.loader import DataLoader
import pandas as pd
from collections import Counter
from preprocessing.utils import pickle_save, pickle_load, save_ckp, load_ckp

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
parser.add_argument('--train_batch', type=int, default=250, help='Training batch size.')
parser.add_argument('--valid_batch', type=int, default=200, help='Validation batch size.')
parser.add_argument('--dropout', type=float, default=0., help='Dropout rate (1 - keep probability).')
parser.add_argument('--seq', type=float, default=0.3, help='Sequence Identity (Sequence Identity).')
parser.add_argument("--ont", default='molecular_function', type=str, help='Ontology under consideration')

device = 'cpu'
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
if args.cuda:
    device = 'cuda'

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


def check_counter(**kwargs):
    """
        Count the number of proteins for each GO term in training set.
    """
    data = pickle_load(Constants.ROOT + "{}/{}/{}".format(kwargs['seq_id'], kwargs['ont'], kwargs['session']))

    all_proteins = []
    for i in data:
        all_proteins.extend(data[i])

    annot = pd.read_csv(Constants.ROOT + 'annot.tsv', delimiter='\t')
    annot = annot.where(pd.notnull(annot), None)
    annot = annot[annot['Protein'].isin(all_proteins)]
    annot = pd.Series(annot[kwargs['ont']].values, index=annot['Protein']).to_dict()

    terms = []
    for i in annot:
        terms.extend(annot[i].split(","))

    counter = Counter(terms)

    for i in counter.most_common():
        print(i)
    print("# of ontologies is {}".format(len(counter)))


def create_class_weights(**kwargs):
    class_weight_path = Constants.ROOT + "{}/{}/class_weights".format(kwargs['seq_id'], kwargs['ont'])
    if os.path.exists(class_weight_path + ".pickle"):
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
    class_weights = [total / i for i in class_weights]
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    return class_weights


#########################################################
# Creating training data #
#########################################################
kwargs = {
    'seq_id': args.seq,
    'ont': args.ont,
    'session': 'train'
}
dataset = load_dataset(root=Constants.ROOT, **kwargs)
class_weights = create_class_weights(**kwargs)
train_dataloader = DataLoader(dataset, batch_size=args.train_batch, drop_last=False, shuffle=True)

kwargs['session'] = 'valid'
val_dataset = load_dataset(root=Constants.ROOT, **kwargs)
valid_dataloader = DataLoader(val_dataset, batch_size=args.valid_batch, drop_last=False, shuffle=True)

print('========================================')
print(f'# training proteins: {len(dataset)}')
print(f'# validation proteins: {len(val_dataset)}')
print('========================================')

num_class = len(pickle_load(Constants.ROOT + 'go_terms')[f'GO-terms-{args.ont}'])

current_epoch = 1
min_val_loss = np.Inf

model = GCN(input_features=dataset.num_features,
            hidden_channels_1=args.hidden1,
            hidden_channels_2=args.hidden2,
            hidden_channels_3=args.hidden3,
            num_classes=num_class)
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
criterion = torch.nn.BCELoss(reduction='none')


def train(start_epoch, min_val_loss, model, optimizer, criterion, data_loader):
    min_val_loss = min_val_loss

    for epoch in range(start_epoch, args.epochs):

        # initialize variables to monitor training and validation loss
        epoch_loss, epoch_precision, epoch_recall, epoch_accuracy, epoch_f1 = 0.0, 0.0, 0.0, 0.0, 0.0
        val_loss, val_precision, val_recall, val_accuracy, val_f1 = 0.0, 0.0, 0.0, 0.0, 0.0

        t = time.time()
        with torch.autograd.set_detect_anomaly(True):

            ###################
            # train the model #
            ###################
            model.train()
            for data in data_loader['train']:

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

            epoch_accuracy = epoch_accuracy / len(loaders['train'])
            epoch_precision = epoch_precision / len(loaders['train'])
            epoch_recall = epoch_recall / len(loaders['train'])
            epoch_f1 = epoch_f1 / len(loaders['train'])

            ###################
            # Validate the model #
            ###################
            print(" ---------- EVALUATE ON VALIDATION SET ----------")
            model.eval()
            for data in data_loader['valid']:
                output = model(data.to(device))

                _val_loss = criterion(output, data.molecular_function)
                _val_loss = (_val_loss * class_weights).mean()

                val_loss += _val_loss.data.item()
                val_accuracy += accuracy_score(data.molecular_function.cpu(), output.cpu() > 0.5)
                val_precision += precision_score(data.molecular_function.cpu(), output.cpu() > 0.5, average="samples")
                val_recall += recall_score(data.molecular_function.cpu(), output.cpu() > 0.5, average="samples")
                val_f1 += f1_score(data.molecular_function.cpu(), output.cpu() > 0.5, average="samples")

            val_loss = val_loss / len(loaders['valid'])
            val_accuracy = val_accuracy / len(loaders['valid'])
            val_precision = val_precision / len(loaders['valid'])
            val_recall = val_recall / len(loaders['valid'])
            val_f1 = val_f1 / len(loaders['valid'])

            print(  'Epoch: {:04d}'.format(epoch),
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

            # # save checkpoint
            save_ckp(checkpoint, False, Constants.ROOT + 'model_checkpoint',
                     Constants.ROOT + 'best_model')

            if val_loss <= min_val_loss:
                print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'. \
                      format(min_val_loss, val_loss))

                # save checkpoint as best model
                save_ckp(checkpoint, True, Constants.ROOT + 'model_checkpoint',
                         Constants.ROOT + 'best_model')
                min_val_loss = val_loss

    return model


# check_counter(**kwargs)
loaders = {
    'train': train_dataloader,
    'valid': valid_dataloader
}
ckp_pth = Constants.ROOT + 'model_checkpoint'

if os.path.exists(ckp_pth):
    print("Loading model checkpoint")
    model, optimizer, current_epoch, min_val_loss = load_ckp(ckp_pth, model, optimizer)

print("Training model on epoch {}, with minimum validation loss as {}".format(current_epoch, min_val_loss))

trained_model = train(current_epoch, min_val_loss, model=model, optimizer=optimizer, criterion=criterion, data_loader=loaders)

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
