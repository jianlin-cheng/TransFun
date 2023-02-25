import math
import os
import numpy as np
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import Constants
import params1
import wandb

from Dataset.Dataset import load_dataset
from models.gnn import GCN3
import argparse
import torch
import time
from torch_geometric.loader import DataLoader
from preprocessing.utils import pickle_save, pickle_load, save_ckp, load_ckp, class_distribution_counter, \
    draw_architecture, compute_roc

import warnings

warnings.filterwarnings("ignore", category=UserWarning)

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["WANDB_API_KEY"] = "b155b6571149501f01b9790e27f6ddac80ae09b3"
os.environ["WANDB_MODE"] = "online"

parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
parser.add_argument('--train_batch', type=int, default=32, help='Training batch size.')
parser.add_argument('--valid_batch', type=int, default=16, help='Validation batch size.')
parser.add_argument('--test_batch', type=int, default=15, help='Validation batch size.')
parser.add_argument('--seq', type=float, default=0.9, help='Sequence Identity (Sequence Identity).')
parser.add_argument("--ont", default='cellular_component', type=str, help='Ontology under consideration')

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
    ont_kwargs = params1.mol_kwargs
elif args.ont == 'cellular_component':
    ont_kwargs = params1.cc_kwargs
elif args.ont == 'biological_process':
    ont_kwargs = params1.bio_kwargs

wandb.init(project="Transfun_project_{}".format(args.ont), entity='frimpz',
           name="{}_{}".format(args.seq, ont_kwargs['edge_type']))

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

num_class = len(pickle_load(Constants.ROOT + 'go_terms')[f'GO-terms-{args.ont}'])


# some_weights = pickle_load(Constants.ROOT + "computed_weights")
def create_class_weights(cnter, cnter2):
    class_weight_path = Constants.ROOT + "{}/{}/class_weights".format(kwargs['seq_id'], kwargs['ont'])
    if os.path.exists(class_weight_path + ".pickle"):
        print("Loading class weights")
        class_weights = pickle_load(class_weight_path)
    else:
        print("Generating class weights")
        go_terms = pickle_load(Constants.ROOT + "/go_terms")
        terms = go_terms['GO-terms-{}'.format(args.ont)]  # [:600]
        class_weights = [cnter[i] for i in terms]
        # print(class_weights[:10])
        # pickle_save(class_weights, class_weight_path)

    total = sum(class_weights)  # /100
    # print(total)
    _max = max(class_weights)
    # print(max(class_weights), min(class_weights), total)
    # class_weights = torch.tensor([_max - i for i in class_weights], dtype=torch.float).to(device)
    class_weights = torch.tensor([total / i for i in class_weights], dtype=torch.float).to(device)
    # class_weights = torch.tensor([total / (i * num_class) for i in class_weights], dtype=torch.float).to(device)
    # class_weights = torch.tensor([1 / i for i in class_weights], dtype=torch.float).to(device)

    return class_weights


#########################################################
# Creating training data_bp #
#########################################################


# class_weights = create_class_weights(class_distribution_counter(**kwargs), "some_weights")
# print(class_weights)
# exit()
#########################################################
# Creating training data_bp #
#########################################################

dataset = load_dataset(root=Constants.ROOT, **kwargs)

edge_types = list(params1.edge_types - {args.ont})

train_dataloader = DataLoader(dataset,
                              batch_size=args.train_batch,
                              drop_last=False,
                              # sampler=ImbalancedDatasetSampler(dataset, **kwargs, device=device),
                              exclude_keys=edge_types,
                              shuffle=True)

kwargs['session'] = 'valid'
val_dataset = load_dataset(root=Constants.ROOT, **kwargs)
valid_dataloader = DataLoader(val_dataset,
                              batch_size=args.valid_batch,
                              drop_last=False,
                              shuffle=True,
                              exclude_keys=edge_types)

kwargs['session'] = 'test'
kwargs['test_file'] = Constants.ROOT + "/timebased/prots_1"
test_dataset = load_dataset(root=Constants.ROOT, **kwargs)
test_dataloader = DataLoader(test_dataset,
                             batch_size=args.test_batch,
                             drop_last=False,
                             shuffle=True,
                             exclude_keys=edge_types)

print('========================================')
print(f'# training proteins: {len(dataset)}')
print(f'# validation proteins: {len(val_dataset)}')
print(f'# Test proteins: {len(test_dataset)}')
print(f'# Number of classes: {num_class}')
# print(f'# Max class weights: {torch.max(class_weights)}')
# print(f'# Min class weights: {torch.min(class_weights)}')
print('========================================')

current_epoch = 1
min_val_loss = np.Inf

model = GCN3(**ont_kwargs)

model.to(device)
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=ont_kwargs['wd'])
criterion = torch.nn.BCELoss(reduction='none')

labels = pickle_load(Constants.ROOT + "{}_labels".format(args.ont))

exit()
# test_labels = pickle_load(Constants.ROOT + "{}_timebased_labels".format(args.ont))


def train(start_epoch, min_val_loss, model, optimizer, criterion, data_loader):
    min_val_loss = min_val_loss

    for epoch in range(start_epoch, args.epochs):
        print(" ---------- Epoch {} ----------".format(epoch))
        # initialize variables to monitor training and validation loss
        epoch_loss, epoch_precision, epoch_recall, epoch_accuracy, epoch_f1 = 0.0, 0.0, 0.0, 0.0, 0.0
        val_loss, val_precision, val_recall, val_accuracy, val_f1 = 0.0, 0.0, 0.0, 0.0, 0.0
        tst_loss, tst_precision, tst_recall, tst_accuracy, tst_f1 = 0.0, 0.0, 0.0, 0.0, 0.0

        t = time.time()

        with torch.autograd.set_detect_anomaly(True):
            ###################
            # train the model #
            ###################
            model.train()
            for data in data_loader['train']:
                optimizer.zero_grad()

                labs = []
                for la in data['atoms'].protein:
                    labs.append(torch.tensor(labels[la], dtype=torch.float32).view(1, -1))

                labs = torch.cat(labs, dim=0)

                cnts = torch.sum(labs, dim=0)
                total = torch.sum(cnts)

                class_weights = torch.tensor([total / i if i > 0 else total + 1 for i in cnts],
                                              dtype=torch.float).to(device)

                # class_weights = torch.tensor([math.log2(total / i) if i > 0 else math.log2(total) for i in cnts],
                #                              dtype=torch.float).to(device)

                output = model(data.to(device))
                loss = criterion(output, labs.to(device))
                loss = (loss * class_weights).mean()

                loss.backward()
                optimizer.step()
                epoch_loss += loss.data.item()

                out_cpu_5 = output.cpu() > 0.5
                epoch_accuracy += accuracy_score(y_true=labs, y_pred=out_cpu_5)
                epoch_precision += precision_score(y_true=labs, y_pred=out_cpu_5, average="samples")
                epoch_recall += recall_score(y_true=labs, y_pred=out_cpu_5, average="samples")
                epoch_f1 += f1_score(y_true=labs, y_pred=out_cpu_5, average="samples")

                # print(precision_score(y_true=labs, y_pred=out_cpu_5, average="samples"),
                #         recall_score(y_true=labs, y_pred=out_cpu_5, average="samples"),
                #               f1_score(y_true=labs, y_pred=out_cpu_5, average="samples"), loss)

                # print(precision_score(y_true=getattr(data_bp['atoms'], args.ont).cpu(), y_pred=out_cpu_5, average="samples"),
                #       recall_score(y_true=getattr(data_bp['atoms'], args.ont).cpu(), y_pred=out_cpu_5, average="samples"),
                #       f1_score(y_true=getattr(data_bp['atoms'], args.ont).cpu(), y_pred=out_cpu_5, average="samples"))

            epoch_accuracy = epoch_accuracy / len(loaders['train'])
            epoch_precision = epoch_precision / len(loaders['train'])
            epoch_recall = epoch_recall / len(loaders['train'])
            epoch_f1 = epoch_f1 / len(loaders['train'])

        ###################
        # Test the model #
        ###################
        # with torch.no_grad():
        #     model.eval()
        #     for data in data_loader['test']:
        #
        #         labs = []
        #         for la in data['atoms'].protein:
        #             labs.append(test_labels[la])
        #         labs = torch.cat(labs)
        #
        #
        #         output = model(data.to(device))
        #
        #         _tst_loss = criterion(output, labs.to(device))
        #         _tst_loss = (_tst_loss * class_weights).mean()
        #         tst_loss += _tst_loss.data.item()
        #
        #         tst_accuracy += accuracy_score(labs, output.cpu() > 0.5)
        #         tst_precision += precision_score(labs, output.cpu() > 0.5, average="samples")
        #         tst_recall += recall_score(labs, output.cpu() > 0.5, average="samples")
        #         tst_f1 += f1_score(labs, output.cpu() > 0.5, average="samples")
        #
        #     tst_loss = tst_loss / len(loaders['test'])
        #     tst_accuracy = tst_accuracy / len(loaders['test'])
        #     tst_precision = tst_precision / len(loaders['test'])
        #     tst_recall = tst_recall / len(loaders['test'])
        #     tst_f1 = tst_f1 / len(loaders['test'])

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
                  # 'tst_acc: {:.4f}'.format(tst_accuracy),
                  # 'tst_loss: {:.4f}'.format(tst_loss),
                  # 'tst_precision: {:.4f}'.format(tst_precision),
                  # 'tst_recall: {:.4f}'.format(tst_recall),
                  # 'tst_f1: {:.4f}'.format(tst_f1),
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
                       "val_f1": val_f1,
                       # "tst_acc": tst_accuracy,
                       # "tst_loss": tst_loss,
                       # "tst_precision": tst_precision,
                       # "tst_recall": tst_recall,
                       # "tst_f1": tst_f1
                       })

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
    'valid': valid_dataloader,
    'test': test_dataloader
}

ckp_dir = Constants.ROOT + '{}/{}/model_checkpoint/{}/'.format(args.seq, args.ont, ont_kwargs['edge_type'])
ckp_pth = ckp_dir + "current_checkpoint.pt"
if os.path.exists(ckp_pth):
    print("Loading model checkpoint @ {}".format(ckp_pth))
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
