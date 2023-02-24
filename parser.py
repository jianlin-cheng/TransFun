import warnings
import argparse
import os
import torch

warnings.filterwarnings("ignore", category=UserWarning)

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=1e-16, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--train_batch', type=int, default=32, help='Training batch size.')
parser.add_argument('--valid_batch', type=int, default=32, help='Validation batch size.')
parser.add_argument('--dropout', type=float, default=0., help='Dropout rate (1 - keep probability).')
parser.add_argument('--seq', type=float, default=0.5, help='Sequence Identity (Sequence Identity).')
parser.add_argument("--ont", default='biological_process', type=str, help='Ontology under consideration')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()


def get_parser():
    return args
