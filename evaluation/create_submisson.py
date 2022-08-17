import os

from torch_geometric.graphgym import optim
import torch
import Constants
import params1
from Dataset.Dataset import load_dataset
from models.gnn import GCN3
from parser import get_parser
from preprocessing.utils import load_ckp


def create_submssion_file(group_name, model, keywords):
    mylist = [("AUTHOR", "{}".format(group_name), ""),
              ("MODEL", "{}".format(model), ""),
              ("KEYWORDS", "{}.".format(keywords), ""), ]

    # for i in some_list:
    #     append()

    return mylist


def write_sumssion_file(mylist):
    with open(Constants.ROOT + 'eval/daemons.txt', 'w') as fp:
        fp.write('\n'.join('%s %s %s' % x for x in mylist))


mylist = create_submssion_file(group_name="frimpz", model=1, keywords="Transfun, "
                                                                     "sequence alignment, "
                                                                     "sequence-profile "
                                                                     "alignment")


args = get_parser()
if args.cuda:
    device = 'cuda'

if args.ont == 'molecular_function':
    ont_kwargs = params1.mol_kwargs
elif args.ont == 'cellular_component':
    ont_kwargs = params1.cc_kwargs
elif args.ont == 'biological_process':
    ont_kwargs = params1.bio_kwargs


# model
model = model = GCN3(**ont_kwargs)
model.to(device)


# print(model)

# optimizer
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

# define checkpoint saved path
ckp_dir = Constants.ROOT + '{}/{}/model_checkpoint/{}/'.format(args.seq, args.ont, ont_kwargs['edge_type'])
ckp_pth = ckp_dir + "current_checkpoint.pt"


# load the saved checkpoint
if os.path.exists(ckp_pth):
    print("Loading model checkpoint @ {}".format(ckp_pth))
    model, optimizer, current_epoch, min_val_loss = load_ckp(ckp_pth, model, optimizer)


# print("********************************-Parameters-***************************************")
# print("model = ", model)
# print("optimizer = ", optimizer)
# print("start_epoch = ", current_epoch)
# print("valid_loss_min = ", min_val_loss)
# print("valid_loss_min = {:.6f}".format(min_val_loss))
##  # for param in model.parameters():
##  #   print(param.data)
# print("********************************-Parameters-***************************************")

kwargs = {
    'seq_id': args.seq,
    'ont': args.ont,
    'session': 'test'
}
dataset = load_dataset(root=Constants.ROOT, **kwargs)

for i in dataset:
    print(i)
    exit()

model.eval()

test_dataloader = None


for data in test_dataloader:
    with torch.no_grad():
        output = model(data.to(device))
        print(output)

        exit()
for param in model.parameters():
  print(param.data)
exit()


# write_sumssion_file(mylist)
