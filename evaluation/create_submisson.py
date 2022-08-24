import os

import pandas as pd
from torch_geometric.graphgym import optim
import torch
import Constants
import params1
from Dataset.Dataset import load_dataset
from models.gnn import GCN3
from parser import get_parser
from preprocessing.utils import load_ckp, pickle_save, pickle_load
from torch_geometric.loader import DataLoader


def write_sumssion_file(mylist):
    with open(Constants.ROOT + 'eval/dae_1_9606.txt', 'w') as fp:
        fp.write('\n'.join('%s\t%s\t%s' % x for x in mylist))


def create_submssion_file(group_name, model, keywords, proteins, probabilities):
    num_class = pickle_load(Constants.ROOT + 'go_terms')[f'GO-terms-{args.ont}']
    mylist = [("AUTHOR", "{}".format(group_name), ""),
              ("MODEL", "{}".format(model), ""),
              ("KEYWORDS", "{}.".format(keywords), ""), ]

    for i in zip(proteins, probabilities):
        assert len(num_class) == len(i[1])
        for j in zip(num_class, i[1]):
            # if j[1] > 0.01:
            mylist.append((i[0],  j[0], format(j[1], ".2f")))
    mylist.append(("END", "", ""))
    return mylist


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

# optimizer
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

# define checkpoint saved path
ckp_dir = Constants.ROOT + 'may/{}/{}/model_checkpoint/{}/'.format(args.seq, args.ont, ont_kwargs['edge_type'])
ckp_pth = ckp_dir + "current_checkpoint.pt"
print(ckp_pth)
# load the saved checkpoint
if os.path.exists(ckp_pth):
    print("Loading model checkpoint @ {}".format(ckp_pth))
    model, optimizer, current_epoch, min_val_loss = load_ckp(ckp_pth, model, optimizer)
model.eval()

print("********************************-Parameters-***************************************")
# print("model = ", model)
# print("optimizer = ", optimizer)
print("start_epoch = ", current_epoch)
print("valid_loss_min = ", min_val_loss)
print("valid_loss_min = {:.6f}".format(min_val_loss))
#  # for param in model.parameters():
#  #   print(param.data)
print("********************************-Parameters-***************************************")

kwargs = {
    'seq_id': args.seq,
    'ont': args.ont,
    'session': 'test'
}
dataset = load_dataset(root=Constants.ROOT, **kwargs)
print(f'# training proteins: {len(dataset)}')


for i in dataset:
    if i['atoms'].pos.shape[0] != i['atoms'].embedding_features_per_residue.shape[0]:
        print(i['atoms'].protein)
exit()

test_dataloader = DataLoader(dataset,
                             batch_size=32,
                             drop_last=False,
                             # sampler=ImbalancedDatasetSampler(dataset, **kwargs, device=device),
                             # exclude_keys=edge_types,
                             shuffle=True)


probabilities = []
proteins = []
for data in test_dataloader:
    # if not data['atoms'].pos.shape[0] == data['atoms'].embedding_features_per_residue.shape[0]:
    #     print(data['atoms'].protein)
    #     # print(data['atoms'].pos.shape, data['atoms'].embedding_features_per_residue.shape)


    with torch.no_grad():
        proteins.extend(data['atoms'].protein)
        probabilities.extend(model(data.to(device)).tolist())

mylist = create_submssion_file(group_name="Frimpz", model=1, keywords="sequence alignment", proteins=proteins, probabilities=probabilities)
write_sumssion_file(mylist)

#
# # mylist = create_submssion_file(group_name="frimpz", model=1, keywords="Transfun, "
# #                                                                   "sequence alignment, "
# #                                                                   "sequence-profile "
# #                                                                   "alignment")
# # write_sumssion_file(mylist)
# exit()