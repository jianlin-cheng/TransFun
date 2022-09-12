import math
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

TEST_FILE = [("leafonly_MFO", "molecular_function"),
             ("leafonly_CCO", "cellular_component"),
             ("leafonly_BPO", "biological_process")]


# TEST_FILE = "mfo_HUMAN_type1"
# TEST_FILE = "leafonly_CCO"


def write_sumssion_file(mylist, file_name):
    with open(Constants.ROOT + 'eval/predicted/{}.txt'.format(file_name), 'w') as fp:
        fp.write('\n'.join('%s\t%s\t%s' % x for x in mylist))


def create_submssion_file(group_name, model, keywords, results):

    mylist = [("AUTHOR", "{}".format(group_name), ""),
              ("MODEL", "{}".format(model), ""),
              ("KEYWORDS", "{}.".format(keywords), ""), ]
    nans = set()
    for ont in results:
        tmp = results[ont]
        for i in zip(tmp[0], tmp[1]):
            assert len(tmp[2]) == len(i[1])
            for j in zip(tmp[2], i[1]):
                # if j[1] > 0.01:
                if (format(j[1], ".2f")) == 'nan':
                    nans.add(i[0])
                else:
                    mylist.append((i[0], j[0], format(j[1], ".2f")))
    mylist.append(("END", "", ""))
    print(nans)
    return mylist


args = get_parser()
if args.cuda:
    device = 'cuda'

results = {}
for i in TEST_FILE[0:1]:
    args.ont = i[1]

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
    ckp_dir = Constants.ROOT + '{}/{}/model_checkpoint/{}/'.format(args.seq, args.ont, ont_kwargs['edge_type'])
    ckp_pth = ckp_dir + "current_checkpoint.pt"

    # load the saved checkpoint
    if os.path.exists(ckp_pth):
        # print("Loading model checkpoint @ {}".format(ckp_pth))
        model, optimizer, current_epoch, min_val_loss = load_ckp(ckp_pth, model, optimizer)
    else:
        continue

    model.eval()

    # print("********************************-Parameters-***************************************")
    # # print("model = ", model)
    # # print("optimizer = ", optimizer)
    # print("start_epoch = ", current_epoch)
    # print("valid_loss_min = ", min_val_loss)
    # print("valid_loss_min = {:.6f}".format(min_val_loss))
    # #  # for param in model.parameters():
    # #  #   print(param.data)
    # print("********************************-Parameters-***************************************")

    kwargs = {
        'seq_id': args.seq,
        'ont': args.ont,
        'session': 'test',
        'test_file': '{}.txt'.format(i[0])
    }
    dataset = load_dataset(root=Constants.ROOT, **kwargs)
    # print(f'# Evaluation proteins: {len(dataset)}')
    print("********************************-Evaluating for {} {} proteins-***************************************". \
          format(i[1], len(dataset)))

    test_dataloader = DataLoader(dataset,
                                 batch_size=32,
                                 drop_last=False,
                                 # sampler=ImbalancedDatasetSampler(dataset, **kwargs, device=device),
                                 # exclude_keys=edge_types,
                                 shuffle=True)
    probabilities = []
    proteins = []
    for data in test_dataloader:
        with torch.no_grad():
            proteins.extend(data['atoms'].protein)
            probabilities.extend(model(data.to(device)).tolist())

    num_class = pickle_load(Constants.ROOT + 'go_terms')[f'GO-terms-{args.ont}']
    results[args.ont] = (proteins, probabilities, num_class)


mylist = create_submssion_file(group_name="Frimpz",
                               model=1,
                               keywords="sequence alignment",
                               results=results)

NAMESPACES = dict((v, k) for k, v in Constants.NAMESPACES.items())
write_sumssion_file(mylist, file_name=NAMESPACES[kwargs['ont']] + "_" + str(1) + "_" + "all")
