from pathlib import Path
import torch
from dgl.dataloading import GraphDataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from Dataset import load_dataset
from transforms import GraphTransform


# def process_protein_into_dict():
#
#     # Initialize a new PandasPdb object
#     # and fetch the PDB file from rcsb.org
#     ppdb = PandasPdb().read_pdb('/data/pycharm/TransFunData/data/101m.pdb')
#     # print('PDB Code: %s' % ppdb.code)
#     # print('PDB Header Line: %s' % ppdb.header)
#     # print('\nRaw PDB file contents:\n\n%s\n...' % ppdb.pdb_text[:1000])
#     # print(ppdb.df['ATOM'])
#     ppdb.df['ATOM'] = ppdb.df['ATOM'].drop_duplicates(
#         subset=['atom_name', 'residue_name', 'chain_id', 'residue_number']
#     )
#     for record in SeqIO.parse('/data/pycharm/TransFunData/data/101m.pdb', "pdb-atom"):
#         sequence = record.seq
#
#     graph = convert_dfs_to_dgl_graph(ppdb, "pred_filepath", 5, "idt", "output_iviz", sequence)
#
#
#     print(graph)


dataset = load_dataset(str(Path(__file__).parent.absolute()) + '/test_pdb', 'pdb', transform=GraphTransform())


num_examples = len(dataset)
num_train = int(num_examples * 0.8)

train_sampler = SubsetRandomSampler(torch.arange(num_train))

train_dataloader = GraphDataLoader(
    dataset, sampler=train_sampler, batch_size=2, drop_last=False)


for i in train_dataloader:
    print(i)

exit()