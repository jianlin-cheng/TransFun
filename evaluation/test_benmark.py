import os

import Constants
from preprocessing.utils import read_test_set, fasta_to_dictionary, pickle_save, pickle_load
from Bio import SeqIO

ONTS = ['BPO', 'MFO', 'CCO']


def read_line(filename=""):
    with open(filename) as f:
        lines = [line.rstrip() for line in f]
    return lines


# Benchmark contains groundtruth and lists.
# ground_truth cafa terms to go ID
# list only proteins
# both gave me same proteins.
# This is really shitty code.
def get_test_proteins(use='list'):
    if use == 'list':
        pth_pre = Constants.ROOT + "supplementary_data/cafa3/benchmark20171115/lists/"
        ONTS_DIC = {}
        for ont in ONTS:
            ONTS_DIC[ont] = set()
            pth_suf = ['mfo_HUMAN_type2.txt', 'mfo_all_type1.txt', 'bpo_prokarya_type1.txt',
                       'bpo_HUMAN_type1.txt', 'cco_all_type2.txt', 'cco_DICDI_type1.txt',
                       'mfo_HUMAN_type1.txt', 'mfo_all_type2.txt', 'bpo_prokarya_type2.txt',
                       'bpo_HUMAN_type2.txt', 'cco_all_type1.txt', 'cco_DROME_type1.txt',
                       'bpo_eukarya_type1.txt', 'mfo_RAT_type1.txt', 'mfo_prokarya_type1.txt',
                       'cco_RAT_type2.txt', 'cco_DROME_type2.txt', 'bpo_eukarya_type2.txt',
                       'mfo_prokarya_type2.txt', 'cco_RAT_type1.txt', 'bpo_ECOLI_type2.txt',
                       'cco_prokarya_type1.txt', 'cco_ARATH_type1.txt', 'xxo_all_typex.txt',
                       'mfo_ECOLI_type1.txt', 'bpo_ECOLI_type1.txt', 'bpo_all_typex.txt',
                       'cco_prokarya_type2.txt', 'cco_ARATH_type2.txt', 'mfo_ECOLI_type2.txt',
                       'bpo_CANAX_type1.txt', 'bpo_MOUSE_type1.txt', 'mfo_MOUSE_type2.txt',
                       'bpo_MOUSE_type2.txt', 'mfo_MOUSE_type1.txt', 'mfo_ARATH_type2.txt',
                       'cco_eukarya_type1.txt', 'bpo_ARATH_type1.txt', 'bpo_all_type2.txt',
                       'cco_ECOLI_type2.txt', 'mfo_ARATH_type1.txt', 'cco_eukarya_type2.txt',
                       'bpo_ARATH_type2.txt', 'bpo_all_type1.txt', 'cco_ECOLI_type1.txt',
                       'mfo_BACSU_type1.txt', 'mfo_eukarya_type1.txt', 'cco_MOUSE_type1.txt',
                       'bpo_RAT_type2.txt', 'mfo_eukarya_type2.txt', 'bpo_DANRE_type1.txt',
                       'bpo_BACSU_type1.txt', 'cco_MOUSE_type2.txt', 'bpo_RAT_type1.txt',
                       'cco_HUMAN_type1.txt', 'bpo_DICDI_type1.txt', 'mfo_SALTY_type1.txt',
                       'mfo_all_typex.txt', 'cco_HUMAN_type2.txt', 'cco_all_typex.txt',
                       'mfo_DROME_type2.txt', 'bpo_SCHPO_type1.txt', 'mfo_SCHPO_type2.txt',
                       'bpo_DROME_type1.txt', 'mfo_DROME_type1.txt', 'bpo_SCHPO_type2.txt',
                       'bpo_DROME_type2.txt']
            for pth in pth_suf:
                if ont.lower() in pth:
                    ONTS_DIC[ont].update(read_line(pth_pre + pth))

    elif use == 'groundtruth':
        ONTS_DIC = {'BPO': {}, 'MFO': {}, 'CCO': {}}
        for ont in ONTS:
            olp = Constants.ROOT + "supplementary_data/cafa3/benchmark20171115/" \
                                   "groundtruth/leafonly_{}.txt".format(ont)

            with open(olp) as f:
                lines = [line.rstrip().split('\t') for line in f]

            for line in lines:
                if line[0] in ONTS_DIC[ont]:
                    ONTS_DIC[ont][line[0]].append(line[1])
                else:
                    ONTS_DIC[ont][line[0]] = list(line[1])

    return ONTS_DIC


def qcheck_bench():
    for i in ONTS:
        leaf = get_test_proteins(use='list')
        gtruth = get_test_proteins(use='groundtruth')
        print("**************{}*****************".format(i))
        print("Total # proteins in leaf {}".format(len(leaf[i])))
        print("Total # proteins in groundtruth {}".format(len(set(gtruth[i].keys()))))
        print("Leaf - Gtruth {}".format(leaf[i].difference(set(gtruth[i].keys()))))
        print("Gtruth - Leaf {}".format(set(gtruth[i].keys()).difference(leaf[i])))
        print("*******************************")


def map_cafaID_proteinnames():
    pth = Constants.ROOT + "supplementary_data/cafa3/CAFA3_targets/Mapping files/{}"

    all_mappings = ['10116', '170187', '559292', '8355', '99287', '273057', '321314',
                    '284812', '83333', '3702', '7955', '243232', '9606', '224308',
                    '85962', '160488', '223283', '44689', '10090', '243273']

    xtra_mappings = ['mapping.7227.map', 'mapping.208963.map', 'mapping.237561.map', 'target_moonlight.map']

    full_mapping = []
    for mapping_file in all_mappings:
        maps = read_test_set(pth.format("sp_species.{}.map".format(mapping_file)))
        maps = [map + [mapping_file] for map in maps]
        full_mapping.extend(maps)

    maps = read_test_set(pth.format("target_moonlight.map"))
    maps = [map + ["moonlight"] for map in maps]
    full_mapping.extend(maps)

    full_mapping = {i[0]: (i[1], i[2]) for i in full_mapping}
    return full_mapping


def collect_test():
    test_proteins_list = []
    test_proteins_not_found_list = []
    test_proteins_not_found_fasta = []
    fasta_dictionary = fasta_to_dictionary(Constants.ROOT + 'uniprot/uniprot_sprot.fasta', identifier='protein_name')
    cafaID_proteins = map_cafaID_proteinnames()
    test_proteins = get_test_proteins(use='list')

    for ont in ONTS:
        for test_protein in test_proteins[ont]:
            if test_protein in cafaID_proteins:
                name = cafaID_proteins[test_protein][0]
                if name in fasta_dictionary:
                    tmp = fasta_dictionary[name][0].split("|")
                    acc = tmp[1]
                    sp = cafaID_proteins[test_protein][1]
                    seq = str(fasta_dictionary[cafaID_proteins[test_protein][0]][3])
                    test_proteins_list.append((acc, name, test_protein, sp, ont, seq))
                else:
                    test_proteins_not_found_fasta.append((name, test_protein))
            else:
                test_proteins_not_found_list.append((test_protein,))
    print("{} files found {} files not found {} not found in fasta, maybe in trembl" \
          .format(len(test_proteins_list),
                  len(test_proteins_not_found_list),
                  len(test_proteins_not_found_fasta)))

    pickle_save(test_proteins_list, Constants.ROOT + "eval/test_proteins_list")
    pickle_save(test_proteins_not_found_list, Constants.ROOT + "eval/test_proteins_not_found_list")
    pickle_save(test_proteins_not_found_fasta, Constants.ROOT + "eval/test_proteins_not_found_fasta")


# view pickled data
def view_saved():
    print(pickle_load(Constants.ROOT + 'eval/test_proteins_list'))
    # print(pickle_load(Constants.ROOT + 'eval/test_proteins_not_found_fasta'))
    # print(pickle_load(Constants.ROOT + 'eval/test_proteins_not_found_list'))


# create evaluation directory
eval_pth = Constants.ROOT + 'eval'
if not os.path.exists(eval_pth):
    os.mkdir(eval_pth)

# collect_test()

view_saved()
