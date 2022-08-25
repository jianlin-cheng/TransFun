import os
import shutil

import torch

import Constants
from preprocessing.utils import read_test_set, fasta_to_dictionary, pickle_save, pickle_load, cafa_fasta_to_dictionary, \
    create_seqrecord, get_sequence_from_pdb, count_proteins
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


lop = {'T37020005978', 'T100900003996', 'T96060017383', 'T2848120001267', 'T37020006649', 'T79550002031',
       'T96060016545', 'T37020007637', 'T100900016182', 'T833330003802', 'T37020009389', 'T37020014229',
       'T2848120003992', 'T96060019147', 'T2848120002016', 'T2848120003588', 'T96060005380', 'T96060015794',
       'T37020013496', 'T96060002954', 'T96060002752', 'T2848120001404', 'T96060005870', 'T96060012141',
       'T37020012229', 'T96060013032', 'T96060005274', 'T446890002955', 'T100900005305', 'T37020009192',
       'T96060014796', 'T96060012540', 'T96060008065', 'T100900006761', 'T96060013575', 'T96060015290',
       'T100900003509', 'T446890001863',
       'T100900015687', 'T446890001560', 'T96060008400', 'T37020003722', 'T833330004243', 'T96060012120',
       'T37020012479', 'T96060015038', 'T100900013200', 'T2848120000472', 'T96060013579', 'T100900014234',
       'T96060018347', 'T100900006830', 'T100900002577', 'T79550000305', 'T2848120003245', 'T2848120000602',
       'T100900014226', 'T79550002315', 'T96060009108', 'T96060019045', 'T446890003376', 'T37020003779', 'T37020001637',
       'T37020002028', 'T100900016693', 'T101160002059', 'T96060005086', 'T37020005875', 'T2848120000506',
       'T100900009309', 'T96060015549', 'T2848120002352', 'T96060015548', 'T100900016430', 'T96060019408',
       'T37020008396', 'T96060016023', 'T100900009533', 'T100900007110', 'T37020003777', 'T37020009774',
       'T100900016795', 'T96060010850', 'T37020010240', 'T79550002464', 'T37020014565', 'T96060008263', 'T100900012926',
       'T37020002022', 'T37020007636', 'T2243080000511', 'T96060009857', 'T96060008030', 'T37020003241', 'T96060003753',
       'T37020000258', 'T37020013847', 'T79550001437', 'T5592920001321', 'T37020012427', 'T37020002642',
       'T100900009767', 'T96060018291', 'T446890002612', 'T100900016692', 'T100900007178', 'T2848120001042',
       'T37020006298', 'T100900011593', 'T37020008478', 'T96060017229', 'T37020013495', 'T100900016257',
       'T2848120003786', 'T446890000952', 'T2848120001806', 'T2848120004784', 'T2848120000137', 'T96060013558',
       'T37020011435', 'T2848120004777', 'T37020012201', 'T96060008939', 'T101160003669', 'T100900003260',
       'T37020000188', 'T37020002020', 'T2848120002411', 'T37020008028', 'T2848120001866', 'T2848120002406',
       'T100900009017', 'T100900003995', 'T2848120003191', 'T37020012243', 'T100900012368', 'T100900008642',
       'T446890001563', 'T37020012970', 'T446890000954', 'T100900014230', 'T96060009216', 'T37020002025',
       'T37020000185', 'T100900013197', 'T96060009572', 'T2848120001937', 'T37020011636', 'T2848120002118',
       'T37020000296', 'T100900009708', 'T96060003237', 'T100900005545', 'T100900012162', 'T2848120002326',
       'T96060000951', 'T37020014618', 'T100900008777', 'T100900010472', 'T96060014778', 'T446890003351',
       'T96060007414', 'T37020009390', 'T37020011693', 'T100900011485', 'T96060009167', 'T101160004697', 'T37020001283',
       'T37020002641', 'T96060008739', 'T101160006681', 'T2848120004928', 'T2848120004514', 'T96060010847',
       'T96060006328', 'T100900014216', 'T96060019392', 'T100900011979', 'T2848120002271', 'T96060016883',
       'T2848120001621', 'T37020013556', 'T96060013740', 'T101160005651', 'T37020008439', 'T37020012828',
       'T37020006857', 'T96060014288', 'T100900010963', 'T37020000253', 'T101160004221', 'T100900013522',
       'T96060018324', 'T37020002021', 'T37020007747', 'T2848120003546', 'T37020002384', 'T2848120000366',
       'T100900000760', 'T2848120000822', 'T96060016448', 'T96060008730', 'T96060010152', 'T96060019899',
       'T96060013867', 'T100900000026', 'T2848120001938', 'T37020008022', 'T37020000621', 'T37020002377',
       'T96060012069', 'T100900001927', 'T37020011437', 'T100900016676', 'T833330000648', 'T37020010241',
       'T37020000211', 'T100900004008', 'T96060018325', 'T446890003063', 'T2848120004370', 'T96060011559',
       'T2848120002953', 'T37020008508', 'T446890002956', 'T37020011429', 'T37020002027', 'T96060013745',
       'T2848120000210', 'T96060013535', 'T79550000061', 'T37020008244', 'T96060016817', 'T2848120000611',
       'T37020003398', 'T79550002846', 'T2848120000294', 'T96060012976', 'T96060001910', 'T101160004422',
       'T37020010586', 'T96060010968'}


# Collect proteins
def collect_test():
    test_proteins_list = []
    all_map = {}
    no_alpha_fold = set()
    cafa_id_not_mapped = set()

    map_names = fasta_to_dictionary(Constants.ROOT + 'uniprot/uniprot_sprot.fasta', identifier='protein_name')
    cafa_fasta = cafa_fasta_to_dictionary(Constants.ROOT + 'supplementary_data/cafa3/CAFA3_targets/Target '
                                                           'files/all.fasta')
    cafaID_proteins = map_cafaID_proteinnames()
    test_proteins = get_test_proteins(use='list')

    for ont in ONTS:
        for test_protein in test_proteins[ont]:

            if test_protein in cafaID_proteins:
                name = cafaID_proteins[test_protein][0]

                # uniprot ID
                if name in map_names:
                    tmp = map_names[name][0].split("|")[1]
                else:
                    tmp = ""

                if not test_protein in all_map:
                    src = os.path.join(Constants.ROOT, "alphafold/AF-{}-F1-model_v2.pdb.gz".format(tmp))
                    if os.path.isfile(src):
                        alpha_fold_seq = get_sequence_from_pdb(src, "A")
                        # if alpha_fold_seq != str(cafa_fasta[name][3]):
                        #     unmatched_with_alpha_fold.add(test_protein)
                        # else:
                        all_map[test_protein] = (name, tmp)
                        test_proteins_list.append(create_seqrecord(id="cafa|" + test_protein + "|" + name,
                                                                   description=tmp,
                                                                   seq=str(alpha_fold_seq)))
                    else:
                        no_alpha_fold.add(test_protein)
            else:
                cafa_id_not_mapped.add(test_protein)

    SeqIO.write(test_proteins_list, Constants.ROOT + "eval/test.fasta".format("test"), "fasta")
    pickle_save(all_map, Constants.ROOT + "eval/all_map")
    pickle_save(no_alpha_fold, Constants.ROOT + "eval/no_alpha_fold")
    pickle_save(cafa_id_not_mapped, Constants.ROOT + "eval/cafa_id_not_mapped")


# collect_test()


def collect_seq_gt_1021():
    tmp = []
    input_seq_iterator = SeqIO.parse(Constants.ROOT + "eval/test.fasta", "fasta")
    for record in input_seq_iterator:
        if len(record.seq) > 1021:
            tmp.append(record)
    SeqIO.write(tmp, Constants.ROOT + "eval/{}.fasta".format("longer"), "fasta")


# collect_seq_gt_1021()


def crop_fasta():
    new_seqs = []
    files = []
    input_seq_iterator = SeqIO.parse(Constants.ROOT + "eval/{}.fasta".format("longer"), "fasta")
    for record in input_seq_iterator:
        for i in range(int(len(record.seq) / 1021) + 1):
            tmp = record.id.split("|")[1] + "_____" + str(i)
            id = "cafa|{}|{}".format(tmp, record.id.split("|")[2])
            seq = str(record.seq[i * 1021:(i * 1021) + 1021])
            new_seqs.append(create_seqrecord(id=id, name=id, description="", seq=seq))
            files.append(tmp)
    SeqIO.write(new_seqs, Constants.ROOT + "eval/{}_cropped.fasta".format("test"), "fasta")
    pickle_save(files, Constants.ROOT + "eval/cropped_files")


# crop_fasta()

def merge_pts():
    files = pickle_load(Constants.ROOT + "eval/cropped_files")
    unique_files = {i.split("_____")[0] for i in files}
    levels = {i.split("_____")[1] for i in files}
    embeddings = [0, 32, 33]

    for i in unique_files:
        print(i)
        fasta = fasta_to_dictionary(Constants.ROOT + "eval/longer.fasta")
        tmp = []
        for j in levels:
            os_path = Constants.ROOT + 'bnm/{}_____{}.pt'.format(i, j)
            if os.path.isfile(os_path):
                tmp.append(torch.load(os_path))

        data = {'representations': {}, 'mean_representations': {}}
        for index in tmp:
            splits = index['label'].split("|")
            data['label'] = splits[0] + "|" + splits[1].split("_____")[0] + "|" + splits[2]

            for rep in embeddings:
                assert torch.equal(index['mean_representations'][rep], torch.mean(index['representations'][rep], dim=0))

                if rep in data['representations']:
                    data['representations'][rep] = torch.cat(
                        (data['representations'][rep], index['representations'][rep]))
                else:
                    data['representations'][rep] = index['representations'][rep]

        assert len(fasta[i][3]) == data['representations'][33].shape[0]

        for rep in embeddings:
            data['mean_representations'][rep] = torch.mean(data['representations'][rep], dim=0)

        print("saving {}".format(i))

        # print(data['representations'][33].shape)
        # print(data['representations'][0].shape)
        # print(data['representations'][32].shape)
        print(data['mean_representations'][32].shape)
        # print(len(fasta[i][3]))

        torch.save(data, Constants.ROOT + "merged/{}.pt".format(i))


merge_pts()

# script to delete and remake
def delete_from_processed(pth):
    files = pickle_load(Constants.ROOT + "eval/cropped_files")
    unique_files = {i.split("_____")[0] for i in files}
    for i in unique_files:
        os_pth = pth + "{}.pt".format(i)
        if os.path.exists(os_pth):
            os.remove(os_pth)
            print("removed " + i)


# delete_from_processed(pth=Constants.ROOT + "processed/")

# delete_from_processed(pth="/media/fbqc9/Icarus/processed/")


# x = pickle_load(Constants.ROOT + "eval/all_map").keys()
# sam = []
# psam = []
# for i in x:
#     os_pth = Constants.ROOT + "processed/{}.pt".format(i)
#     if os.path.exists(os_pth):
#         sam.append(os_pth)
#     else:
#         psam.append(os_pth)
