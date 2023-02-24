import os
import csv
import subprocess
import networkx as nx
import numpy as np
import obonet
import pandas as pd
from Bio.Seq import Seq
from Bio import SeqIO, SwissProt
from Bio.SeqRecord import SeqRecord

import Constants
from preprocessing.utils import pickle_save, pickle_load, get_sequence_from_pdb, fasta_for_msas, \
    count_proteins_biopython, count_proteins, fasta_for_esm, fasta_to_dictionary, read_dictionary, \
    get_proteins_from_fasta, create_seqrecord, read_test_set, alpha_seq_fasta_to_dictionary, collect_test, is_ok, \
    test_annotation, get_test_classes


def extract_id(header):
    return header.split('|')[1]


def compare_sequence(uniprot_fasta_file, save=False):
    """
    Script is used to compare sequences of uniprot & alpha fold.
    :param uniprot_fasta_file: input uniprot fasta file.
    :param save: whether to save the proteins that are similar & different.
    :return: None
    """
    identical = []
    unidentical = []
    detected = 0
    for seq_record in SeqIO.parse(uniprot_fasta_file, "fasta"):
        uniprot_id = extract_id(seq_record.id)
        # print("Uniprot ID is {} and sequence is {}".format(uniprot_id, str(seq_record.seq)))
        # Check if alpha fold predicted structure
        src = os.path.join(Constants.ROOT, "alphafold/AF-{}-F1-model_v2.pdb.gz".format(uniprot_id))
        if os.path.isfile(src):
            detected = detected + 1
            # compare sequence
            alpha_fold_seq = get_sequence_from_pdb(src, "A")
            uniprot_sequence = str(seq_record.seq)
            if alpha_fold_seq == uniprot_sequence:
                identical.append(uniprot_id)
            else:
                unidentical.append(uniprot_id)
    print("{} number of sequence structures detected, {} identical to uniprot sequence & {} "
          "not identical to uniprot sequence".format(detected, len(identical), len(unidentical)))
    if save:
        pickle_save(identical, Constants.ROOT + "uniprot/identical")
        pickle_save(unidentical, Constants.ROOT + "uniprot/unidentical")


def filtered_sequences(uniprot_fasta_file):
    """
         Script is used to create fasta files based on alphafold sequence, by replacing sequences that are different.
        :param uniprot_fasta_file: input uniprot fasta file.
        :return: None
        """
    identified = set(pickle_load(Constants.ROOT + "uniprot/identical"))
    unidentified = set(pickle_load(Constants.ROOT + "uniprot/unidentical"))

    input_seq_iterator = SeqIO.parse(uniprot_fasta_file, "fasta")
    identified_seqs = [record for record in input_seq_iterator if extract_id(record.id) in identified]

    input_seq_iterator = SeqIO.parse(uniprot_fasta_file, "fasta")
    unidentified_seqs = []
    for record in input_seq_iterator:
        uniprot_id = extract_id(record.id)
        if uniprot_id in unidentified:
            src = os.path.join(Constants.ROOT, "alphafold/AF-{}-F1-model_v2.pdb.gz".format(uniprot_id))
            record.seq = Seq(get_sequence_from_pdb(src))
            unidentified_seqs.append(record)

    new_seq = identified_seqs + unidentified_seqs
    print(len(identified_seqs), len(unidentified_seqs), len(new_seq))
    SeqIO.write(new_seq, Constants.ROOT + "uniprot/{}.fasta".format("cleaned"), "fasta")


def get_protein_go(uniprot_sprot_dat=None, save_path=None):
    """
    Get all the GO terms associated with a protein in a clean format
    Creates file with structure: ACCESSION, ID, DESCRIPTION, WITH_STRING, EVIDENCE, GO_ID
    :param uniprot_sprot_dat:
    :param save_path:
    :return: None
    """
    handle = open(uniprot_sprot_dat)
    all = [["ACC", "ID", "DESCRIPTION", "WITH_STRING", "EVIDENCE", "GO_ID", "ORGANISM", "TAXONOMY"]]
    for record in SwissProt.parse(handle):
        primary_accession = record.accessions[0]
        entry_name = record.entry_name
        cross_refs = record.cross_references
        organism = record.organism
        taxonomy = record.taxonomy_id
        for ref in cross_refs:
            if ref[0] == "GO":
                assert len(ref) == 4
                go_id = ref[1]
                description = ref[2]
                evidence = ref[3].split(":")
                with_string = evidence[1]
                evidence = evidence[0]
                all.append(
                    [primary_accession, entry_name, description, with_string,
                     evidence, go_id, organism, taxonomy])
    with open(save_path, "w") as f:
        wr = csv.writer(f, delimiter='\t')
        wr.writerows(all)


def generate_go_counts(fname="", go_graph="", cleaned_proteins =None):
    """
       Get only sequences that meet the criteria of sequence length and sequences.
       :param cleaned_proteins: proteins filtered for alphafold sequence
       :param go_graph: obo-basic graph
       :param fname: accession2go file
       :param chains: the proteins we are considering
       :return: None
    """

    df = pd.read_csv(fname, delimiter="\t")

    df = df[df['EVIDENCE'].isin(Constants.exp_evidence_codes)]
    df = df[df['ACC'].isin(cleaned_proteins)]

    protein2go = {}
    go2info = {}
    # for index, row in df.iterrows():
    for line_number, (index, row) in enumerate(df.iterrows()):
        acc = row[0]
        evidence = row[4]
        go_id = row[5]

        # if (acc in chains) and (go_id in go_graph) and (go_id not in Constants.root_terms):
        if go_id in go_graph:
            if acc not in protein2go:
                protein2go[acc] = {'goterms': [go_id], 'evidence': [evidence]}
            namespace = go_graph.nodes[go_id]['namespace']
            go_ids = nx.descendants(go_graph, go_id)
            go_ids.add(go_id)
            go_ids = go_ids.difference(Constants.root_terms)
            for go in go_ids:
                protein2go[acc]['goterms'].append(go)
                protein2go[acc]['evidence'].append(evidence)
                name = go_graph.nodes[go]['name']
                if go not in go2info:
                    go2info[go] = {'ont': namespace, 'goname': name, 'accessions': set([acc])}
                else:
                    go2info[go]['accessions'].add(acc)
    return protein2go, go2info


def one_line_format(input_file, dir):
    """
         Script takes the mm2seq cluster output and converts to representative seq1, seq2, seq3 ....
        :param input_file: The clusters as csv file
        :return: None
        """
    data = {}
    with open(input_file) as file:
        lines = file.read().splitlines()
        for line in lines:
            x = line.split("\t")
            if x[0] in data:
                data[x[0]].append(x[1])
            else:
                data[x[0]] = list([x[1]])
    result = [data[i] for i in data]
    with open(dir + "/final_clusters.csv", "w") as f:
        wr = csv.writer(f, delimiter='\t')
        wr.writerows(result)


def get_prot_id_and_prot_name(cafa_proteins):
    print('Mapping CAFA PROTEINS')
    cafa_id_mapping = dict()
    with open(Constants.ROOT + 'uniprot/idmapping_selected.tab') as file:
        for line in file:
            _tmp = line.split("\t")[:2]
            if _tmp[1] in cafa_proteins:
                cafa_id_mapping[_tmp[1]] = _tmp[0]
            if len(cafa_id_mapping) == 97105:
                break
    return cafa_id_mapping


def target_in_swissprot_trembl_no_alpha():
    gps = ["LK_bpo", "LK_mfo", "LK_cco", "NK_bpo", "NK_mfo", "NK_cco"]
    targets = set()
    for ts in gps:
        ts_func_old = read_test_set("/data_bp/pycharm/TransFunData/data_bp/195-200/{}".format(ts))
        targets.update(set([i[0] for i in ts_func_old]))

        ts_func_new = read_test_set("/data_bp/pycharm/TransFunData/data_bp/205-now/{}".format(ts))
        targets.update(set([i[0] for i in ts_func_new]))

    target = []
    for seq_record in SeqIO.parse(Constants.ROOT + "uniprot/uniprot_trembl.fasta", "fasta"):
        if extract_id(seq_record.id) in targets:
            target.append(seq_record)
            print(len(target))
            if len(target) == len(targets):
                break
    for seq_record in SeqIO.parse(Constants.ROOT + "uniprot/uniprot_sprot.fasta", "fasta"):
        if extract_id(seq_record.id) in targets:
            target.append(seq_record)
            print(len(target))
            if len(target) == len(targets):
                break
    SeqIO.write(target, Constants.ROOT + "uniprot/{}.fasta".format("target_and_sequence"), "fasta")


# target_in_swissprot_trembl_no_alpha()


def cluster_sequence(seq_id, proteins=None, add_target=False):
    """
         Script is used to cluster the proteins with mmseq2.
        :param threshold:
        :param proteins:
        :param add_target: Add CAFA targets
        :param input_fasta: input uniprot fasta file.
        :return: None

        1. sequence to cluster is cleaned sequence.
        2. Filter for only selected proteins
        3. Add proteins in target not in the filtered list:
            3.1
            3.2
        """
    input_fasta = Constants.ROOT + "uniprot/cleaned.fasta"
    print("Number of proteins in raw cleaned is {}".format(count_proteins(input_fasta)))
    print("Number of selected proteins in raw cleaned is {}".format(len(proteins)))
    wd = Constants.ROOT + "{}/mmseq".format(seq_id)
    if not os.path.exists(wd):
        os.mkdir(wd)
    if proteins:
        fasta_path = wd + "/fasta_{}".format(seq_id)
        if os.path.exists(fasta_path):
            input_fasta = fasta_path
        else:
            input_seq_iterator = SeqIO.parse(input_fasta, "fasta")
            cleaned_fasta = [record for record in input_seq_iterator if extract_id(record.id) in proteins]
            SeqIO.write(cleaned_fasta, fasta_path, "fasta")
            assert len(cleaned_fasta) == len(proteins)
            input_fasta = fasta_path
    # Add sequence for target not in the uniprotKB
    if add_target:
        cleaned_missing_target_sequence = Constants.ROOT + "uniprot/cleaned_missing_target_sequence.fasta"
        if os.path.exists(cleaned_missing_target_sequence):
            input_fasta = cleaned_missing_target_sequence
        else:
            missing_targets_205_now = []
            missing_targets_195_200 = []
            # Adding missing target sequence
            all_list = set([extract_id(i.id) for i in (SeqIO.parse(input_fasta, "fasta"))])
            extra_alpha_fold = alpha_seq_fasta_to_dictionary(Constants.ROOT + "uniprot/alphafold_sequences.fasta")
            extra_trembl = fasta_to_dictionary(Constants.ROOT + "uniprot/target_and_sequence.fasta",
                                               identifier='protein_id')
            for ts in Constants.TEST_GROUPS:
                ts_func_old = read_test_set("/data_bp/pycharm/TransFunData/data_bp/195-200/{}".format(ts))
                ts_func_old = set([i[0] for i in ts_func_old])

                ts_func_new = read_test_set("/data_bp/pycharm/TransFunData/data_bp/205-now/{}".format(ts))
                ts_func_new = set([i[0] for i in ts_func_new])

                print("Adding 195-200 {}".format(ts))
                for _id in ts_func_old:
                    # Alphafold sequence always takes precedence
                    if _id not in all_list:
                        if _id in extra_alpha_fold:
                            _mp = extra_alpha_fold[_id]
                            missing_targets_195_200.append(SeqRecord(id=_mp[0].replace("AFDB:", "").
                                                                     replace("AF-", "").
                                                                     replace("-F1", ""),
                                                                     name=_mp[1],
                                                                     description=_mp[2],
                                                                     seq=_mp[3]))
                            # print("found {} in alphafold".format(_id))
                        elif _id in extra_trembl:
                            _mp = extra_trembl[_id]
                            missing_targets_195_200.append(SeqRecord(id=_mp[0],
                                                                     name=_mp[1],
                                                                     description=_mp[2],
                                                                     seq=_mp[3]))
                            # print("found {} in trembl".format(_id))
                        else:
                            print("Found in none for {}".format(_id))

                print("Adding 205-now {}".format(ts))
                for _id in ts_func_new:
                    # Alphafold sequence always takes precedence
                    if _id not in all_list:
                        if _id in extra_alpha_fold:
                            _mp = extra_alpha_fold[_id]
                            missing_targets_205_now.append(SeqRecord(id=_mp[0].replace("AFDB:", "").
                                                                     replace("AF-", "").
                                                                     replace("-F1", ""),
                                                                     name=_mp[1],
                                                                     description=_mp[2],
                                                                     seq=_mp[3]))
                        # print("found {} in alphafold".format(_id))
                        elif _id in extra_trembl:
                            _mp = extra_trembl[_id]
                            missing_targets_205_now.append(SeqRecord(id=_mp[0],
                                                                     name=_mp[1],
                                                                     description=_mp[2],
                                                                     seq=_mp[3]))
                            # print("found {} in trembl".format(_id))
                        else:
                            print("Found in none for {}".format(_id))

            # save missing sequence
            SeqIO.write(missing_targets_195_200, Constants.ROOT + "uniprot/{}.fasta".format("missing_targets_195_200"),
                        "fasta")
            SeqIO.write(missing_targets_205_now, Constants.ROOT + "uniprot/{}.fasta".format("missing_targets_205_now"),
                        "fasta")

            input_seq_iterator = list(SeqIO.parse(input_fasta, "fasta"))
            SeqIO.write(input_seq_iterator + missing_targets_195_200 + missing_targets_205_now, Constants.ROOT +
                        "uniprot/{}.fasta".format("cleaned_missing_target_sequence"), "fasta")

            input_fasta = cleaned_missing_target_sequence

        # input_seq_iterator = SeqIO.parse(Constants.ROOT +
        #                 "uniprot/{}.fasta".format("cleaned_missing_target_sequence"), "fasta")
        #
        # cleaned_fasta = set()
        # for record in input_seq_iterator:
        #     if record.id.startswith("AFDB"):
        #         cleaned_fasta.add(record.id.split(':')[1].split('-')[1])
        #     else:
        #         cleaned_fasta.add(extract_id(record.id))
        #
        # print(len(collect_test() - cleaned_fasta), len(cleaned_fasta))

        print("Number of proteins in cleaned_missing_target_sequence is {}".format(count_proteins(input_fasta)))

        command = "mmseqs createdb {} {} ; " \
                  "mmseqs cluster {} {} tmp --min-seq-id {};" \
                  "mmseqs createtsv {} {} {} {}.tsv" \
                  "".format(input_fasta, "targetDB", "targetDB", "outputClu", seq_id, "targetDB", "targetDB",
                            "outputClu", "outputClu")
        subprocess.call(command, shell=True, cwd="{}".format(wd))
        one_line_format(wd + "/outputClu.tsv", wd)


def accession2sequence(fasta_file=""):
    """
       Extract sequnce for each accession into dictionary.
       :param fasta_file:
       :return: None
    """
    input_seq_iterator = SeqIO.parse(fasta_file, "fasta")
    acc2seq = {extract_id(record.id): str(record.seq) for record in input_seq_iterator}
    pickle_save(acc2seq, Constants.ROOT + "uniprot/acc2seq")


def collect_test_clusters(cluster_path):
    # collect test and clusters
    total_test = collect_test()

    computed = pd.read_csv(cluster_path, names=['cluster'], header=None).to_dict()['cluster']
    computed = {i: set(computed[i].split('\t')) for i in computed}

    cafa3_cluster = set()
    new_cluster = set()
    train_cluster_indicies = []
    for i in computed:
        # cafa3
        if total_test[0].intersection(computed[i]):
            cafa3_cluster.update(computed[i])
        # new set
        elif total_test[1].intersection(computed[i]):
            new_cluster.update(computed[i])
        else:
            train_cluster_indicies.append(i)

    print(len(cafa3_cluster))
    print(len(new_cluster))
    exit()
    return test_cluster, train_cluster_indicies


def write_output_files(protein2go, go2info, seq_id):
    onts = ['molecular_function', 'biological_process', 'cellular_component']

    selected_goterms = {ont: set() for ont in onts}
    selected_proteins = set()

    print("Number of GO terms is {} proteins is {}".format(len(go2info), len(protein2go)))

    # for each go term count related proteins; if they are from 50 to 5000
    # then we can add them to our data_bp.
    for goterm in go2info:
        prots = go2info[goterm]['accessions']
        num = len(prots)
        namespace = go2info[goterm]['ont']
        if num >= 60:
            selected_goterms[namespace].add(goterm)
            selected_proteins = selected_proteins.union(prots)

    # Convert the accepted go terms into list, so they have a fixed order
    # Add the names of corresponding go terms.
    selected_goterms_list = {ont: list(selected_goterms[ont]) for ont in onts}
    selected_gonames_list = {ont: [go2info[goterm]['goname'] for goterm in selected_goterms_list[ont]] for ont in onts}

    # print the count of each go term
    for ont in onts:
        print("###", ont, ":", len(selected_goterms_list[ont]))

    terms = {}
    for ont in onts:
        terms['GO-terms-' + ont] = selected_goterms_list[ont]
        terms['GO-names-' + ont] = selected_gonames_list[ont]

    terms['GO-terms-all'] = selected_goterms_list['molecular_function'] + \
                            selected_goterms_list['biological_process'] + \
                            selected_goterms_list['cellular_component']

    terms['GO-names-all'] = selected_gonames_list['molecular_function'] + \
                            selected_goterms_list['biological_process'] + \
                            selected_goterms_list['cellular_component']

    pickle_save(terms, Constants.ROOT + 'go_terms')
    fasta_dic = fasta_to_dictionary(Constants.ROOT + "uniprot/cleaned.fasta")

    protein_list = set()
    terms_count = {'mf': set(), 'bp': set(), 'cc': set()}
    with open(Constants.ROOT + 'annot.tsv', 'wt') as out_file:
        tsv_writer = csv.writer(out_file, delimiter='\t')
        tsv_writer.writerow(["Protein", "molecular_function", "biological_process", "cellular_component", "all"])

        for chain in selected_proteins:
            goterms = set(protein2go[chain]['goterms'])
            if len(goterms) > 2 and is_ok(str(fasta_dic[chain][3])):
                # selected goterms
                mf_goterms = goterms.intersection(set(selected_goterms_list[onts[0]]))
                bp_goterms = goterms.intersection(set(selected_goterms_list[onts[1]]))
                cc_goterms = goterms.intersection(set(selected_goterms_list[onts[2]]))
                if len(mf_goterms) > 0 or len(bp_goterms) > 0 or len(cc_goterms) > 0:
                    terms_count['mf'].update(mf_goterms)
                    terms_count['bp'].update(bp_goterms)
                    terms_count['cc'].update(cc_goterms)
                    protein_list.add(chain)
                    tsv_writer.writerow([chain, ','.join(mf_goterms), ','.join(bp_goterms), ','.join(cc_goterms),
                                         ','.join(mf_goterms.union(bp_goterms).union(cc_goterms))])

    assert len(terms_count['mf']) == len(selected_goterms_list['molecular_function']) \
           and len(terms_count['mf']) == len(selected_goterms_list['molecular_function']) \
           and len(terms_count['mf']) == len(selected_goterms_list['molecular_function'])


    print("Creating Clusters")
    cluster_path = Constants.ROOT + "{}/mmseq/final_clusters.csv".format(seq_id)
    if not os.path.exists(cluster_path):
        cluster_sequence(seq_id, protein_list, add_target=True)

    # Remove test proteins & their cluster
    # Decided to remove irrespective of mf, bp | cc
    # It should be fine.
    # test_cluster, train_cluster_indicies = collect_test_clusters(cluster_path)
    # train_list = protein_list - test_cluster
    # assert len(protein_list.intersection(test_cluster)) == len(protein_list.intersection(collect_test())) == 0
    # print(len(protein_list), len(protein_list.intersection(cafa3)), len(protein_list.intersection(new_test)))

    print("Getting test cluster")
    cafa3, new_test = collect_test()

    train_list = protein_list - (cafa3.union(new_test))
    assert len(train_list.intersection(cafa3)) == len(train_list.intersection(new_test)) == 0

    validation_len = 6000 #int(0.2 * len(protein_list))
    validation_list = set()

    for chain in train_list:
        goterms = set(protein2go[chain]['goterms'])
        mf_goterms = set(goterms).intersection(set(selected_goterms_list[onts[0]]))
        bp_goterms = set(goterms).intersection(set(selected_goterms_list[onts[1]]))
        cc_goterms = set(goterms).intersection(set(selected_goterms_list[onts[2]]))

        if len(mf_goterms) > 0 and len(bp_goterms) > 0 and len(cc_goterms) > 0:
            validation_list.add(chain)

        if len(validation_list) >= validation_len:
            break

    pickle_save(validation_list, Constants.ROOT + '/{}/valid'.format(seq_id))
    train_list = train_list - validation_list

    print("Total number of train nrPDB=%d" % (len(train_list)))

    annot = pd.read_csv(Constants.ROOT + 'annot.tsv', delimiter='\t')
    for ont in onts + ['all']:
        _pth = Constants.ROOT + '{}/{}'.format(seq_id, ont)
        if not os.path.exists(_pth):
            os.mkdir(_pth)

        tmp = annot[annot[ont].notnull()][['Protein', ont]]
        tmp_prot_list = set(tmp['Protein'].to_list())
        tmp_prot_list = tmp_prot_list.intersection(train_list)

        computed = pd.read_csv(cluster_path, names=['cluster'], header=None)

        # train_indicies = computed.index.isin(train_cluster_indicies)

        # computed = computed.loc[train_indicies].to_dict()['cluster']
        computed = computed.to_dict()['cluster']
        computed = {ont: set(computed[ont].split('\t')) for ont in computed}

        new_computed = {}
        index = 0
        for i in computed:
            _tmp = tmp_prot_list.intersection(computed[i])
            if len(_tmp) > 0:
                new_computed[index] = _tmp
                index += 1

        _train = set.union(*new_computed.values())
        print("Total proteins for {} is {} in {} clusters".format(ont, len(_train), len(new_computed)))
        assert len(cafa3.intersection(_train)) == 0 and len(validation_list.intersection(_train)) == 0

        pickle_save(new_computed, _pth + '/train')


def pipeline(compare=False, curate_protein_goterms=False, generate_go_count=False,
             generate_msa=False, generate_esm=False, seq_id=0.3):
    """
    section 1
    1. First compare the sequence in uniprot and alpha fold and retrieve same sequence and different sequences.
    2. Replace mismatched sequences with alpha fold sequence & create the fasta from only alphafold sequences
    3. Just another comparison to be sure, we have only alphafold sequences.

    section 2
    GO terms associated with a protein

    section 3
    1. Convert Fasta to dictionary
    2. Read OBO graph
    3. Get proteins and related go terms & go terms and associated proteins

    :param generate_msa:
    :param generate_esm:
    :param generate_go_count:
    :param curate_protein_goterms:
    :param compare: Compare sequence between uniprot and alphafold
    :return:
    """

    # section 1
    if compare:
        compare_sequence(Constants.ROOT + "uniprot/uniprot_sprot.fasta", save=True)  # 1
        filtered_sequences(Constants.ROOT + "uniprot/uniprot_sprot.fasta")  # 2 create cleaned.fasta
        compare_sequence(Constants.ROOT + "uniprot/cleaned.fasta", save=False)  # 3

    # section 2
    if curate_protein_goterms:
        get_protein_go(uniprot_sprot_dat=Constants.ROOT + "uniprot/uniprot_sprot.dat",
                       save_path=Constants.ROOT + "protein2go.csv")  # 4 contains proteins and go terms.

    # section 3
    if generate_go_count:
        cleaned_proteins = fasta_to_dictionary(Constants.ROOT + "uniprot/cleaned.fasta")
        go_graph = obonet.read_obo(open(Constants.ROOT + "obo/go-basic.obo", 'r'))  # 5
        protein2go, go2info = generate_go_counts(fname=Constants.ROOT + "protein2go.csv", go_graph=go_graph,
                                                 cleaned_proteins=list(cleaned_proteins.keys()))
        pickle_save(protein2go, Constants.ROOT + "protein2go")
        pickle_save(go2info, Constants.ROOT + "go2info")

    protein2go = pickle_load(Constants.ROOT + "protein2go")
    go2info = pickle_load(Constants.ROOT + "go2info")

    print("Writing output for sequence identity {}".format(seq_id))
    write_output_files(protein2go, go2info, seq_id=seq_id)

    if generate_msa:
        fasta_file = Constants.ROOT + "cleaned.fasta"
        protein2go_primary = set(protein2go)
        fasta_for_msas(protein2go_primary, fasta_file)

    if generate_esm:
        fasta_file = Constants.ROOT + "cleaned.fasta"
        fasta_for_esm(protein2go, fasta_file)

# print(count_proteins(Constants.ROOT + "uniprot/{}.fasta".format("target_and_sequence")))
#


seq = [0.3, 0.5, 0.9, 0.95]
for i in seq:
    pipeline(compare=False,
             curate_protein_goterms=False,
             generate_go_count=False,
             generate_msa=False,
             generate_esm=False,
             seq_id=i)


exit()
groups = ['molecular_function', 'cellular_component', 'biological_process']

for i in seq:
    for j in groups:

        train = pd.read_pickle(Constants.ROOT + "{}/{}/train.pickle".format(i, j))
        valid = set(pd.read_pickle(Constants.ROOT + "{}/{}/valid.pickle".format(i, j)))
        test_cluster,_ = collect_test_clusters(Constants.ROOT + "{}/mmseq/final_clusters.csv".format(i))
        test = collect_test()
        print(i, j, len(test_cluster), len(test), len(test_cluster - test), len(test - test_cluster))

        # assert len(train.intersection(test_cluster)) == 0
        # assert len(train.intersection(test)) == 0

        assert len(valid.intersection(test_cluster)) == 0
        assert len(valid.intersection(test)) == 0
