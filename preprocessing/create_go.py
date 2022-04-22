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
    get_proteins_from_fasta, create_seqrecord


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
    mapping = pickle_load(Constants.ROOT + "uniprot/mapping_targets_name")
    cafa_id_mapping_reverse = pickle_load("cafa_id_mapping_reverse")
    for seq_record in SeqIO.parse(uniprot_fasta_file, "fasta"):
        # uniprot_id = cafa_id_mapping_reverse[mapping[seq_record.id]]#.split("'t\'")[1]
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
                    [primary_accession, entry_name, description, with_string, evidence, go_id, organism, taxonomy])
    with open(save_path, "w") as f:
        wr = csv.writer(f, delimiter='\t')
        wr.writerows(all)


def generate_go_counts(fname="", go_graph=""):
    """
       Get only sequences that meet the criteria of sequence length and sequences.
       :param go_graph: obo-basic graph
       :param fname: accession2go file
       :param chains: the proteins we are considering
       :return: None
    """

    df = pd.read_csv(fname, delimiter="\t")

    df = df[df['EVIDENCE'].isin(Constants.exp_evidence_codes)]
    # df = df[df['ACC'].isin(chains)]

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


def extract_cafa_4_targets(fasta_path):
    '''
    Add target fasta to our fasta file
    :param fasta_path:
    :return:
    '''
    cafa_proteins = dict()
    mapping = dict()

    for i in Constants.CAFA_TARGETS:
        mapping.update(read_dictionary(Constants.ROOT + 'CAFA4-export/MappingFiles/mapping.{}.map'.format(i)))
        fasta = Constants.ROOT + 'CAFA4-export/TargetFiles/sp_species.{}.tfa'.format(i)
        input_seq_iterator = SeqIO.parse(fasta, "fasta")

        for record in input_seq_iterator:
            cafa_proteins[mapping[record.id]] = record.seq

    if not os.path.exists(Constants.ROOT + "uniprot/id_mapping"):
        prot_id_to_prot_name = get_prot_id_and_prot_name(cafa_proteins.keys())
        pickle_save(prot_id_to_prot_name, Constants.ROOT + "uniprot/id_mapping")
    prot_id_to_prot_name = pickle_load(Constants.ROOT + "uniprot/id_mapping")

    xx = fasta_to_dictionary(fasta_path, identifier='protein_name')
    _difference = set(cafa_proteins.keys()).difference(set(list(xx.keys())))

    difference = []
    for diff in _difference:
        if diff in prot_id_to_prot_name:
            difference.append(create_seqrecord(name=diff, seq=cafa_proteins[diff], id=prot_id_to_prot_name[diff]))
        else:
            difference.append(create_seqrecord(name=diff, seq=cafa_proteins[diff], id=diff))

    cleaned_seq_iterator = list(SeqIO.parse(fasta_path, "fasta"))
    new_seq = cleaned_seq_iterator + difference

    SeqIO.write(new_seq, Constants.ROOT + "uniprot/{}.fasta".format("cleaned_targets"), "fasta")
    pickle_save(cafa_proteins, Constants.ROOT + "uniprot/all_cafa_target_proteins")


def cluster_sequence(seq_id, proteins=None, add_target=False):
    """
         Script is used to cluster the proteins with mmseq2.
        :param threshold:
        :param proteins:
        :param add_target: Add CAFA targets
        :param input_fasta: input uniprot fasta file.
        :return: None
        """
    input_fasta = Constants.ROOT + "uniprot/cleaned.fasta"
    wd = Constants.ROOT + "{}/mmseq".format(seq_id)
    if not os.path.exists(wd):
        os.mkdir(wd)
    if proteins:
        fasta_path = wd + "/fasta_{}".format(seq_id)
        if os.path.exists(fasta_path):
            input_fasta = fasta_path
        else:
            input_seq_iterator = SeqIO.parse(input_fasta, "fasta")
            cleaned_fasta = [record for record in input_seq_iterator if
                             is_ok(str(record.seq)) and extract_id(record.id) in proteins]
            SeqIO.write(cleaned_fasta, fasta_path, "fasta")
            input_fasta = fasta_path
    if add_target:
        target_fasta_path = Constants.ROOT + "uniprot/{}.fasta".format("cleaned_targets")
        target_path = Constants.ROOT + "uniprot/all_cafa_target_proteins"
        if os.path.exists(target_fasta_path) and os.path.exists(target_path):
            input_fasta = Constants.ROOT + "uniprot/cleaned_targets.fasta"
        else:
            extract_cafa_4_targets(input_fasta)
            input_fasta = Constants.ROOT + "uniprot/cleaned_targets.fasta"

    command = "mmseqs createdb {} {} ; " \
              "mmseqs cluster {} {} tmp --min-seq-id {};" \
              "mmseqs createtsv {} {} {} {}.tsv" \
              "".format(input_fasta, "targetDB", "targetDB", "outputClu", seq_id, "targetDB", "targetDB",
                        "outputClu", "outputClu")
    subprocess.call(command, shell=True, cwd="{}".format(wd))
    one_line_format(wd + "/outputClu.tsv", wd)


def is_ok(seq, MINLEN=49, MAXLEN=1001):
    """
       Checks if sequence is of good quality
       :param MAXLEN:
       :param MINLEN:
       :param seq:
       :return: None
       """
    if len(seq) < MINLEN or len(seq) > MAXLEN:
        return False
    for c in seq:
        if c in Constants.INVALID_ACIDS:
            return False
    return True


def filter_sequences(uniprot_fasta_file, max_len, min_len):
    """
       Get only sequences that meet the criteria of sequence length and sequences.
       :param max_len:
       :param min_len:
       :param uniprot_fasta_file:
       :return: None
       """
    input_seq_iterator = SeqIO.parse(uniprot_fasta_file, "fasta")
    cleaned = [record.id.split("|")[1] for record in input_seq_iterator if is_ok(str(record.seq))]
    pickle_save(cleaned, Constants.ROOT + "anotated")


def accession2sequence(fasta_file=""):
    """
       Extract sequnce for each accession into dictionary.
       :param fasta_file:
       :return: None
    """
    input_seq_iterator = SeqIO.parse(fasta_file, "fasta")
    acc2seq = {extract_id(record.id): str(record.seq) for record in input_seq_iterator}
    pickle_save(acc2seq, Constants.ROOT + "uniprot/acc2seq")


def load_cluster_proteins(seq_id):
    def get_position(row, pos, column, split):
        primary = row[column].split(split)[pos]
        return primary

    def count_members(row, column, split):
        primary = row[column].split(split)
        return len(primary)

    cluster = pd.read_csv(Constants.ROOT + "{}/mmseq/final_clusters.csv".format(seq_id),
                          names=['cluster'], header=None)
    cluster['rep'] = cluster.apply(lambda row: get_position(row, 0, 'cluster', '\t'), axis=1)
    cluster['count'] = cluster.apply(lambda row: count_members(row, 'cluster', '\t'), axis=1)

    return pd.Series(cluster['count'].values, index=cluster['rep']).to_dict()


def write_output_files(protein2go, go2info, seq_id):
    onts = ['molecular_function', 'biological_process', 'cellular_component']

    selected_goterms = {ont: set() for ont in onts}
    selected_proteins = set()

    # for each go term count related proteins; if they are from 50 to 5000
    # then we can add them to our data.
    for goterm in go2info:
        prots = go2info[goterm]['accessions']
        num = len(prots)
        namespace = go2info[goterm]['ont']
        if num >= 50:
            selected_goterms[namespace].add(goterm)
            selected_proteins = selected_proteins.union(prots)

    # Convert the accepted go terms into list, so they have a fixed order
    # Add the names of corresponding go terms.
    selected_goterms_list = {ont: list(selected_goterms[ont]) for ont in onts}
    selected_gonames_list = {ont: [go2info[goterm]['goname'] for goterm in selected_goterms_list[ont]] for ont in onts}

    # print the count of each go term
    for ont in onts:
        print("###", ont, ":", len(selected_goterms_list[ont]))

    # get all annotations
    protein_list = set()

    terms = {}
    for ont in onts:
        terms['GO-terms-' + ont] = selected_goterms_list[ont]
        terms['GO-names-' + ont] = selected_gonames_list[ont]

    pickle_save(terms, Constants.ROOT + '{}/term2name'.format(seq_id))

    with open(Constants.ROOT + '{}/annot.tsv'.format(seq_id), 'wt') as out_file:
        tsv_writer = csv.writer(out_file, delimiter='\t')
        tsv_writer.writerow(["Protein", "molecular_function", "biological_process", "cellular_component", "all"])

        for chain in selected_proteins:
            goterms = set(protein2go[chain]['goterms'])
            if len(goterms) > 2:
            # selected goterms
                mf_goterms = goterms.intersection(set(selected_goterms_list[onts[0]]))
                bp_goterms = goterms.intersection(set(selected_goterms_list[onts[1]]))
                cc_goterms = goterms.intersection(set(selected_goterms_list[onts[2]]))
                if len(mf_goterms) > 0 or len(bp_goterms) > 0 or len(cc_goterms) > 0:
                    protein_list.add(chain)
                    tsv_writer.writerow([chain, ','.join(mf_goterms), ','.join(bp_goterms), ','.join(cc_goterms), ','.join(mf_goterms.union(bp_goterms).union(cc_goterms))])

    print("Creating Clusters")
    cluster_path = Constants.ROOT + "{}/mmseq/final_clusters.csv".format(seq_id)
    if not os.path.exists(cluster_path):
        cluster_sequence(seq_id, protein_list, add_target=True)

    exit()
    cluster = load_cluster_proteins(seq_id=seq_id)
    protein_list = list(protein_list.intersection(set(cluster)))

    # # # #
    np.random.seed(1234)
    np.random.shuffle(protein_list)
    print("Total number of annot nrPDB=%d" % (len(protein_list)))

    # select test set based in 30% sequence identity
    test_list = set()
    i = 0
    while len(test_list) < 5000 and i < len(protein_list):
        goterms = protein2go[protein_list[i]]['goterms']
        # selected goterms
        mf_goterms = set(goterms).intersection(set(selected_goterms_list[onts[0]]))
        bp_goterms = set(goterms).intersection(set(selected_goterms_list[onts[1]]))
        cc_goterms = set(goterms).intersection(set(selected_goterms_list[onts[2]]))
        if len(mf_goterms) > 0 and len(bp_goterms) > 0 and len(cc_goterms) > 0:
            if cluster[protein_list[i]] < 5:
                test_list.add(protein_list[i])
        i += 1

    protein_list = list(set(protein_list).difference(test_list))

    print("Total number of train nrPDB=%d" % (len(protein_list)))
    print("Total number of test nrPDB=%d" % (len(test_list)))

    annot = pd.read_csv('/data/pycharm/TransFunData/data/0.3/annot.tsv', delimiter='\t')

    for i in onts + ['all']:
        _pth = Constants.ROOT + '{}/{}'.format(seq_id, i)
        if not os.path.exists(_pth):
            os.mkdir(_pth)

        tmp = annot[annot[i].notnull()][['Protein', i]]
        tmp_prot_list = set(tmp['Protein'].to_list())
        assert set(test_list) == set(test_list).intersection(set(tmp_prot_list))

        tmp_prot_list = list(tmp_prot_list.intersection(protein_list))

        np.random.shuffle(tmp_prot_list)
        idx = int(0.8 * len(tmp_prot_list))

        pickle_save(test_list, _pth + '/test')
        pickle_save(tmp_prot_list[:idx], _pth + '/train')
        pickle_save(tmp_prot_list[idx:], _pth + '/valid')

    # print(annot[annot['all'].notnull()][['Protein', i]])


def pipeline(compare=False, curate_protein_goterms=False, generate_go_count=False,
             generate_msa=False, generate_esm=False):
    """
    1. First compare the sequence in uniprot and alpha fold and retrieve same sequence and different sequences.
    2. Replace mismatched sequences with alpha fold sequence & create the fasta from only alphafold sequences
    3. Just another comparison to be sure, we have only alphafold sequences.

    :param generate_msa:
    :param generate_esm:
    :param generate_go_count:
    :param curate_protein_goterms:
    :param compare: Compare sequence between uniprot and alphafold
    :return:
    """

    if compare:
        compare_sequence(Constants.ROOT + "uniprot/uniprot_sprot.fasta", save=True) #1
        filtered_sequences(Constants.ROOT + "uniprot/uniprot_sprot.fasta") #2
        compare_sequence(Constants.ROOT + "uniprot/cleaned.fasta", save=False) #3

    if curate_protein_goterms:
        get_protein_go(uniprot_sprot_dat=Constants.ROOT + "uniprot/uniprot_sprot.dat",
                       save_path=Constants.ROOT + "protein2go.csv") #4

    if generate_go_count:
        go_graph = obonet.read_obo(open(Constants.ROOT + "obo/go-basic.obo", 'r')) #5
        protein_go = Constants.ROOT + "protein2go.csv"
        protein2go, go2info = generate_go_counts(fname=protein_go, go_graph=go_graph)
        pickle_save(protein2go, Constants.ROOT + "protein2go")
        pickle_save(go2info, Constants.ROOT + "go2info")

    protein2go = pickle_load(Constants.ROOT + "protein2go")
    go2info = pickle_load(Constants.ROOT + "go2info")

    print("Writing output")
    write_output_files(protein2go, go2info, seq_id=0.95)

    if generate_msa:
        fasta_file = Constants.ROOT + "cleaned.fasta"
        protein2go_primary = set(protein2go)
        fasta_for_msas(protein2go_primary, fasta_file)

    if generate_esm:
        fasta_file = Constants.ROOT + "cleaned.fasta"
        fasta_for_esm(protein2go, fasta_file)


pipeline(compare=False,
         curate_protein_goterms=False,
         generate_go_count=False,
         generate_msa=False,
         generate_esm=False)


# print(get_proteins_from_fasta("/data/pycharm/TransFunData/data/0.95/mmseq/fasta_0.95")[67594])
