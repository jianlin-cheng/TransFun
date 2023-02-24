import os, subprocess
from typing import Any, Mapping, MutableMapping, Optional, Sequence, Union
from absl import logging
from tools import jackhmmer, parsers, residue_constants, msa_identifiers, hhblits
import numpy as np
import shutil
from absl import app
from absl import logging
import multiprocessing
from glob import glob
import sys

jackhmmer_binary_path = shutil.which('jackhmmer')
uniref90_database_path = "/data_bp/pycharm/Genetic_Databases/uniref90/uniref90.fasta"
mgnify_database_path = "/data_bp/pycharm/Genetic_Databases/mgnify/mgy_clusters_2018_12.fa"
small_bfd_database_path = "/data_bp/pycharm/Genetic_Databases/small_bfd/bfd-first_non_consensus_sequences.fasta"
hhblits_binary_path = "/data_bp/pycharm/Genetic_Databases/small_bfd/bfd-first_non_consensus_sequences.fasta"
uniclust30_database_path = "/data_bp/pycharm/Genetic_Databases/small_bfd/bfd-first_non_consensus_sequences.fasta"


FeatureDict = MutableMapping[str, np.ndarray]


def make_msa_features(msas: Sequence[parsers.Msa], combined_out_path: str) -> FeatureDict:
  """Constructs a feature dict of MSA features."""
  if not msas:
    raise ValueError('At least one MSA must be provided.')

  int_msa = []
  deletion_matrix = []
  uniprot_accession_ids = []
  species_ids = []
  seen_sequences = []
  name_identifiers = []
  for msa_index, msa in enumerate(msas):
    if not msa:
      raise ValueError(f'MSA {msa_index} must contain at least one sequence.')
    for sequence_index, sequence in enumerate(msa.sequences):
      if sequence in seen_sequences:
        continue
      seen_sequences.append(sequence)
      name_identifiers.append(msa.descriptions[sequence_index])
      int_msa.append([residue_constants.HHBLITS_AA_TO_ID[res] for res in sequence])
      deletion_matrix.append(msa.deletion_matrix[sequence_index])
      identifiers = msa_identifiers.get_identifiers(msa.descriptions[sequence_index])
      uniprot_accession_ids.append(identifiers.uniprot_accession_id.encode('utf-8'))
      species_ids.append(identifiers.species_id.encode('utf-8'))

  num_res = len(msas[0].sequences[0])
  num_alignments = len(int_msa)
  features = {}
  features['deletion_matrix_int'] = np.array(deletion_matrix, dtype=np.int32)
  features['msa'] = np.array(int_msa, dtype=np.int32)
  features['num_alignments'] = np.array([num_alignments] * num_res, dtype=np.int32)
  features['msa_uniprot_accession_identifiers'] = np.array(uniprot_accession_ids, dtype=np.object_)
  features['msa_species_identifiers'] = np.array(species_ids, dtype=np.object_)

  with open(combined_out_path, 'w') as f:
      for item in zip(seen_sequences, name_identifiers):
          f.write(">%s\n" % item[1])
          f.write("%s\n" % item[0])



def run_msa_tool(msa_runner, input_fasta_path: str, msa_out_path: str, msa_format: str, use_precomputed_msas: bool, base_path: str) -> Mapping[str, Any]:
  """Runs an MSA tool, checking if output already exists first."""
  if not use_precomputed_msas or not os.path.exists(msa_out_path):
    result = msa_runner.query(input_fasta_path, base_path=base_path)[0]
    with open(msa_out_path, 'w') as f:
      f.write(result[msa_format])
  else:
    logging.error('Reading MSA from file %s', msa_out_path)
    with open(msa_out_path, 'r') as f:
      result = {msa_format: f.read()}
  return result

class DataPipeline:
  """Runs the alignment tools and assembles the input features."""

  def __init__(self, jackhmmer_binary_path: str, hhblits_binary_path: str, uniref90_database_path: str, mgnify_database_path: str, small_bfd_database_path: Optional[str],  uniclust30_database_path: str, bfd_database_path: Optional[str]):
    """Initializes the data_bp pipeline."""

    self.jackhmmer_uniref90_runner = jackhmmer.Jackhmmer(binary_path=jackhmmer_binary_path, database_path=uniref90_database_path)

    #self.jackhmmer_small_bfd_runner = jackhmmer.Jackhmmer(binary_path=jackhmmer_binary_path, database_path=small_bfd_database_path)

    self.hhblits_bfd_uniclust_runner = hhblits.HHBlits( binary_path=hhblits_binary_path, databases=[bfd_database_path, uniclust30_database_path])

    self.jackhmmer_mgnify_runner = jackhmmer.Jackhmmer(binary_path=jackhmmer_binary_path, database_path=mgnify_database_path)

    self.use_precomputed_msas = False

    self.mgnify_max_hits = 501

    self.uniref_max_hits = 10000

  def process(self, input_fasta_path: str, msa_output_dir: str, base_path: str, protein: str, combine: bool, make_diverse: bool) -> FeatureDict:
    """Runs alignment tools on the input sequence and creates features."""

    uniref90_msa = "None"
    bfd_msa = "None"
    mgnify_msa = "None"

    combined_out_path = os.path.join(msa_output_dir, 'combined.a3m')
    diverse_out_path = os.path.join(msa_output_dir, 'diverse_{}.a3m')

    with open(input_fasta_path) as f:
      input_fasta_str = f.read()
    input_seqs, input_descs = parsers.parse_fasta(input_fasta_str)
    if len(input_seqs) != 1:
      raise ValueError(f'More than one input sequence found in {input_fasta_path}.')

    if os.path.isfile(combined_out_path):
        logging.error("Combined already generated for {}".format(input_fasta_path))
    else:
        uniref90_out_path = os.path.join(msa_output_dir, 'uniref90_hits.sto')
        if not os.path.isfile(uniref90_out_path):
            logging.error("Generating msa for {} from {}".format(protein, "uniref90"))
            jackhmmer_uniref90_result = run_msa_tool(self.jackhmmer_uniref90_runner, input_fasta_path, uniref90_out_path, 'sto', self.use_precomputed_msas, base_path=base_path)
            uniref90_msa = parsers.parse_stockholm(jackhmmer_uniref90_result['sto'])
            uniref90_msa = uniref90_msa.truncate(max_seqs=self.uniref_max_hits)
        else:
            if combine and not os.path.isfile(combined_out_path):
                logging.error("Loading msa for {} from {} @ {}".format(protein, "uniref90", uniref90_out_path))
                with open(uniref90_out_path, 'r') as f:
                    jackhmmer_uniref90_result = {'sto': f.read()}
                uniref90_msa = parsers.parse_stockholm(jackhmmer_uniref90_result['sto'])
                uniref90_msa = uniref90_msa.truncate(max_seqs=self.uniref_max_hits)

        mgnify_out_path = os.path.join(msa_output_dir, 'mgnify_hits.sto')
        if not os.path.isfile(mgnify_out_path):
            logging.error("Generating msa for {} from {}".format(protein, "mgnify"))
            jackhmmer_mgnify_result = run_msa_tool(self.jackhmmer_mgnify_runner, input_fasta_path, mgnify_out_path, 'sto', self.use_precomputed_msas, base_path=base_path)
            mgnify_msa = parsers.parse_stockholm(jackhmmer_mgnify_result['sto'])
            mgnify_msa = mgnify_msa.truncate(max_seqs=self.mgnify_max_hits)
        else:
            if combine and not os.path.isfile(combined_out_path):
                logging.error("Loading msa for {} from {} @ {}".format(protein, "mgnify", mgnify_out_path))
                with open(mgnify_out_path, 'r') as f:
                    jackhmmer_mgnify_result = {'sto': f.read()}
                mgnify_msa = parsers.parse_stockholm(jackhmmer_mgnify_result['sto'])
                mgnify_msa = mgnify_msa.truncate(max_seqs=self.mgnify_max_hits)

        bfd_out_path = os.path.join(msa_output_dir, 'bfd_uniclust_hits.a3m')
        if not os.path.isfile(bfd_out_path):
            logging.error("Generating msa for {} from {}".format(protein, "Bfd"))
            hhblits_bfd_uniclust_result = run_msa_tool(self.hhblits_bfd_uniclust_runner, input_fasta_path, bfd_out_path, 'a3m', self.use_precomputed_msas, base_path=base_path)
            bfd_msa = parsers.parse_a3m(hhblits_bfd_uniclust_result['a3m'])
            # jackhmmer_small_bfd_result = run_msa_tool(self.jackhmmer_small_bfd_runner, input_fasta_path, bfd_out_path, 'sto', self.use_precomputed_msas, base_path=base_path)
            # bfd_msa = parsers.parse_stockholm(jackhmmer_small_bfd_result['sto'])
        else:
            if combine and not os.path.isfile(combined_out_path):
                logging.error("Loading msa for {} from {} @ {}".format(protein, "small_bfd", bfd_out_path))
                with open(bfd_out_path, 'r') as f:
                    hhblits_small_bfd_result = {'a3m': f.read()}
                bfd_msa = parsers.parse_stockholm(hhblits_small_bfd_result['a3m'])
                #     jackhmmer_small_bfd_result = {'sto': f.read()}
                # bfd_msa = parsers.parse_stockholm(jackhmmer_small_bfd_result['sto'])


        msa_features = make_msa_features((uniref90_msa, bfd_msa, mgnify_msa), combined_out_path=combined_out_path)

    if make_diverse:
        if not os.path.isfile(diverse_out_path.format(64)):
            subprocess.call(
                'hhfilter -i {} -o {} -diff {}'.format(combined_out_path, diverse_out_path.format(64), 64), shell=True)

        if not os.path.isfile(diverse_out_path.format(128)):
            subprocess.call(
                'hhfilter -i {} -o {} -diff {}'.format(combined_out_path, diverse_out_path.format(128), 128), shell=True)

        if not os.path.isfile(diverse_out_path.format(256)):
            subprocess.call(
                'hhfilter -i {} -o {} -diff {}'.format(combined_out_path, diverse_out_path.format(256), 256), shell=True)

        if not os.path.isfile(diverse_out_path.format(512)):
            subprocess.call(
                'hhfilter -i {} -o {} -diff {}'.format(combined_out_path, diverse_out_path.format(512), 512), shell=True)

    # logging.info('Uniref90 MSA size: %d sequences.', len(uniref90_msa))
    # logging.info('BFD MSA size: %d sequences.', len(bfd_msa))
    # logging.info('MGnify MSA size: %d sequences.', len(mgnify_msa))




def run_main(directory):
    pipeline = DataPipeline(jackhmmer_binary_path, hhblits_binary_path, uniref90_database_path, mgnify_database_path, small_bfd_database_path, uniclust30_database_path, bfd_database_path)
    base_path = base+"{}".format(directory)
    logging.info("Generating for protein {}".format(directory))
    input_path = base_path+"/{}.fasta".format(directory)
    output_path = base_path+"/msas"
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    pipeline.process(input_fasta_path=input_path, msa_output_dir=output_path, base_path=base_path, protein=directory, combine=True, make_diverse=True)


base = "/storage/htc/bdm/Frimpong/TransFun/msa_files/two/{}/".format(sys.argv[1])
directories = [x for x in os.listdir(base)]

logging.info("Started")
pool = multiprocessing.Pool(4)
pool.map(run_main, directories)
pool.close()
