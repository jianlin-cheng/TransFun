residues = {
    "A": 0, "C": 1, "D": 2, "E": 3, "F": 4, "G": 5, "H": 6, "I": 7, "K": 8, "L": 9, "M": 10,
    "N": 11, "P": 12, "Q": 13, "R": 14, "S": 15, "T": 16, "V": 17, "W": 18, "Y": 19, "X": 20,
}

INVALID_ACIDS = {'U', 'O', 'B', 'Z', 'J', 'X', '*'}

amino_acids = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E",
    "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F",
    "PRO": "P", "PYL": "O", "SER": "S", "SEC": "U", "THR": "T", "TRP": "W", "TYR": "Y",
    "VAL": "V", "ASX": "B", "GLX": "Z", "XAA": "X", "XLE": "J"
}

root_terms = {'GO:0008150', 'GO:0003674', 'GO:0005575'}

exp_evidence_codes = {'EXP', 'IDA', 'IPI', 'IMP', 'IGI', 'IEP', 'TAS', 'IC'}
exp_evidence_codes = set([
    'EXP', 'IDA', 'IPI', 'IMP', 'IGI', 'IEP', 'TAS', 'IC',
    'HTP', 'HDA', 'HMP', 'HGI', 'HEP'])


ROOT = "/data/pycharm/TransFunData/data/"


# CAFA4 Targets
CAFA_TARGETS = {'287', '3702', '4577', '6239', '7227', '7955', '9606', '9823', '10090', '10116', '44689', '83333',
                '99287', '226900', '243273', '284812', '559292'}


NAMESPACES = {
    'cc': 'cellular_component',
    'mf': 'molecular_function',
    'bp': 'biological_process'
}