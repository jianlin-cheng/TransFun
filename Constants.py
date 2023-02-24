residues = {
    "A": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6, "H": 7, "I": 8, "K": 9, "L": 10, "M": 11,
    "N": 12, "P": 13, "Q": 14, "R": 15, "S": 16, "T": 17, "V": 18, "W": 19, "Y": 20
}

INVALID_ACIDS = {"U", "O", "B", "Z", "J", "X", "*"}

amino_acids = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E",
    "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F",
    "PRO": "P", "PYL": "O", "SER": "S", "SEC": "U", "THR": "T", "TRP": "W", "TYR": "Y",
    "VAL": "V", "ASX": "B", "GLX": "Z", "XAA": "X", "XLE": "J"
}

root_terms = {"GO:0008150", "GO:0003674", "GO:0005575"}

exp_evidence_codes = {"EXP", "IDA", "IPI", "IMP", "IGI", "IEP", "TAS", "IC"}
exp_evidence_codes = set([
    "EXP", "IDA", "IPI", "IMP", "IGI", "IEP", "TAS", "IC",
    "HTP", "HDA", "HMP", "HGI", "HEP"])

ROOT = "/home/fbqc9/PycharmProjects/TransFunData/data/"
# ROOT = "D:/Workspace/python-3/transfunData/data_bp/"
# ROOT = "/data_bp/pycharm/TransFunData/data_bp/"

# CAFA4 Targets
CAFA_TARGETS = {"287", "3702", "4577", "6239", "7227", "7955", "9606", "9823", "10090", "10116", "44689", "83333",
                "99287", "226900", "243273", "284812", "559292"}

NAMESPACES = {
    "cc": "cellular_component",
    "mf": "molecular_function",
    "bp": "biological_process"
}

FUNC_DICT = {
    'cc': 'GO:0005575',
    'mf': 'GO:0003674',
    'bp': 'GO:0008150'}

BENCH_DICT = {
    'cc': "CCO",
    'mf': 'MFO',
    'bp': 'BPO'
}

NAMES = {
    "cc": "Cellular Component",
    "mf": "Molecular Function",
    "bp": "Biological Process"
}

TEST_GROUPS = ["LK_bpo", "LK_mfo", "LK_cco", "NK_bpo", "NK_mfo", "NK_cco"]

Final_thresholds = {
    "cellular_component": 0.50,
    "molecular_function": 0.90,
    "biological_process": 0.50
}

TFun_Plus_thresholds = {
    "cellular_component": (0.13, 0.87),
    "molecular_function": (0.36, 0.61),
    "biological_process": (0.38, 0.62)
}
