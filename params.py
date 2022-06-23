bio_kwargs = {
    'hidden1': 1000,
    'hidden2': 1000,
    'hidden3': 1000,
    'input_features_size': 1280,
    'num_classes': 3774,
    'fc2_out': 3000,
    'edge_type': 'dist_3',
    'pool': ''
}

mol_kwargs = {
    'hidden1': 1000,
    'hidden2': 800,
    'hidden3': 600,
    'input_features_size': 1280,
    'num_classes': 600,
    'fc2_out': 3000,
    'edge_type': 'cbrt',
    'pool': 'mean',
    'layers': 1
}

cc_kwargs = {
    'hidden1': 1000,
    'hidden2': 800,
    'hidden3': 1000,
    'input_features_size': 1280,
    'num_classes': 547,
    'fc2_out': 547,
    'edge_type': 'cbrt',
    'layers': 1
}

edge_types = set(['sequence_letters', 'pos', 'all',
                  'molecular_function', 'biological_process',
                  'cellular_component', 'sequence_features',
                  'names', 'protein'])