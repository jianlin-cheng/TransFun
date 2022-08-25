bio_kwargs = {
    'hidden1': 128,
    'hidden2': 64,
    'hidden3': 32,
    'input_features_size': 1280,
    'num_classes': 3774,
    'fc2_out': 4000,
    'edge_type': 'sqrt',
    'layers': 1,
}

mol_kwargs = {
    'hidden1': 1000,
    'hidden2': 1000,
    'hidden3': 1000,
    'input_features_size': 1280,
    'num_classes': 3774,
    'fc2_out': 3000,
    'edge_type': 'cbrt'
}

cc_kwargs = {
    'hidden1': 128,
    'hidden2': 64,
    'hidden3': 32,
    'input_features_size': 1280,
    'num_classes': 547,
    'fc2_out': 550,
    'edge_type': 'cbrt',
    'layers': 1,
}

edge_types = set(['sqrt', 'cbrt', 'dist_3', 'dist_4', 'dist_6', 'dist_10', 'dist_12',
                  'molecular_function', 'biological_process', 'cellular_component',
                  'all', 'names', 'sequence_letters', 'protein', 'ptr', 'sequence_features'])