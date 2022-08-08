bio_kwargs = {
    'hidden1': 1000,
    'hidden2': 1000,
    'hidden3': 1000,
    'input_features_size': 1280,
    'num_classes': 3774,
    'fc2_out': 3000,
    'edge_type': 'sqrt',
    'layers': 2,
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
    'hidden1': 1000,
    'hidden2': 1000,
    'hidden3': 1000,
    'input_features_size': 1280,
    'num_classes': 3774,
    'fc2_out': 3000,
    'edge_type': 'dist_3'
}

edge_types = set(['sqrt', 'cbrt', 'dist_3', 'dist_4', 'dist_6', 'dist_10', 'dist_12'])