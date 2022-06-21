bio_kwargs = {
    'hidden1': 1000,
    'hidden2': 1000,
    'hidden3': 1000,
    'input_features_size': 1280,
    'num_classes': 3774,
    'fc2_out': 3000,
    'edge_type': 'cbrt'
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
    'edge_type': 'cbrt'
}

data_keys = {'sqrt', 'cbrt', 'dist_3', 'dist_4', 'dist_6', 'dist_10', 'dist_12', 'sqrt_edge_attr', 'cbrt_edge_attr',
             'dist_3_edge_attr', 'dist_4_edge_attr', 'dist_6_edge_attr', 'dist_10_edge_attr', 'dist_12_edge_attr',
             'pos', 'molecular_function', 'biological_process', 'cellular_component',
             'names', 'sequence_letters', 'all', 'pos'}
