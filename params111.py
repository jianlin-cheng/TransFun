bio_kwargs = {
    'hidden': 16,
    'input_features_size': 1280,
    'num_classes': 4784,
    'edge_type': 'cbrt',
    'edge_features': 0,
    'egnn_layers': 8,
    'layers': 1,
    'device': 'cuda',
    'wd': 0.001
}

mol_kwargs = {
    'hidden': 8,
    'input_features_size': 1280,
    'num_classes': 762,
    'edge_type': 'cbrt',
    'edge_features': 0,
    'egnn_layers': 8,
    'layers': 1,
    'device': 'cuda',
    'wd': 0.001
}

cc_kwargs = {
    'hidden': 8,
    'input_features_size': 1280,
    'num_classes': 659,
    'edge_type': 'cbrt',
    'edge_features': 0,
    'egnn_layers': 8,
    'layers': 1,
    'device': 'cuda',
    'wd': 0.001
}

edge_types = set(['sqrt', 'cbrt', 'dist_3', 'dist_4', 'dist_6', 'dist_10', 'dist_12',
                  'molecular_function', 'biological_process', 'cellular_component',
                  'all', 'names', 'sequence_letters', 'ptr', 'sequence_features'])