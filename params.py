bio_kwargs = {
    'hidden': 16,
    'input_features_size': 1280,
    'num_classes': 3774,
    'edge_type': 'cbrt',
    'edge_features': 0,
    'egnn_layers': 12,
    'layers': 1,
    'device': 'cuda',
    'wd': 5e-4
}

mol_kwargs = {
    'hidden': 16,
    'input_features_size': 1280,
    'num_classes': 600,
    'edge_type': 'cbrt',
    'edge_features': 0,
    'egnn_layers': 12,
    'layers': 1,
    'device': 'cuda',
    'wd': 0.001
}

cc_kwargs = {
    'hidden': 16,
    'input_features_size': 1280,
    'num_classes': 547,
    'edge_type': 'cbrt',
    'edge_features': 0,
    'egnn_layers': 12,
    'layers': 1,
    'device': 'cuda',
    'wd': 0.001 #5e-4
}

edge_types = set(['sqrt', 'cbrt', 'dist_3', 'dist_4', 'dist_6', 'dist_10', 'dist_12',
                  'molecular_function', 'biological_process', 'cellular_component',
                  'all', 'names', 'sequence_letters', 'ptr', 'sequence_features'])
