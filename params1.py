bio_kwargs = {
    'hidden': 16,
    'input_features_size': 1280,
    'num_classes': 3774,
    'fc2_out': 3000,
    'edge_type': 'cbrt',
    'edge_features': 0,
    'egnn_layers': 12,
    'layers': 1,
    'device': 'cuda'
}

mol_kwargs = {
    'hidden': 16,
    'input_features_size': 1280,
    'num_classes': 600,
    'fc2_out': 650,
    'edge_type': 'cbrt',
    'edge_features': 0,
    'egnn_layers': 12,
    'layers': 1,
    'device': 'cuda'
}

cc_kwargs = {
    'hidden': 16,
    'input_features_size': 1280,
    'num_classes': 547,
    'fc2_out': 550,
    'edge_type': 'cbrt',
    'edge_features': 0,
    'egnn_layers': 12,
    'layers': 1,
    'device': 'cuda'

}

edge_types = set(['sqrt', 'cbrt', 'dist_3', 'dist_4', 'dist_6', 'dist_10', 'dist_12',
                  'molecular_function', 'biological_process', 'cellular_component',
                  'all', 'names', 'sequence_letters', 'protein', 'ptr', 'sequence_features'])