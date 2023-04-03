# TransFun for Protein Function Prediction
TransFun is a method using a transformer-based protein language model and 3D-equivariant graph neural networks (EGNN) to distill information from both protein sequences and structures to predict protein function in terms of Gene Ontology (GO) terms. It extracts feature embeddings from protein sequences using a pre-trained protein language model (ESM) via transfer learning and combines them with 3D structures of proteins predicted by AlphaFold2 through EGNN to predict function. It achieved the state-of-the-art performance on the CAFA3 test dataset and a new test dataset.



## Installation
```
# clone project
git clone https://github.com/jianlin-cheng/TransFun.git
cd TransFun/

# download trained models and test sample
curl https://calla.rnet.missouri.edu/rnaminer/transfun/data --output data.zip
unzip data

# create conda environment
conda env create -f environment.yml
conda activate transfun
```


## Prediction
1. To predict protein function with protein structures in the PDB format as input (note: protein sequences are automatically extracted from the PDB files in the input pdb path).
```
    python predict.py --data-path path_to_store_intermediate_files --ontology GO_function_category --input-type pdb --pdb-path data/alphafold --output output_file --cut-off probability_threshold
```

2. To predict protein function with protein sequences in the fasta format and protein structures in the PDB format as input: 
```
    python predict.py --data-path path_to_store_intermediate_files --ontology GO_function_category --input-type fasta --pdb-path data/alphafold --fasta-path path_to_a_fasta_file --output result.txt --cut-off probability_threshold
```

3. Full prediction command: 
```
Predict protein functions with TransFun

optional arguments:
  -h, --help            Help message
  --data-path DATA_PATH
                        Path to store intermediate data files
  --ontology ONTOLOGY   GO function category: cellular_component, molecular_function, biological_process
  --no-cuda NO_CUDA     Disables CUDA training
  --batch-size BATCH_SIZE
                        Batch size
  --input-type {fasta,pdb}
                        Input data type: fasta file or PDB files
  --fasta-path FASTA_PATH
                        Path to a fasta containing one or more protein sequences
  --pdb-path PDB_PATH   Path to the directory of one or more protein structure files in the PDB format
  --cut-off CUT_OFF     Cut-off probability threshold to report function
  --output OUTPUT       A file to save output. All the predictions are stored in this file
  
```

4. An example of predicting cellular component of some proteins: 
```
    python predict.py --data-path data --ontology cellular_component --input-type pdb --pdb-path test/pdbs/ --output result.txt
```

5. An example of predicting molecular function of some proteins: 
```
    python predict.py --data-path data --ontology molecular_function --input-type pdb --pdb-path test/pdbs/ --output result.txt
```

## Reference
```
@Article{Boadu2023-et,
     title    = "Combining protein sequences and structures with transformers and
                 equivariant graph neural networks to predict protein function",
     author   = "Boadu, Frimpong and Cao, Hongyuan and Cheng, Jianlin",
     journal  = "bioRxiv",
     month    =  jan,
     year     =  2023
}
```
