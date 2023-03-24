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
1. To predict protein function with protein structures in the PDB format as input (note: protein sequences are automatically extracted from the PDB files).
```
    python predict.py --data-path data --ontology cellular_component --input-type pdb --pdb-path data/alphafold --output result.txt --cut-off 0.5
```

2. To predict protein function with protein sequences in the fasta format and protein structures in the PDB format as input: 
```
    python predict.py --data-path data --ontology cellular_component --input-type fasta --pdb-path data/alphafold --fasta-path data/sequence.fasta--output result.txt --cut-off 0.5
```

3. Full prediction command: 
```
Predict protein functions with TransFun

optional arguments:
  -h, --help            show this help message and exit
  --data-path DATA_PATH
                        Path to data files
  --ontology ONTOLOGY   Path to data files
  --no-cuda NO_CUDA     Disables CUDA training.
  --batch-size BATCH_SIZE
                        Batch size.
  --input-type {fasta,pdb}
                        Input Data: fasta file or PDB files
  --fasta-path FASTA_PATH
                        Path to Fasta
  --pdb-path PDB_PATH   Path to directory of PDBs
  --cut-off CUT_OFF     Cut of to report function
  --output OUTPUT       File to save output
  
```

4. Example Prediction: 
```
    python predict.py --data-path data --ontology cellular_component --input-type pdb --pdb-path test/pdbs/ --output result.txt
```

5. Example Prediction: 
```
    python predict.py --data-path data --ontology cellular_component --input-type pdb --pdb-path test/pdbs/ --output result.txt
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
