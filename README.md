# TransFun(Readme In progress)
Transformer for protein function prediction

## Install dependencies
```
# clone project
git clone https://github.com/jianlin-cheng/TransFun.git
cd TransFun/

# download trained models
wget https://calla.rnet.missouri.edu/rnaminer/transfun/data
unzip data

# create conda environment
conda env create -f environment.yml
conda activate transfun
```


## Prediction
1. To predict with PDBs only(note fasta sequence is extracted from PDB file).
```
    python predict.py --input-type pdb --pdb-path data/alphafold --output res --cut-off 0.5
```

2. To predict with fasta and PDBs: 
```
    python predict.py --input-type pdb --pdb-path data/alphafold --fasta-path data/sequence.fasta--output res --cut-off 0.5
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
  --add-ancestors ADD_ANCESTORS
                        Add ancestor terms to prediction
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
