# TransFun(Readme In progress)
Transformer for protein function prediction


Install dependencies

```bash
# clone project
git clone https://github.com/jianlin-cheng/TransFun.git
cd TransFun/

# download trained models
wget url

# create conda environment
conda env create -f environment.yml
conda activate transfun
```


## Prediction
python predict.py --input-type pdb --output res --cut-off 0.5
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
  --fasta FASTA         Path to Fasta
  --pdb PDB             Path to directory of PDBs
  --cut-off CUT_OFF     Cut of to report function
  --output OUTPUT       File to save output
  --add-ancestors ADD_ANCESTORS
                        Add ancestor terms to prediction

```


## Reference

