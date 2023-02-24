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
conda env create -f packagelist.yml
conda activate transfun
```


## Requirements:
```
biopandas==0.3.0dev0
biopython==1.79
numpy==1.21.3
pandas==1.3.4
scipy==1.7.1
torch==1.10.0
```
Install [Transformer protein language models](https://github.com/facebookresearch/esm) by the following command:

```
pip install git+https://github.com/facebookresearch/esm.git
```


## Prediction

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

