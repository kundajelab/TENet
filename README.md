# Transcriptional Effector Network (TENet)
This repository contains code for the paper: [Prediction and design of transcriptional repressor domains with large-scale mutational scans and deep learning](TODO). 
By: Raeline Valbuena, AkshatKumar Nigam, Josh Tycko, Peter Suzuki, Kaitlyn Spees, Aradhana, Sophia Arana, Peter Du, Roshni A. Patel, Lacramiora Bintu, Anshul Kundaje, Michael C. Bassik

The primary goal of this repository is to provide users with the necessary tools to successfully train TENet using data from our screening experiments. Additionally, users can apply the trained model within an evolutionary algorithm (EA) framework to design novel transcriptional repressors.

- The list of initial domains, along with their sequences and classifications from our deep mutational scans, can be found [here](https://drive.google.com/file/d/1zDlblOi9r16k3Pmpf30UYcFnM4bms-2l/view?usp=sharing).
- The results of the HT-recruit screen are available [here](https://drive.google.com/file/d/1ZD2vPHar5mjUDK792QuBgWXZCJD8ZJpP/view?usp=sharing).
- Interactive UMAP visualizations of all domains, structured using different embeddings, can be downloaded from the following links:  
   (1) [Using ESM Embeddings (Figure 4D, left of the manuscript)](https://drive.google.com/file/d/1649fPY9hp3mmzuNY1jZ4zFtBlCS00z-z/view?usp=sharing)  
   (2) [Using TENet-based embeddings (Figure 4E, right of the manuscript)](https://drive.google.com/file/d/1lh0kG5H6HaKEq8KkF3otwzXwY2a2HEz5/view?usp=sharing)

![Overview of TENet](./misc/TENet.png)

## Prerequisites

Please ensure the following packages are installed prior to running any of the code provided within this repository:

- [Python 3.0 or higher](https://www.python.org/download/releases/3.0/) (this code was run on version 3.11.5)
- [PyTorch v2.2.0](https://pytorch.org/)
- [Numpy](https://pypi.org/project/numpy/)
- [Scipy](https://pypi.org/project/scipy/)
- [Matplotlib](https://matplotlib.org/stable/users/installing.html)
- [imbalanced-learn](https://pypi.org/project/imbalanced-learn/) 
- [ESM-2](https://github.com/facebookresearch/esm)
- [ESMFold](https://github.com/facebookresearch/esm)

To install these dependencies, run:

```bash
pip install torch numpy scipy matplotlib imbalanced-learn fair-esm 

# For installation of ESMFold
pip install "fair-esm[esmfold]"
# OpenFold and its remaining dependency
pip install 'dllogger @ git+https://github.com/NVIDIA/dllogger.git'
pip install 'openfold @ git+https://github.com/aqlaboratory/openfold.git@4b41059694619831a7db195b7e0988fc4ff3a307'
```

## File Navigator

- **create_data.py**: This script processes amino acid sequences, extracts features using ESM-2 embeddings, and computes contact maps and amino acid descriptors. The processed data is saved in chunks as pickle files in the `DATA/pickle_files` directory.
- **model.py**: Contains the architecture of the Transcriptional Effector Network (TENet), a deep learning model used for sequence processing and prediction tasks. It also includes code for training the model end-to-end on the provided data.
- **eval_model.py**: Loads the pre-trained TENet model and performs evaluation on the test data. This script computes predictions and evaluates the model’s performance by comparing predicted and actual values. Evaluation metrics, such as scatter plots and correlation values, are saved in the `PLOTS/` directory.
- **submit.sh**: A shell script for automating the training of TENet and subsequent evaluation of the trained model on a SLURM-based cluster. It calls `model.py` and `eval_model.py` for training and evaluation, respectively. Users must modify the `--account` flag to include their appropriate cluster account name before running `sbatch submit.sh`.
- **esmFold_bulk_structure_pred/**: A subdirectory for predicting protein structures in bulk using the ESMFold model.
  - **esmFold_bulk_pred.py**: A script designed to process large-scale protein sequence data in chunks using ESMFold for structure prediction. The results are saved as PDB files in the `pdb_files/` directory.
  - **submit_esmFoldBulk.sh**: A shell script to submit the `esmFold_bulk_pred.py` script to a SLURM-based cluster, enabling parallel processing of large datasets via job arrays. Adjust the `--account` flag and number of jobs as needed before running.
- **EA/**: A subdirectory implementing an Evolutionary Algorithm (EA) for protein sequence optimization.
  - **MAIN.py**: The main script that performs sequence optimization using an evolutionary approach. It generates mutant sequences, predicts their structures, and evaluates their fitness using a pre-trained model. The results, including PDB files and fitness scores, are saved in the `pdb_files/` and `OUTPUTS/` directories.
  - **submit.sh**: A shell script for submitting `MAIN.py` to a SLURM-based cluster, automating the evolutionary optimization process over multiple generations.
- **misc/**: A directory containing miscellaneous files, such as `TENet.png` (used for visualization in the documentation).
- **DATA/**: A directory where input data and processed outputs are stored. After running the scripts and training the model end-to-end, the following files and subdirectories are expected to exist:
  - `aa_descriptors.csv`: Contains amino acid descriptors utilized by the model.
  - `NEW_PARSE_4.csv`: Input CSV file containing amino acid sequences and experimental data.
  - `pdb_files/`: Directory containing PDB files used to calculate distance matrices for protein structures.
  - `pickle_files/`: Directory where processed data chunks are stored as `.pkl` files.



## Quick Start
This guide will walk you through the steps to prepare the data, train the TENet model, and apply the trained model within an Evolutionary Algorithm (EA) framework to optimize transcriptional repressors.

### 1. Preparing the Data

#### (a) Generate PDB Structures for Protein Sequences

If you are starting from scratch, the first step is to generate PDB structures using the **esmFold_bulk_structure_pred** scripts.

1. Navigate to the `esmFold_bulk_structure_pred/` directory.
2. Submit the job to generate PDB files in bulk:

```bash
cd esmFold_bulk_structure_pred
sbatch submit_esmFoldBulk.sh
```
This will predict structures for protein sequences in bulk using ESMFold, saving the results as PDB files in the pdb_files/ directory.

#### (b) Generate Inputs for the Neural Network
Once the PDB structures are generated, you can run create_data.py to extract features (such as ESM-2 embeddings, contact maps, and amino acid descriptors) and save them as pickle files for further model training.

```bash
python create_data.py
```
The processed data will be saved as chunks in the DATA/pickle_files/ directory.

Note: If you prefer to skip this step, you can download the pre-processed pickle files directly from [here](https://drive.google.com/drive/folders/1BzVvDauaOOaHIXXIMZNWkpg6OzvntLq1?usp=sharing) and place them in the `DATA/pickle_files/` directory.

### 2. Training the Model
Once the data is prepared, you can train the TENet model. The architecture of TENet is implemented in `model.py`. You can either run this script directly or submit it to a SLURM-based cluster using the provided submit.sh script (`sbatch submit.sh`).

### 3. Using a Trained Model in an Evolutionary Algorithm (EA)
Once the model is trained, you can use it within the Evolutionary Algorithm (EA) framework to design optimized transcriptional repressors. This EA is implemented in the EA/MAIN.py script.

1. Ensure the trained model is saved in the appropriate path (e.g., `model_no_attention.pt` in the `EA/` directory).
2. Submit the EA optimization job via SLURM:
    ```bash
    cd EA
    sbatch submit.sh
    ```
This process will run the evolutionary optimization, generating mutant sequences, predicting their structures, and evaluating their fitness using the trained model. The results, including PDB structures of mutant sequences and corresponding fitness scores, will be saved in the pdb_files/ and OUTPUTS/ directories.

## Pre-trained Models

We release 1,290 trained TENet checkpoints covering every leave-one-domain-out fold of our training set, evaluated under four data-augmentation regimes. They can be downloaded as a single folder from Google Drive:

**[TENet trained models (Google Drive)](https://drive.google.com/drive/folders/15TxVUp4dfInWMfeV1oRS03sp-SuzYiA1?usp=sharing)**

### How the data was split

The training set contains 54 domains. For each domain we performed leave-one-domain-out evaluation: a model was trained on the data from the remaining 53 domains and evaluated on the held-out 54th. Training was repeated 6 times per held-out domain (`replicate_0` through `replicate_5`) to give an estimate of replicate-to-replicate variability, yielding 324 models per setting (54 × 6).

The four settings differ only in *what supervision from the held-out domain was added to training* — the train/test fold definitions are otherwise identical across settings:

| # | Setting | Drive subfolder | What is added from the held-out domain |
|---|---|---|---|
| 1 | Zero-shot | `1_Zero_shot/` | Nothing — no reads from the held-out domain are added to training. |
| 2 | One-shot (WT) | `2_One_shot_WT/` | The wild-type sequence of the held-out domain (identified using the ;WT; identifier in the label). |
| 3 | Few-shot (windowed-shuffle, size 4) | `3_Few_shot_Windowed_4/` | Windowed-shuffle reads of size 4 from the held-out domain (identified using the WIND_4 identifier in the label). |
| 4 | Few-shot (windowed-shuffle, sizes 4 and 8) | `4_Few_shot_Windowed_4_8/` | Windowed-shuffle reads of size 4 and size 8 from the held-out domain (identified using the WIND_4 and WIND_8 identifier in the label). |

Setting 4 covers 53 of the 54 folds (the `Short_nuclear_domain;NGN2_HUMAN;HLH;99;52` fold is excluded because one of its six replicates is missing), giving **318 models** for that setting and **1,290 models in total** (324 + 324 + 324 + 318).

### Folder layout

```
TENet_Trained_Models/
├── 1_Zero_shot/
│   └── <held-out-domain-name>/
│       ├── replicate_0/model.pt
│       ├── replicate_1/model.pt
│       ├── ...
│       └── replicate_5/model.pt
├── 2_One_shot_WT/
├── 3_Few_shot_Windowed_4/
└── 4_Few_shot_Windowed_4_8/
```

The held-out-domain folder name encodes the full domain identifier, e.g. `Short_nuclear_domain;ASCL1_HUMAN;HLH;105;52`. When you pick a checkpoint, you are loading a model that never saw any reads from this domain during training (or saw only the augmentation reads specified by the chosen setting).

### Loading a checkpoint and running a forward pass

Each `model.pt` is a PyTorch `state_dict` for the `Model` class defined in [`model.py`](./model.py). To load a checkpoint and run a forward pass:

```python
import torch
from model import Model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Instantiate the same architecture used at training time
model = Model(dropout_rate=0.5).to(device)

# Load one of the pre-trained checkpoints
checkpoint_path = ('TENet_Trained_Models/1_Zero_shot/'
                   'Short_nuclear_domain;ASCL1_HUMAN;HLH;105;52/'
                   'replicate_0/model.pt')
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()

# Build inputs for one or more sequences using create_data.py
# (token_representations, dist_matrix, aa_descriptors_sequence).
# Expected input shapes — B = batch size, L = sequence length:
#   token_representations    : (B, L, 1280)  ESM-2 per-residue embeddings
#   dist_matrix              : (B, L, L)     residue-residue distance matrix
#   aa_descriptors_sequence  : (B, L, 66)    per-residue AA descriptors

with torch.no_grad():
    p_off = model(token_representations.to(device),
                  dist_matrix.to(device),
                  aa_descriptors_sequence.to(device))

# p_off has shape (B, 1) — the predicted P(OFF) for each input sequence.
```

To produce the three input tensors for arbitrary new sequences, see [`create_data.py`](./create_data.py), which builds them from an amino-acid sequence plus an ESMFold-predicted PDB structure.

## Questions, problems?
Make a github issue 😄. Please be as clear and descriptive as possible. Please feel free to reach
out in person: (akshat98[AT]stanford[DOT]edu)