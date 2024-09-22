"""
This script processes amino acid sequences, extracts their features using ESM-2 embeddings, and computes contact maps and 
amino acid descriptors. It outputs the following files as a result:

1. Pickle files containing chunks of the processed data. These are stored in the 'DATA/pickle_files/' directory.
2. Each pickle file includes:
   - Token representations from ESM-2.
   - Attention maps.
   - Distance matrices calculated from the PDB files.
   - Amino acid descriptors for each sequence.

Make sure to have the following:
- The amino acid descriptors in './DATA/aa_descriptors.csv'.
- The PDB files in './DATA/pdb_files/'.
- Input CSV file './DATA/NEW_PARSE_4.csv'.

If the PDB files directory does not exist, an exception will be raised.
"""

import torch
import esm
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
from Bio.PDB import *
import numpy as np
import pickle
import warnings
import pandas as pd
import os

warnings.filterwarnings('ignore')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
start_time = time.time()

# Ensure the 'DATA/pickle_files' directory exists
if not os.path.exists('DATA/pickle_files'):
    os.makedirs('DATA/pickle_files')

# Ensure the 'DATA/pdb_files' directory exists, or raise an exception
if not os.path.exists('DATA/pdb_files'):
    raise FileNotFoundError("The 'DATA/pdb_files' directory does not exist. Please provide the necessary PDB files.")

# Load the amino acid descriptors
aa_descriptors = pd.read_csv('./DATA/aa_descriptors.csv')
aa_descriptors.set_index('AA', inplace=True)

# Define a function to calculate the distance between two residues
def calc_residue_dist(residue_one, residue_two):
    """Returns the C-alpha distance between two residues"""
    diff_vector = residue_one["CA"].coord - residue_two["CA"].coord
    return np.sqrt(np.sum(diff_vector * diff_vector))

# Define a function to calculate the distance matrix between two chains
def calc_dist_matrix(chain_one, chain_two):
    """Returns a matrix of C-alpha distances between two chains"""
    answer = np.zeros((len(chain_one), len(chain_two)), float)
    for row, residue_one in enumerate(chain_one):
        for col, residue_two in enumerate(chain_two):
            answer[row, col] = calc_residue_dist(residue_one, residue_two)
    return answer

# Define a function to save data in chunks as pickle files
def save_data_chunks(data, chunk_size=1000, directory='DATA/pickle_files'):
    """
    Saves the data in chunks to the specified directory.
    
    Args:
    data (list): List of data to save.
    chunk_size (int): Number of data points per chunk.
    directory (str): Directory where the data chunks will be saved.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

    for i in range(0, len(data), chunk_size):
        print('Saving chunk: {}/{}'.format(i, i+chunk_size))
        chunk = data[i:i+chunk_size]
        with open(os.path.join(directory, f'data_{i//chunk_size}.pkl'), 'wb') as f:
            pickle.dump(chunk, f)

# Function to get amino acid descriptors for a given sequence
def get_amino_acid_descriptors(aa_sequence):
    """Gets the descriptors for each amino acid in the sequence"""
    aa_sequence = aa_sequence.replace("\"", "")  # Remove quotation marks
    descriptors = []
    for aa in aa_sequence:
        if aa in aa_descriptors.index:
            descriptors.append(aa_descriptors.loc[aa].values)
        else:
            raise Exception('AA not found!')
    return np.array(descriptors)

# Read the input CSV file
with open('./DATA/NEW_PARSE_4.csv', 'r') as f:
    lines = f.readlines()

header = lines[0]
lines = lines[1:]

# Load the ESM-2 model
model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
model = model.to(device)  # Move the model to GPU if available
batch_converter = alphabet.get_batch_converter()
model.eval()  # Disable dropout for deterministic results

data_all = []

# Wrap the main loop with tqdm for progress tracking
for idx_, item in enumerate(lines):

    if idx_ % 10000 == 0 and idx_ > 0:
        print('On: {}/{} Time: {}'.format(idx_, len(lines), time.time()-start_time))
        start_time = time.time()

    # Extract all information about the sequence from the CSV
    A = item.split(',')
    try:
        label = A[0]
        R1, countsOFF_R1, countsON_R1, R2, countsOFF_R2, countsON_R2, Avg = (
            float(A[1]),
            float(A[2]),
            float(A[3]),
            float(A[4]),
            float(A[5]),
            float(A[6]),
            float(A[7]),
        )
        aa_seq = A[9]
    except:
        continue  # Skip lines with missing reads or other issues

    # Collect amino acid descriptor information
    aa_descriptors_sequence = get_amino_acid_descriptors(aa_seq)
    aa_descriptors_sequence = torch.tensor(aa_descriptors_sequence)

    # ESM embedding for the sequence
    data = [("protein1", aa_seq)]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)  # +2 added for each string
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

    with torch.no_grad():
        batch_tokens = batch_tokens.to(device)
        results = model(batch_tokens, repr_layers=[33], return_contacts=True)

    token_representations = results["representations"][33]  # Shape: [1, seq_len, 1280]
    token_representations = token_representations.squeeze(0).detach().cpu()

    # Attention map for the sequence
    attention_contacts = results["contacts"].squeeze(0).detach().cpu()

    # Add in the contact map from the PDB file
    pdb_file = './DATA/pdb_files/{}.pdb'.format(idx_)
    if not os.path.exists(pdb_file):
        print(f"File {pdb_file} does not exist. Skipping...")
        continue

    structure = PDBParser().get_structure(pdb_file, pdb_file)
    pdb_model = structure[0]
    dist_matrix = calc_dist_matrix(pdb_model["A"], pdb_model["A"])
    dist_matrix = dist_matrix / np.max(dist_matrix)
    dist_matrix = torch.tensor(dist_matrix)

    # Append data to the list
    data_all.append((token_representations, attention_contacts, dist_matrix, aa_descriptors_sequence, aa_seq, label, R1, countsOFF_R1, countsON_R1, R2, countsOFF_R2, countsON_R2, Avg))

    # Free up GPU memory
    torch.cuda.empty_cache()

# Save the data in chunks
save_data_chunks(data_all)
