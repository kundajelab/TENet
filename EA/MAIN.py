"""
The goal is to generate mutant sequences from a wild-type protein, predict their 3D structures, and evaluate 
them using a traned TENet model. In this example, we will be creating mutants starting with WT HES3, and maximizing the repression stength. 

**Key functionalities:**
1. Loads a pre-trained ESMFold model for protein structure prediction.
2. Use a trained TENet to evaluate mutant sequences based on ESM embeddings, contact maps, and amino acid descriptors.
3. Generates new mutants iteratively, evaluating and selecting the best ones based on the model output.
4. Saves the predicted 3D structures as PDB files and outputs key metrics to CSV files.

**Input:**
- A wild-type protein sequence.
- Pre-trained NN model (`model_no_attention.pt`) for evaluating mutants.

**Output:**
- Generated mutants in PDB format saved in the `pdb_files/` directory.
- Mutant evaluation metrics stored in CSV files under the `OUTPUTS/` directory.

@author: aksha
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers.utils import send_example_telemetry
from transformers import AutoTokenizer, EsmForProteinFolding
from transformers.models.esm.openfold_utils.protein import to_pdb, Protein as OFProtein
from transformers.models.esm.openfold_utils.feats import atom14_to_atom37
from io import BytesIO
import pandas
import csv
import time 
from Bio.PDB import *
import numpy as np
import pickle
import warnings
import pandas as pd
import esm 
import random

warnings.filterwarnings('ignore')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
start_time = time.time()


# Load the amino acid descriptors
aa_descriptors = pd.read_csv('./DATA/aa_descriptors.csv')
aa_descriptors.set_index('AA', inplace=True)

# Load ESM-2 model
model_ESM, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
# model = model.to(device)  # Move the model to GPU if available
batch_converter = alphabet.get_batch_converter()
model_ESM.eval()  # disables dropout for deterministic results

# Preparing model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
model_fold = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1", low_cpu_mem_usage=True)
model_fold = model_fold.cuda() # Uncomment for GPU

# Define a function to calculate the distance
def calc_residue_dist(residue_one, residue_two):
    """Returns the C-alpha distance between two residues"""
    diff_vector  = residue_one["CA"].coord - residue_two["CA"].coord
    return np.sqrt(np.sum(diff_vector * diff_vector))

def calc_dist_matrix(chain_one, chain_two):
    """Returns a matrix of C-alpha distances between two chains"""
    answer = np.zeros((len(chain_one), len(chain_two)), float)
    for row, residue_one in enumerate(chain_one):
        for col, residue_two in enumerate(chain_two) :
            answer[row, col] = calc_residue_dist(residue_one, residue_two)
    return answer

def save_data_chunks(data, chunk_size=1000, directory='DATA/pickle_files'):
    if not os.path.exists(directory):
        os.makedirs(directory)

    for i in range(0, len(data), chunk_size):
        print('On: {}/{}', i, i+chunk_size)
        chunk = data[i:i+chunk_size]
        with open(os.path.join(directory, f'data_{i//chunk_size}.pkl'), 'wb') as f:
            pickle.dump(chunk, f)
            
def get_amino_acid_descriptors(aa_sequence):
    """Gets the descriptors for each amino acid in the sequence"""
    aa_sequence = aa_sequence.replace("\"", "")  # remove quotation marks
    descriptors = []
    for aa in aa_sequence:
        if aa in aa_descriptors.index:
            descriptors.append(aa_descriptors.loc[aa].values)
        else:
            raise Exception('AA not found!')
    return np.array(descriptors)


def convert_outputs_to_pdb(outputs):
    final_atom_positions = atom14_to_atom37(outputs["positions"][-1], outputs)
    outputs = {k: v.to("cpu").numpy() for k, v in outputs.items()}
    final_atom_positions = final_atom_positions.cpu().numpy()
    final_atom_mask = outputs["atom37_atom_exists"]
    pdbs = []
    for i in range(outputs["aatype"].shape[0]):
        aa = outputs["aatype"][i]
        pred_pos = final_atom_positions[i]
        mask = final_atom_mask[i]
        resid = outputs["residue_index"][i] + 1
        pred = OFProtein(
            aatype=aa,
            atom_positions=pred_pos,
            atom_mask=mask,
            residue_index=resid,
            b_factors=outputs["plddt"][i],
            chain_index=outputs["chain_index"][i] if "chain_index" in outputs else None,
        )
        pdbs.append(to_pdb(pred))
    return pdbs

def pred_structure(seq, filename, gen): 

    # Sequence to be processed
    AA_without_adapter = [seq] 
    
    # Tokenize sequence
    ecoli_tokenized = tokenizer(AA_without_adapter, padding=False, add_special_tokens=False)['input_ids']
    
    outputs = []
    
    with torch.no_grad():
        for input_ids in tqdm(ecoli_tokenized):
            input_ids = torch.tensor(input_ids, dtype=torch.long, device='cuda').unsqueeze(0) # Explicitly convert to long and move to GPU
            output = model_fold(input_ids)
            outputs.append({key: val.cpu() for key, val in output.items()})
    
    # Convert outputs to pdb (assuming convert_outputs_to_pdb is defined)
    pdb_list = [convert_outputs_to_pdb(output) for output in outputs]
    
    # Define the directory to save the files
    save_dir = f"pdb_files/{gen}"
    
    # Create the directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Save all pdb files in the directory
    protein_identifiers = [filename]
    for identifier, pdb in zip(protein_identifiers, pdb_list):
        with open(f"{save_dir}/{identifier}.pdb", "w") as f:
            f.write("".join(pdb))
            
    
# Function to generate a mutant sequence
def generate_mutant(seq, max_num_mutations):
    # Amino acid alphabet
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    
    # Choose the number of mutations for this particular mutant
    num_mutations = random.randint(1, max_num_mutations)
    
    # Indices to mutate
    indices_to_mutate = random.sample(range(len(seq)), num_mutations)
    
    # Generate mutant sequence
    mutant_seq = list(seq)
    for i in indices_to_mutate:
        # Select a new amino acid that is different from the original
        new_aa = random.choice([aa for aa in amino_acids if aa != seq[i]])
        
        # Apply the mutation
        mutant_seq[i] = new_aa
    
    mut_seq = ''.join(mutant_seq)  
    
    return mut_seq, num_mutations, indices_to_mutate

class Model(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super(Model, self).__init__()

        # Sequence encoder
        self.gru = nn.GRU(input_size=1280, hidden_size=256, num_layers=2, bidirectional=True, batch_first=True)
        self.bn_gru = nn.BatchNorm1d(512) # Batch Normalization for GRU
        self.dropout_gru = nn.Dropout(dropout_rate) # Dropout for GRU
        
        # AA Descriptor encoder
        self.gru_enc = nn.GRU(input_size=66, hidden_size=256, num_layers=2, bidirectional=True, batch_first=True)
        self.bn_gru_enc = nn.BatchNorm1d(512) # Batch Normalization for GRU
        self.dropout_gru_enc = nn.Dropout(dropout_rate) # Dropout for GRU

        # Contact map encoder
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32) # Batch Normalization for conv1
        self.dropout1 = nn.Dropout2d(dropout_rate) # Dropout for conv1
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32) # Batch Normalization for conv2
        self.dropout2 = nn.Dropout2d(dropout_rate) # Dropout for conv2


        # FC layer for prediction
        self.fc1 = nn.Linear(13824, 512)
        self.bn_fc1 = nn.BatchNorm1d(512) # Batch Normalization for fc1
        self.dropout_fc1 = nn.Dropout(dropout_rate) # Dropout for fc1
        self.fc2 = nn.Linear(512, 1)

    def forward(self, seq, contact_map, aa_descriptors_sequence):
        # Sequence
        _, h_n = self.gru(seq) # Assuming seq has shape (batch, seq_len, 1280)
        seq_feature = torch.cat((h_n[-2,:,:], h_n[-1,:,:]), dim=-1) # Concatenating the last hidden states of both directions
        seq_feature = self.dropout_gru(self.bn_gru(seq_feature)) # Applying Batch Normalization and Dropout
        
        # Add code here for using aa descriptors: 
        _, h_n_enc = self.gru_enc(aa_descriptors_sequence) # Assuming seq has shape (batch, seq_len, 66)
        aa_descrp = torch.cat((h_n_enc[-2,:,:], h_n_enc[-1,:,:]), dim=-1) # Concatenating the last hidden states of both directions
        aa_descrp = self.dropout_gru_enc(self.bn_gru_enc(aa_descrp)) # Applying Batch Normalization and Dropout

        # Contact map
        contact_map = contact_map.unsqueeze(1) 
        x1 = self.dropout1(F.relu(self.bn1(self.conv1(contact_map))))
        x1 = F.max_pool2d(x1, 2)
        x1 = self.dropout2(F.relu(self.bn2(self.conv2(x1))))
        x1 = F.max_pool2d(x1, 2)
        map_feature = torch.flatten(x1, start_dim=1)

        # Concatenate and predict
        x = torch.cat((seq_feature, map_feature, aa_descrp), dim=-1)
        x = self.dropout_fc1(F.relu(self.bn_fc1(self.fc1(x))))
        x = torch.sigmoid(self.fc2(x))
        
        return x

start_time = time.time()

# load the model: 
path_ = './model_hes3/Silencer_tiles;ENSG00000173673;10_1/model/model_no_attention.pt'
model = Model(dropout_rate=0.5).to(device) 
model.load_state_dict(torch.load(path_))
model = model.to(device)  # Move the model to the device
model.eval()  # Set the model to evaluation mode

seq = 'GSTMDSAGLGQEAPALFRPCTPAVWAPAPAAGGPRSPPPLLLLPESLPGSSASVPPPQPASSRCAESPGLGLRVWRPWGS' # WT HES3 domain
gen = 0
num_samples = 500 # Number of samples per iteration of optimization
max_num_mutations = 4
num_generations = 50
TRACK_ALL = {} 

# Create the OUTPUTS directory if it doesn't exist
if not os.path.exists('OUTPUTS'):
    os.makedirs('OUTPUTS')
    

# Generate mutants
for i in range(num_samples):
    
    while True: # Ensure an unseen mutant is found 
        mut_seq, num_mutations, indices_to_mutate = generate_mutant(seq, max_num_mutations)        
        if mut_seq not in TRACK_ALL:
            break  # Break the loop because we've found a unique mutant

    # Generate the PDB Structures: 
    filename = i
    pred_structure(mut_seq, filename, gen)
    save_dir = f"pdb_files/{gen}"
    file_path = f"{save_dir}/{i}.pdb"
    
    # Generate features for input to the model: 
    aa_descriptors_sequence = get_amino_acid_descriptors(mut_seq)
    aa_descriptors_sequence = torch.tensor(aa_descriptors_sequence)

    # ESM embedding for the sequence: 
    data = [ ("protein1", mut_seq) ]
    batch_labels, batch_strs, batch_tokens = batch_converter(data) # +2 is added for each string
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
    with torch.no_grad():
        # batch_tokens = batch_tokens.to(device)
        results = model_ESM(batch_tokens, repr_layers=[33], return_contacts=True)
    token_representations = results["representations"][33] 
    token_representations = token_representations.squeeze(0)
    token_representations = token_representations.detach().cpu()

    # PDB contact map: 
    structure = PDBParser().get_structure(file_path, file_path)
    pdb_model = structure[0]
    dist_matrix = calc_dist_matrix(pdb_model["A"], pdb_model["A"])
    dist_matrix = dist_matrix / np.max(dist_matrix)
    dist_matrix = torch.tensor(dist_matrix)
    
    # Inputs to NN: token_representations (ESM), dist_matrix (contact map), aa_descriptors_sequence (CPT)
    token_representations = token_representations.reshape((1, 82, 1280)).float()
    dist_matrix = dist_matrix.reshape((1, 80, 80)).float()
    aa_descriptors_sequence = aa_descriptors_sequence.reshape((1, 80, 66)).float()
    token_representations = token_representations.to(device)
    aa_descriptors_sequence = aa_descriptors_sequence.to(device)
    dist_matrix = dist_matrix.to(device)

    # Forward pass
    output = model(token_representations, dist_matrix, aa_descriptors_sequence)
    output = output.detach().float()
    output = output.item()
    
    TRACK_ALL[mut_seq] = [output, seq, file_path, num_mutations, ';'.join(str(x) for x in indices_to_mutate), gen]



# Saving TRACK_ALL to a CSV
with open(f'./OUTPUTS/track_all_gen_{gen}.csv', 'a+', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['mut_seq', 'output', 'parent_seq', 'file_path', 'num_mutations', 'indices_to_mutate', 'gen'])
    for key, value in TRACK_ALL.items():
        writer.writerow([key] + value)


print('Iteration 0 Total time: ', time.time() - start_time)

gen = gen + 1

while gen < num_generations: 
    
    TRACK_ALL_GEN = {}
    
    # Sort sequence population from best to worst
    track_all_items = [(mut_seq, props) for mut_seq, props in TRACK_ALL.items()]
    sorted_track_all = sorted(track_all_items, key=lambda x: x[1][0], reverse=True)
    sorted_properties = [props for mut_seq, props in sorted_track_all]
    sorted_mut_seqs   = [mut_seq for mut_seq, props in sorted_track_all]
    
    # Print or return the sorted lists
    print("Sorted Properties:", sorted_properties[0:10])
    print("Sorted Mutant Sequences:", sorted_mut_seqs[0:10])
        
    # Take the top 10 sequences and their properties
    top_10_properties = sorted_properties[:10]
    top_10_mut_seqs = sorted_mut_seqs[:10]
    
    # Create a list of probabilities proportional to the output value for the top 10
    probabilities = [props[0] for props in top_10_properties]
    total_output = sum(probabilities)
    probabilities = [p / total_output for p in probabilities]

    
    for i in range(num_samples):

        
        # Weighted random selection from the top 10 based on the output value
        selected_index = random.choices(range(len(top_10_mut_seqs)), probabilities, k=1)[0]
        
        # Get the selected sequence and its properties
        selected_mut_seq = top_10_mut_seqs[selected_index]
        selected_properties = top_10_properties[selected_index]
        
        while True: # Ensure an unseen mutant is found 
            mut_seq, num_mutations, indices_to_mutate = generate_mutant(selected_mut_seq, max_num_mutations)        
            if mut_seq not in TRACK_ALL:
                break  # Break the loop because we've found a unique mutant

        # Generate the PDB Structures: 
        filename = i
        pred_structure(mut_seq, filename, gen)
        save_dir = f"pdb_files/{gen}"
        file_path = f"{save_dir}/{i}.pdb"
        
        # Generate features for input to the model: 
        aa_descriptors_sequence = get_amino_acid_descriptors(mut_seq)
        aa_descriptors_sequence = torch.tensor(aa_descriptors_sequence)

        # ESM embedding for the sequence: 
        data = [ ("protein1", mut_seq) ]
        batch_labels, batch_strs, batch_tokens = batch_converter(data) # +2 is added for each string
        batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
        with torch.no_grad():
            # batch_tokens = batch_tokens.to(device)
            results = model_ESM(batch_tokens, repr_layers=[33], return_contacts=True)
        token_representations = results["representations"][33] 
        token_representations = token_representations.squeeze(0)
        token_representations = token_representations.detach().cpu()

        # PDB contact map: 
        structure = PDBParser().get_structure(file_path, file_path)
        pdb_model = structure[0]
        dist_matrix = calc_dist_matrix(pdb_model["A"], pdb_model["A"])
        dist_matrix = dist_matrix / np.max(dist_matrix)
        dist_matrix = torch.tensor(dist_matrix)
        
        # Inputs to NN: token_representations (ESM), dist_matrix (contact map), aa_descriptors_sequence (CPT)
        token_representations = token_representations.reshape((1, 82, 1280)).float()
        dist_matrix = dist_matrix.reshape((1, 80, 80)).float()
        aa_descriptors_sequence = aa_descriptors_sequence.reshape((1, 80, 66)).float()
        token_representations = token_representations.to(device)
        aa_descriptors_sequence = aa_descriptors_sequence.to(device)
        dist_matrix = dist_matrix.to(device)

        # Forward pass
        output = model(token_representations, dist_matrix, aa_descriptors_sequence)
        output = output.detach().float()
        output = output.item()
        
        TRACK_ALL[mut_seq]     = [output, selected_mut_seq, file_path, num_mutations, ';'.join(str(x) for x in indices_to_mutate), gen]
        TRACK_ALL_GEN[mut_seq] = [output, selected_mut_seq, file_path, num_mutations, ';'.join(str(x) for x in indices_to_mutate), gen]
        
    # Write generation results onto a CSV
    with open(f'./OUTPUTS/track_all_gen_{gen}.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['mut_seq', 'output', 'parent_seq', 'file_path', 'num_mutations', 'indices_to_mutate', 'gen'])
        for key, value in TRACK_ALL_GEN.items():
            writer.writerow([key] + value)
    
        
    gen = gen + 1 # Increase the iteratrion count