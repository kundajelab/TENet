"""
ESMFold Bulk Prediction Script

This script is designed to process large-scale protein sequence data in chunks, using ESMFold to predict protein structures
and output the results in PDB format. The script can be used in a high-performance computing (HPC) environment with job 
array support, allowing for parallel processing across multiple jobs.

**Workflow:**
1. The script reads protein sequences from a csv file (`NEW_PARSE_4.csv`), focusing on a chunk of data defined by the job array index.
2. It tokenizes the protein sequences using ESMFold’s tokenizer.
3. The ESMFold model processes the sequences to predict 3D protein structures.
4. The predicted structures are converted into PDB format.
5. The resulting PDB files are saved in a specified output directory (`pdb_files`).

**Key features:**
- Uses ESMFold model from Hugging Face’s `transformers` library.
- GPU support for faster processing.
- Handles chunking of large datasets via job arrays in a SLURM HPC environment.
- Error handling for directory creation and PDB file I/O.

**Input:**
- Protein sequence data from `NEW_PARSE_4.csv`.
- `protein_seq` column contains the sequences to be folded.

**Output:**
- PDB files for each protein sequence, saved in the `pdb_files` directory.

**Usage:**
This script is designed to be used in an HPC environment with SLURM job array, using the script submit_esmFoldBulk.sh

Each array job processes a different chunk of the dataset. Make sure to adjust the number of jobs according to the size of the dataset.

@author: aksha
"""
import os
import torch
import argparse
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, EsmForProteinFolding
from transformers.models.esm.openfold_utils.protein import to_pdb, Protein as OFProtein
from transformers.models.esm.openfold_utils.feats import atom14_to_atom37

# Helper function to convert ESMFold outputs to PDB format
def convert_outputs_to_pdb(outputs):
    """Convert ESMFold outputs to PDB format."""
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

def main(chunk1, chunk2):
    """
    Main function to process a chunk of protein sequences and save PDB files.

    Args:
    chunk1 (int): Start index for protein sequences.
    chunk2 (int): End index for protein sequences.
    """
    # Load data: Ensure the input file is tab-separated
    df = pd.read_csv('NEW_PARSE_4.csv', sep=',')
    df_chunk = df.iloc[chunk1:chunk2]  # Process a specific chunk of the dataset

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
    model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1", low_cpu_mem_usage=True)

    # Move the model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Tokenize protein sequences
    protein_seqs = df_chunk['AA_without_adapter'].tolist()
    tokenized_ = tokenizer(protein_seqs, padding=False, add_special_tokens=False)['input_ids']

    outputs = []
    with torch.no_grad():
        for input_ids in tqdm(tokenized_, desc="Processing sequences"):
            input_ids = torch.tensor(input_ids, device=device).unsqueeze(0)
            output = model(input_ids)
            outputs.append({key: val.cpu() for key, val in output.items()})

    pdb_list = [convert_outputs_to_pdb(output) for output in outputs]

    # Create the directory to save the PDB files if it doesn't exist
    save_dir = "pdb_files"
    try:
        os.makedirs(save_dir, exist_ok=True)
    except OSError as e:
        print(f"Error creating directory {save_dir}: {e}")
        return

    # Save PDB files
    protein_identifiers = df_chunk.index.tolist()
    for identifier, pdb in zip(protein_identifiers, pdb_list):
        try:
            with open(f"{save_dir}/{identifier}.pdb", "w") as f:
                f.write("".join(pdb))
        except IOError as e:
            print(f"Error saving PDB file for protein {identifier}: {e}")

if __name__ == "__main__":
    # Parse arguments for job array indices
    parser = argparse.ArgumentParser(description="ESMFold Bulk Prediction Script")
    parser.add_argument("job_id", help="Array job ID (SLURM_ARRAY_TASK_ID)")
    parser.add_argument("--num_jobs", type=int, default=1000, help="Total number of jobs")
    args = parser.parse_args()

    job_idx = int(args.job_id)  # Convert job_id to an integer

    # Read number of sequences and define chunk size
    num_sequences = 116907 + 1000 # Number of lines in csv + offset
    chunk_size = num_sequences // args.num_jobs

    # Define start and end indices for the current chunk
    start_idx = (job_idx - 1) * chunk_size
    end_idx = min(start_idx + chunk_size, num_sequences)
    print(f'Starting job {job_idx}: Processing sequences {start_idx} to {end_idx}')

    # Run the main function
    main(start_idx, end_idx)
