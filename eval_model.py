# -*- coding: utf-8 -*-
"""
This script evaluates a trained neural network model (TENet) for predicting transcriptional repressor functions for a test set domain (in this case ASCL1).
Upon successful execution, the following files will be created:

1. './PLOTS/OUTPUT.csv': Contains the predicted and actual P(OFF), log ratios, and other evaluation metrics for each sample.
2. './PLOTS/predicted_vs_actual.png': A scatter plot comparing the predicted and actual P(OFF) values.
3. './PLOTS/logit_predicted_vs_actual.png': A scatter plot comparing the predicted and actual P(OFF) values in logit space.
4. './PLOTS/predicted_vs_actual_distrb.png': A scatter plot with marginal histograms comparing predicted and actual P(OFF) values.
5. './PLOTS/log_ratio_predicted_vs_actual.png': A scatter plot comparing predicted and actual log ratios of OFF/ON values.
6. './PLOTS/RAND_predicted_vs_actual.png': A scatter plot comparing predicted and actual P(OFF) values for random ASCL1 controls.
7. './PLOTS/SINGLE_MUTN_predicted_vs_actual.png': A scatter plot comparing predicted and actual P(OFF) values for single mutant ASCL1 samples.
8. './PLOTS/WIND_MUTN_predicted_vs_actual.png': A scatter plot comparing predicted and actual P(OFF) values for windowed mutant ASCL1 samples.
9. './PLOTS/heatmap.png': Heatmaps visualizing actual and predicted mutational effects on the ASCL1 domain.

@author: aksha
"""
import os
import pickle
import glob
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import seaborn as sns 
import pandas as pd

# Function to load data from pickle files
def load_data(directory='/scratch/aks998/DATA/pickle_files'):
    """
    Load data from all pickle files in the specified directory.

    Args:
    directory (str): Path to the directory containing the .pkl files.

    Returns:
    list: A combined list of data from all .pkl files.
    """
    file_list = glob.glob(os.path.join(directory, '*.pkl'))
    data = []

    for file in file_list:
        with open(file, 'rb') as f:
            data.extend(pickle.load(f))

    return data

# Function to process and split the data into validation, test, and training sets
def process_data():
    """
    Process data by splitting it into validation, test, and training sets based on specific labels.

    The validation set consists of RYBP, CBX5, and PCGF2 domains.
    The test set consists of all ASCL1 mutants.

    Returns:
    tuple: validation, test, and training data lists.
    """
    combined_data = load_data()

    # Labels for validation and test sets
    validation_labels = ['Silencer_tiles;ENSG00000163602;11', 'Silencer_tiles;ENSG00000094916;11', 'Silencer_tiles;ENSG00000277258;0']  # RYBP, CBX5, PCGF2
    test_labels = ['Short_nuclear_domain;ASCL1_HUMAN;HLH;105;52']
    combined_labels = validation_labels + test_labels

    # Initialize lists
    validation_ls, test_ls, train_ls = [], [], []

    # Split data based on labels
    for data_point in combined_data:
        label = data_point[5]

        if any(sublabel in label for sublabel in validation_labels):
            validation_ls.append(data_point)
        elif any(sublabel in label for sublabel in test_labels):
            test_ls.append(data_point)
        elif not any(sublabel in label for sublabel in combined_labels):
            train_ls.append(data_point)

    return validation_ls, test_ls, train_ls

# Define the neural network model class
class Model(nn.Module):
    """
    Neural network model for sequence, contact map, and AA descriptor encoding using GRU and CNN layers.
    """
    def __init__(self, dropout_rate=0.5):
        super(Model, self).__init__()

        # Sequence encoder using GRU
        self.gru = nn.GRU(input_size=1280, hidden_size=256, num_layers=2, bidirectional=True, batch_first=True)
        self.bn_gru = nn.BatchNorm1d(512)  # Batch Normalization for GRU
        self.dropout_gru = nn.Dropout(dropout_rate)  # Dropout for GRU
        
        # AA Descriptor encoder using GRU
        self.gru_enc = nn.GRU(input_size=66, hidden_size=256, num_layers=2, bidirectional=True, batch_first=True)
        self.bn_gru_enc = nn.BatchNorm1d(512)  # Batch Normalization for GRU
        self.dropout_gru_enc = nn.Dropout(dropout_rate)  # Dropout for GRU

        # Contact map encoder using CNN
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)  # Batch Normalization for conv1
        self.dropout1 = nn.Dropout2d(dropout_rate)  # Dropout for conv1
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)  # Batch Normalization for conv2
        self.dropout2 = nn.Dropout2d(dropout_rate)  # Dropout for conv2

        # Fully connected layer for prediction
        self.fc1 = nn.Linear(13824, 512)
        self.bn_fc1 = nn.BatchNorm1d(512)  # Batch Normalization for fc1
        self.dropout_fc1 = nn.Dropout(dropout_rate)  # Dropout for fc1
        self.fc2 = nn.Linear(512, 1)

    def forward(self, seq, contact_map, aa_descriptors_sequence):
        """
        Forward pass through the model.

        Args:
        seq (torch.Tensor): Input sequence tensor.
        contact_map (torch.Tensor): Input contact map tensor.
        aa_descriptors_sequence (torch.Tensor): Input AA descriptors tensor.

        Returns:
        torch.Tensor: Output prediction of P(OFF).
        """
        # Encode sequence
        _, h_n = self.gru(seq)
        seq_feature = torch.cat((h_n[-2, :, :], h_n[-1, :, :]), dim=-1)
        seq_feature = self.dropout_gru(self.bn_gru(seq_feature))

        # Encode AA descriptors
        _, h_n_enc = self.gru_enc(aa_descriptors_sequence)
        aa_descrp = torch.cat((h_n_enc[-2, :, :], h_n_enc[-1, :, :]), dim=-1)
        aa_descrp = self.dropout_gru_enc(self.bn_gru_enc(aa_descrp))

        # Encode contact map
        contact_map = contact_map.unsqueeze(1)  # Add channel dimension for CNN
        x1 = self.dropout1(F.relu(self.bn1(self.conv1(contact_map))))
        x1 = F.max_pool2d(x1, 2)
        x1 = self.dropout2(F.relu(self.bn2(self.conv2(x1))))
        x1 = F.max_pool2d(x1, 2)
        map_feature = torch.flatten(x1, start_dim=1)

        # Concatenate all features and pass through fully connected layers
        x = torch.cat((seq_feature, map_feature, aa_descrp), dim=-1)
        x = self.dropout_fc1(F.relu(self.bn_fc1(self.fc1(x))))
        x = torch.sigmoid(self.fc2(x))  # Output as probability
        
        return x

# Custom Dataset class for loading the dataset
class MyDataset(Dataset):
    """
    Custom dataset class for loading the test data.

    Args:
    data_list (list): List containing the test data.
    """
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        """
        Retrieve a data point from the dataset.

        Args:
        idx (int): Index of the data point.

        Returns:
        tuple: Data elements including token representations, contact maps, descriptors, and other relevant information.
        """
        # Data Format: token_representations, attention_contacts, dist_matrix, aa_descriptors_sequence, aa_seq, label, R1, countsOFF_R1, countsON_R1, R2, countsOFF_R2, countsON_R2, Avg
        token_representations = torch.tensor(self.data_list[idx][0], dtype=torch.float)
        attention_contacts = torch.tensor(self.data_list[idx][1], dtype=torch.float)
        dist_matrix = torch.tensor(self.data_list[idx][2], dtype=torch.float)
        aa_descriptors_sequence = torch.tensor(self.data_list[idx][3], dtype=torch.float)

        # Compute total reads for each biorep
        total_reads_R1 = self.data_list[idx][7] + self.data_list[idx][8]
        total_reads_R2 = self.data_list[idx][10] + self.data_list[idx][11]
        
        # Weighted average calculation for countsOFF and countsON
        countsOFF = torch.tensor((self.data_list[idx][7] * total_reads_R1 + self.data_list[idx][10] * total_reads_R2) / (total_reads_R1 + total_reads_R2), dtype=torch.float)
        countsON = torch.tensor((self.data_list[idx][8] * total_reads_R1 + self.data_list[idx][11] * total_reads_R2) / (total_reads_R1 + total_reads_R2), dtype=torch.float)
        
        aa_seq = self.data_list[idx][4]
        label_str = self.data_list[idx][5]
        
        return token_representations, attention_contacts, dist_matrix, aa_descriptors_sequence, countsOFF, countsON, aa_seq, label_str

# Load and process the test data
start_time = time.time()
_, test_ls, _ = process_data()  # Load validation and test data
print('Total data load time: ', time.time() - start_time)

# Create the 'PLOTS' directory if it doesn't exist
if not os.path.exists('PLOTS'):
    os.makedirs('PLOTS')

test_data = MyDataset(test_ls)  # Prepare test dataset

test_loader = DataLoader(test_data, batch_size=len(test_ls), shuffle=True)

# Device configuration (use GPU if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_name = './model/model_no_attention.pt'

# Instantiate the model architecture and load the pre-trained weights
model = Model(dropout_rate=0.5).to(device)
model.load_state_dict(torch.load(model_name))
model.eval()  # Set the model to evaluation mode

predicted = []
actual = []

# Perform evaluation on the test data
with torch.no_grad():
    for i, (token_representations, attention_contacts, dist_matrix, aa_descriptors_sequence, countsOFF, countsON, aa_seq, label_str) in enumerate(test_loader):
        # Move tensors to device
        token_representations = token_representations.to(device)
        attention_contacts = attention_contacts.to(device)
        aa_descriptors_sequence = aa_descriptors_sequence.to(device)
        dist_matrix = dist_matrix.to(device)
        countsOFF = countsOFF.unsqueeze(-1).to(device)
        countsON = countsON.unsqueeze(-1).to(device)
        P_off_actual = (countsOFF / (countsOFF + countsON)).detach().cpu().numpy()

        # Forward pass
        outputs = model(token_representations, dist_matrix, aa_descriptors_sequence)
        outputs = outputs.detach().cpu().numpy()

        predicted.extend(outputs)
        actual.extend(P_off_actual)
        
        # Calculate predicted counts based on outputs
        total_counts = (countsOFF + countsON).detach().cpu().numpy()
        countsOFF_predicted = outputs * total_counts
        countsON_predicted = total_counts - countsOFF_predicted
        
        # Compute log2 ratios of counts
        log_ratio_actual = np.log2(countsOFF.detach().cpu().numpy() / countsON.detach().cpu().numpy())
        log_ratio_predicted = np.log2(countsOFF_predicted / countsON_predicted)

        # Generate CSV combining predictions with actual outputs
        if i == 0:
            with open('./PLOTS/OUTPUT.csv', 'w') as f:
                f.write('Label,AASeq,CountsOFF,CountsON,ProbOFF,CountsOFF_predicted,CountsON_predicted,ProbOFF_predicted,Log2RatioOFF_ON_actual,Log2RatioOFF_ON_predicted\n')

        with open('./PLOTS/OUTPUT.csv', 'a') as f:
            for label, seq, c_off, c_on, p_off, c_off_pred, c_on_pred, p_off_pred, log_ratio_act, log_ratio_pred in zip(
                    label_str, aa_seq, countsOFF.detach().cpu().numpy(), countsON.detach().cpu().numpy(),
                    P_off_actual, countsOFF_predicted, countsON_predicted, outputs, log_ratio_actual, log_ratio_predicted):
                f.write(f'{label},{seq},{c_off[0]},{c_on[0]},{p_off[0]},{c_off_pred[0]},{c_on_pred[0]},{p_off_pred[0]},{log_ratio_act[0]},{log_ratio_pred[0]}\n')

# Convert predicted and actual values to numpy arrays
predicted = np.array(predicted).flatten()
actual = np.array(actual).flatten()

# Calculate Spearman and Pearson correlation
spearman_corr, spearman_p = stats.spearmanr(predicted, actual)
pearson_corr, pearson_p = stats.pearsonr(predicted, actual)

# Plot: Predicted vs. Actual P(OFF)
plt.figure(figsize=(8, 8))
plt.scatter(actual, predicted, alpha=0.5)
plt.title('Correlation between Predicted and Actual P(OFF)')
plt.xlabel('Actual P(OFF)')
plt.ylabel('Predicted P(OFF)')
plt.text(0.1, 0.9, 'Spearman: {:.2f} (p={:.3e})\nPearson: {:.2f} (p={:.3e})'.format(
    spearman_corr, spearman_p, pearson_corr, pearson_p), 
         transform=plt.gca().transAxes, bbox=dict(facecolor='red', alpha=0.5))
plt.plot([np.min(actual), np.max(actual)], [np.min(predicted), np.max(predicted)], color='red')  # identity line
plt.grid()
plt.savefig("./PLOTS/predicted_vs_actual.png", dpi=500)
plt.show()

# Plots in logit space ####################################
logit_predicted = np.log(predicted) - np.log(1 - predicted)
logit_actual = np.log(actual) - np.log(1 - actual)

# Calculate Spearman and Pearson correlation for logits
spearman_corr, spearman_p = stats.spearmanr(logit_predicted, logit_actual)
pearson_corr, pearson_p = stats.pearsonr(logit_predicted, logit_actual)

plt.figure(figsize=(8, 8))
plt.scatter(logit_actual, logit_predicted, alpha=0.5)
plt.title('Correlation between Predicted and Actual in logit space')
plt.xlabel('Logit (Actual P(OFF))')
plt.ylabel('Logit (Predicted P(OFF))')
plt.text(0.1, 0.9, 'Spearman: {:.2f} (p={:.3e})\nPearson: {:.2f} (p={:.3e})'.format(
    spearman_corr, spearman_p, pearson_corr, pearson_p), 
         transform=plt.gca().transAxes, bbox=dict(facecolor='red', alpha=0.5))
plt.plot([np.min(logit_actual), np.max(logit_actual)], [np.min(logit_predicted), np.max(logit_predicted)], color='red')
plt.grid()
plt.savefig("./PLOTS/logit_predicted_vs_actual.png", dpi=500)
plt.show()

# Distrb Scatter plot #####################################
fig = plt.figure(figsize=(10, 10))
gs = fig.add_gridspec(2, 2, width_ratios=[7, 2], height_ratios=[2, 7])
ax0 = fig.add_subplot(gs[1, 0])  # The scatter plot
ax1 = fig.add_subplot(gs[0, 0])  # histogram on top
ax2 = fig.add_subplot(gs[1, 1])  # histogram to the right

# Scatter plot
ax0.scatter(actual, predicted, alpha=0.5)

# Marginal histograms
sns.histplot(x=actual, ax=ax1, color='blue', bins=50)
sns.histplot(y=predicted, ax=ax2, color='orange', bins=50)

# Hide the histogram spines
ax1.axis('off')
ax2.axis('off')

# Add a red identity line
ax0.plot([np.min(actual), np.max(actual)], [np.min(predicted), np.max(predicted)], 'r')

# Set the labels
ax0.set_xlabel('Actual P(OFF)')
ax0.set_ylabel('Predicted P(OFF)')

plt.grid()
plt.tight_layout()

# Save the figure before showing it
plt.savefig("./PLOTS/predicted_vs_actual_distrb.png", dpi=500)
plt.show()

# Scatter plot for the log predicted vs. actual ###########
data = pd.read_csv('./PLOTS/OUTPUT.csv')

# Extract actual and predicted log2 ratios
actual = data['Log2RatioOFF_ON_actual']
predicted = data['Log2RatioOFF_ON_predicted']

# Calculate Spearman and Pearson correlation for log2 ratios
spearman_corr, spearman_p = stats.spearmanr(predicted, actual)
pearson_corr, pearson_p = stats.pearsonr(predicted, actual)

# Plot
plt.rcParams.update({'font.size': 18})
plt.figure(figsize=(10, 10))
plt.scatter(actual, predicted, alpha=0.5)
plt.title('Correlation between Predicted and Actual Log Ratios')
plt.xlabel('Actual Log Ratio (OFF/ON)')
plt.ylabel('Predicted Log Ratio (OFF/ON)')
plt.text(0.1, 0.9, 'Spearman: {:.2f} (p={:.3e})\nPearson: {:.2f} (p={:.3e})'.format(
    spearman_corr, spearman_p, pearson_corr, pearson_p), 
         transform=plt.gca().transAxes, bbox=dict(facecolor='red', alpha=0.5))
plt.plot([np.min(actual), np.max(actual)], [np.min(predicted), np.max(predicted)], color='red')
plt.grid()

# Save the figure before showing it
plt.savefig("./PLOTS/log_ratio_predicted_vs_actual.png", dpi=500)

# Display the plot
plt.show()

# Scatter plot for random control points ##################
data = pd.read_csv('./PLOTS/OUTPUT.csv')

# Filter data for random controls
filtered_data = data[data['Label'].str.contains(';rand;')]

# Extract actual and predicted values
actual = filtered_data['ProbOFF']
predicted = filtered_data['ProbOFF_predicted']

# Calculate Spearman and Pearson correlation for random controls
spearman_corr, spearman_p = stats.spearmanr(predicted, actual)
pearson_corr, pearson_p = stats.pearsonr(predicted, actual)

# Plot
plt.rcParams.update({'font.size': 18})
plt.figure(figsize=(12, 12))
plt.scatter(actual, predicted, alpha=0.5)
plt.title('Correlation between Predicted and Actual P(OFF) for random ASCL1 controls')
plt.xlabel('Actual P(OFF)')
plt.ylabel('Predicted P(OFF)')
plt.text(0.1, 0.9, 'Spearman: {:.2f} (p={:.3e})\nPearson: {:.2f} (p={:.3e})'.format(
    spearman_corr, spearman_p, pearson_corr, pearson_p), 
         transform=plt.gca().transAxes, bbox=dict(facecolor='red', alpha=0.5))
plt.plot([np.min(actual), np.max(actual)], [np.min(predicted), np.max(predicted)], color='red')
plt.grid()

# Save the figure before showing it
plt.savefig("./PLOTS/RAND_predicted_vs_actual.png", dpi=500)

# Display the plot
plt.show()

# Scatter plot for single mutants #########################
filtered_data = data[data['Label'].str.contains(';mutate;')]

# Extract actual and predicted values
actual = filtered_data['ProbOFF']
predicted = filtered_data['ProbOFF_predicted']

# Calculate Spearman and Pearson correlation for single mutants
spearman_corr, spearman_p = stats.spearmanr(predicted, actual)
pearson_corr, pearson_p = stats.pearsonr(predicted, actual)

# Plot
plt.rcParams.update({'font.size': 18})
plt.figure(figsize=(12, 12))
plt.scatter(actual, predicted, alpha=0.5)
plt.title('Correlation between Predicted and Actual P(OFF) for single mutant ASCL1')
plt.xlabel('Actual P(OFF)')
plt.ylabel('Predicted P(OFF)')
plt.text(0.1, 0.9, 'Spearman: {:.2f} (p={:.3e})\nPearson: {:.2f} (p={:.3e})'.format(
    spearman_corr, spearman_p, pearson_corr, pearson_p), 
         transform=plt.gca().transAxes, bbox=dict(facecolor='red', alpha=0.5))
plt.plot([np.min(actual), np.max(actual)], [np.min(predicted), np.max(predicted)], color='red')
plt.grid()

# Save the figure before showing it
plt.savefig("./PLOTS/SINGLE_MUTN_predicted_vs_actual.png", dpi=500)

# Display the plot
plt.show()

# Scatter plot for windowed mutants #######################
filtered_data = data[data['Label'].str.contains(';WIND_')]

# Extract actual and predicted values
actual = filtered_data['ProbOFF']
predicted = filtered_data['ProbOFF_predicted']

# Calculate Spearman and Pearson correlation for windowed mutants
spearman_corr, spearman_p = stats.spearmanr(predicted, actual)
pearson_corr, pearson_p = stats.pearsonr(predicted, actual)

# Plot
plt.rcParams.update({'font.size': 18})
plt.figure(figsize=(12, 12))
plt.scatter(actual, predicted, alpha=0.5)
plt.title('Correlation between Predicted and Actual P(OFF) for windowed mutant ASCL1')
plt.xlabel('Actual P(OFF)')
plt.ylabel('Predicted P(OFF)')
plt.text(0.1, 0.9, 'Spearman: {:.2f} (p={:.3e})\nPearson: {:.2f} (p={:.3e})'.format(
    spearman_corr, spearman_p, pearson_corr, pearson_p), 
         transform=plt.gca().transAxes, bbox=dict(facecolor='red', alpha=0.5))
plt.plot([np.min(actual), np.max(actual)], [np.min(predicted), np.max(predicted)], color='red')
plt.grid()

# Save the figure before showing it
plt.savefig("./PLOTS/WIND_MUTN_predicted_vs_actual.png", dpi=500)

# Display the plot
plt.show()

# Single Mutn Heatmap ####################################
data = pd.read_csv('./PLOTS/OUTPUT.csv')

# Filter for single mutant data
filtered_data = data[data['Label'].str.contains(';mutate;')]

wt_domain_sequence = 'SGFGYSLPQQQPAAVARRNERERNRVKLVNLGFATLREHVPNGAANKKMSKVETLRSAVEYIRALQQLLDEHDAVSAAFQ'
aa_list = ['P', 'Q', 'N', 'C', 'T', 'S', 'E', 'D', 'H', 'R', 'K', 'W', 'Y', 'F', 'I', 'M', 'L', 'V', 'A', 'G']

matrix1 = np.empty((20, 80), dtype=np.float64)
matrix2 = np.empty((20, 80), dtype=np.float64)
matrix1[:] = np.nan
matrix2[:] = np.nan

all_values = []

# Fill matrices with actual and predicted values for heatmap
for idx, row in filtered_data.iterrows():
    label = row['Label']
    if ';mutate;' not in label:
        continue
    avg1 = row['ProbOFF']
    avg2 = row['ProbOFF_predicted']
    wt_aa = row['AASeq']
    indices_of_mutn = [i for i, (x, y) in enumerate(zip(wt_domain_sequence, wt_aa)) if x != y][0]
    aa_substituted = wt_aa[indices_of_mutn]
    
    all_values.append(avg1)
    all_values.append(avg2)
    
    if np.isnan(matrix1[aa_list.index(aa_substituted)][indices_of_mutn]):
        matrix1[aa_list.index(aa_substituted)][indices_of_mutn] = avg1
    if np.isnan(matrix2[aa_list.index(aa_substituted)][indices_of_mutn]):
        matrix2[aa_list.index(aa_substituted)][indices_of_mutn] = avg2

vmin = np.nanmin(all_values)
vmax = np.nanmax(all_values)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(24, 16))

# Heatmap 1: Actual Mutational Effects
sns.heatmap(matrix1, cmap="inferno", ax=ax1, linewidths=1.0, linecolor="black", square=True, cbar_kws={"fraction": 0.012}, vmin=vmin, vmax=vmax)
cbar1 = ax1.collections[0].colorbar
cbar1.outline.set_edgecolor("black")
cbar1.outline.set_linewidth(1.5)
ax1.set_xticks(np.arange(len(wt_domain_sequence)))
ax1.set_xticklabels(list(wt_domain_sequence), fontsize=22)
ax1.set_yticklabels(aa_list, fontsize=22)
ax1.text(1.02, 0.5, "P $(\mathrm{OFF})$", va="center", rotation=+90, fontsize=28, transform=ax1.transAxes)
ax1.set_xlabel("Wild Type Residue", fontsize=28)
ax1.set_ylabel("Mutant Residue", fontsize=28)
ax1.set_title("Actual Mutational Effects on ASCL1 Domain", fontsize=28)

# Heatmap 2: Predicted Mutational Effects
sns.heatmap(matrix2, cmap="inferno", ax=ax2, linewidths=1.0, linecolor="black", square=True, cbar_kws={"fraction": 0.012}, vmin=vmin, vmax=vmax)
cbar2 = ax2.collections[0].colorbar
cbar2.outline.set_edgecolor("black")
cbar2.outline.set_linewidth(1.5)
ax2.set_xticks(np.arange(len(wt_domain_sequence)))
ax2.set_xticklabels(list(wt_domain_sequence), fontsize=22)
ax2.set_yticklabels(aa_list, fontsize=22)
ax2.text(1.02, 0.5, "P $(\mathrm{OFF})$ Predicted", va="center", rotation=+90, fontsize=28, transform=ax2.transAxes)
ax2.set_xlabel("Wild Type Residue", fontsize=28)
ax2.set_ylabel("Mutant Residue", fontsize=28)
ax2.set_title("Predicted Mutational Effects on ASCL1 Domain", fontsize=28)

plt.tight_layout()
plt.savefig('./PLOTS/heatmap.png', dpi=500)
