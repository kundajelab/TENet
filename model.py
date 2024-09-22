# -*- coding: utf-8 -*-
"""
This module implements the training and evaluation of a neural network model (TENet) for designing transcriptional repressors.
Upon successfully running this code, the following files will be generated:

1. './model/model_no_attention.pt': This file contains the trained neural network model.
2. 'PERFORMANCE.txt': This file logs the performance metrics for each training epoch, including loss values and validation loss.

Ensure the required directories and files are properly set up to avoid issues during runtime. Specifically, please ensure that the file 'DATA/pickle_files' exists in the working directory.

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
from torch import optim
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from imblearn.over_sampling import RandomOverSampler
from collections import Counter

def load_data(directory='DATA/pickle_files'):
    """
    Load all .pkl files from the specified directory.

    Args:
    directory (str): Directory containing the pickle files. Defaults to 'DATA/pickle_files'.

    Returns:
    list: A list of data loaded from pickle files.
    """
    file_list = glob.glob(os.path.join(directory, '*.pkl'))
    data = []

    for file in file_list:
        with open(file, 'rb') as f:
            data.extend(pickle.load(f))

    return data

def process_data():
    """
    Processes the dataset by splitting it into training, validation, and test sets.
    
    The validation set consists of data from the RYBP, CBX5, and PCGF2 domains, 
    while all ASCL1 mutants are reserved for the test set. These splits are used to 
    evaluate the model's performance after training.

    Returns:
    tuple: Three lists containing the validation, test, and training data, respectively.
    """
    combined_data = load_data()

    # Define validation and test labels
    validation_labels = ['Silencer_tiles;ENSG00000163602;11', 'Silencer_tiles;ENSG00000094916;11', 'Silencer_tiles;ENSG00000277258;0']  # RYBP, CBX5, PCGF2
    test_labels = ['Short_nuclear_domain;ASCL1_HUMAN;HLH;105;52']
    combined_labels = validation_labels + test_labels  # Combined validation and test labels

    validation_ls, test_ls, train_ls = [], [], []

    # Separate data into training, validation, and test sets based on labels
    for data_point in combined_data:
        label = data_point[5]

        if any(sublabel in label for sublabel in validation_labels):
            validation_ls.append(data_point)
        elif any(sublabel in label for sublabel in test_labels):
            test_ls.append(data_point)
        elif not any(sublabel in label for sublabel in combined_labels):
            train_ls.append(data_point)

    return validation_ls, test_ls, train_ls

class Model(nn.Module):
    """
    A PyTorch model for sequence, contact map, and AA descriptor encoding using GRU and CNN layers.
    """
    def __init__(self, dropout_rate=0.5):
        super(Model, self).__init__()

        # Sequence encoder
        self.gru = nn.GRU(input_size=1280, hidden_size=256, num_layers=2, bidirectional=True, batch_first=True)
        self.bn_gru = nn.BatchNorm1d(512)  # Batch Normalization for GRU
        self.dropout_gru = nn.Dropout(dropout_rate)  # Dropout for GRU
        
        # AA Descriptor encoder
        self.gru_enc = nn.GRU(input_size=66, hidden_size=256, num_layers=2, bidirectional=True, batch_first=True)
        self.bn_gru_enc = nn.BatchNorm1d(512)  # Batch Normalization for GRU
        self.dropout_gru_enc = nn.Dropout(dropout_rate)  # Dropout for GRU
        
        # CNN for contact map encoding
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)  # Batch Normalization for conv1
        self.dropout1 = nn.Dropout2d(dropout_rate)  # Dropout for conv1
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)  # Batch Normalization for conv2
        self.dropout2 = nn.Dropout2d(dropout_rate)  # Dropout for conv2

        # Fully connected layer for final prediction
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
        torch.Tensor: Output prediction.
        """
        # Sequence encoding
        _, h_n = self.gru(seq)  # GRU output
        seq_feature = torch.cat((h_n[-2,:,:], h_n[-1,:,:]), dim=-1)  # Concatenate hidden states
        seq_feature = self.dropout_gru(self.bn_gru(seq_feature))  # Apply Batch Normalization and Dropout
        
        # AA Descriptors encoding
        _, h_n_enc = self.gru_enc(aa_descriptors_sequence)  # GRU for AA descriptors
        aa_descrp = torch.cat((h_n_enc[-2,:,:], h_n_enc[-1,:,:]), dim=-1)  # Concatenate hidden states
        aa_descrp = self.dropout_gru_enc(self.bn_gru_enc(aa_descrp))  # Apply Batch Normalization and Dropout

        # Contact map encoding using CNN
        contact_map = contact_map.unsqueeze(1)  # Add channel dimension
        x1 = self.dropout1(F.relu(self.bn1(self.conv1(contact_map))))
        x1 = F.max_pool2d(x1, 2)
        x1 = self.dropout2(F.relu(self.bn2(self.conv2(x1))))
        x1 = F.max_pool2d(x1, 2)
        map_feature = torch.flatten(x1, start_dim=1)

        # Concatenate sequence, map, and AA descriptor features and predict
        x = torch.cat((seq_feature, map_feature, aa_descrp), dim=-1)
        x = self.dropout_fc1(F.relu(self.bn_fc1(self.fc1(x))))
        x = torch.sigmoid(self.fc2(x))
        
        return x

class MyDataset(Dataset):
    """
    Custom Dataset class to load training data for PyTorch.
    """
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        """
        Fetch a single data point at the given index.

        Args:
        idx (int): Index of the data point.

        Returns:
        tuple: A tuple of token representations, contact maps, AA descriptors, countsOFF, and countsON.
        """
        # Data Format: token_representations, attention_contacts, dist_matrix, aa_descriptors_sequence, aa_seq, label, R1, countsOFF_R1, countsON_R1, R2, countsOFF_R2, countsON_R2, Avg
        token_representations = torch.tensor(self.data_list[idx][0], dtype=torch.float)
        attention_contacts = torch.tensor(self.data_list[idx][1], dtype=torch.float)
        dist_matrix = torch.tensor(self.data_list[idx][2], dtype=torch.float)
        aa_descriptors_sequence = torch.tensor(self.data_list[idx][3], dtype=torch.float)

        # Compute total reads for each biorep
        total_reads_R1 = self.data_list[idx][7] + self.data_list[idx][8]
        total_reads_R2 = self.data_list[idx][10] + self.data_list[idx][11]
        
        # Use total reads as weights to compute weighted averages
        countsOFF = torch.tensor((self.data_list[idx][7] * total_reads_R1 + self.data_list[idx][10] * total_reads_R2) / (total_reads_R1 + total_reads_R2), dtype=torch.float)
        countsON = torch.tensor((self.data_list[idx][8] * total_reads_R1 + self.data_list[idx][11] * total_reads_R2) / (total_reads_R1 + total_reads_R2), dtype=torch.float)
        
        return token_representations, attention_contacts, dist_matrix, aa_descriptors_sequence, countsOFF, countsON

def loss_fn(P_OFF, N_OFF, N_ON, epsilon=1e-8):
    """
    Custom loss function for the model.

    Args:
    P_OFF (torch.Tensor): Predicted P_OFF values.
    N_OFF (torch.Tensor): Ground truth OFF counts.
    N_ON (torch.Tensor): Ground truth ON counts.
    epsilon (float): Small value to prevent division by zero.

    Returns:
    torch.Tensor: Computed loss.
    """
    return -N_OFF * torch.log(P_OFF + epsilon) - N_ON * torch.log(1 - P_OFF + epsilon)

def load_and_process_data():
    """
    Loads and processes the data, including oversampling to handle class imbalance.

    Returns:
    tuple: Train dataset, validation dataset, validation data length.
    """
    start_time = time.time()
    
    validation_ls, _, train_ls = process_data()  # Load and process data
    
    with open('logfile.txt', 'a') as f:
        f.write('Total data load time: {}\n'.format(time.time() - start_time))
    
    # Label the training data
    train_label = [1 if data_point[7] / (data_point[7] + data_point[8]) > 0.5 else 0 for data_point in train_ls]
    with open('logfile.txt', 'a') as f:
        f.write('Original dataset shape {}\n'.format(Counter(train_label)))
    
    # Perform Random OverSampling to balance classes
    train_resampled, train_label_resampled = [], []
    max_samples = max(Counter(train_label).values())
    class_indices = {label: [] for label in set(train_label)}
    for i, label in enumerate(train_label):
        class_indices[label].append(i)
    
    for label, indices in class_indices.items():
        oversampled_indices = np.random.choice(indices, size=max_samples, replace=True)
        train_resampled.extend(train_ls[i] for i in oversampled_indices)
        train_label_resampled.extend(train_label[i] for i in oversampled_indices)
    
    with open('logfile.txt', 'a') as f:
        f.write('Resampled dataset shape {}\n'.format(Counter([data_point[7] / (data_point[7] + data_point[8]) > 0.5 for data_point in train_resampled])))
    
    train_data = MyDataset(train_resampled)
    val_data = MyDataset(validation_ls)
    
    return train_data, val_data, len(validation_ls)

# Training parameters
batch_size = 512
learning_rate = 0.001  
model_name = './model/model_no_attention.pt'
patience = 30  # Early stopping patience

# Create the 'model' directory if it doesn't exist
if not os.path.exists('model'):
    os.makedirs('model')

train_data, val_data, val_data_len = load_and_process_data()  # Load and process data

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)

# Set up device (GPU if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Instantiate model, optimizer, and learning rate scheduler
model = Model(dropout_rate=0.5).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

# Training Loop
num_epochs = 100
best_val_loss = float('inf')  # Track the best validation loss for early stopping
num_bad_epochs = 0  # Track the number of bad epochs for early stopping

for epoch in range(num_epochs):
    model.train()  # Set model to training mode
    start_time_epoch = time.time()  # Start timing for epoch
    
    total_loss = 0
    total_batches = 0
    
    # Training loop
    for i, (token_representations, attention_contacts, dist_matrix, aa_descriptors_sequence, countsOFF, countsON) in enumerate(train_loader):
        
        start_time_batch = time.time()  # Start timing for batch
        
        # Move tensors to device (GPU or CPU)
        token_representations = token_representations.to(device)
        dist_matrix = dist_matrix.to(device)
        aa_descriptors_sequence = aa_descriptors_sequence.to(device)
        countsOFF = countsOFF.unsqueeze(-1).to(device)
        countsON = countsON.unsqueeze(-1).to(device) 
        P_off_actual = countsOFF / (countsOFF + countsON)  # Compute P_OFF
        
        # Forward pass
        outputs = model(token_representations, dist_matrix, aa_descriptors_sequence)
        loss = loss_fn(outputs, countsOFF, countsON).mean()  # Compute loss

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_batches = i + 1 
        
        torch.cuda.empty_cache()  # Clear cache at the end of each batch

    avg_loss = total_loss / total_batches  # Average loss for epoch

    # Validation loop
    model.eval()  # Set model to evaluation mode
    val_loss = 0
    with torch.no_grad():
        for i, (token_representations, attention_contacts, dist_matrix, aa_descriptors_sequence, countsOFF, countsON) in enumerate(val_loader):
            # Move tensors to device
            token_representations = token_representations.to(device)
            dist_matrix = dist_matrix.to(device)
            aa_descriptors_sequence = aa_descriptors_sequence.to(device)
            countsOFF = countsOFF.unsqueeze(-1).to(device)
            countsON = countsON.unsqueeze(-1).to(device)
            P_off_actual = countsOFF / (countsOFF + countsON)  # Compute P_OFF
            
            # Forward pass
            outputs = model(token_representations, dist_matrix, aa_descriptors_sequence)
            loss = loss_fn(outputs, countsOFF, countsON).mean()  # Compute loss
            val_loss += loss.item()

    val_loss /= len(val_loader)  # Average validation loss
    
    # Write performance to file
    with open('PERFORMANCE.txt', 'a') as f:
        f.write(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}, Epoch Time: {time.time()-start_time_epoch:.2f} sec\n')
    
    # Adjust learning rate based on validation loss
    scheduler.step(val_loss)

    # Early stopping logic
    if val_loss < best_val_loss:
        torch.save(model.state_dict(), model_name)  # Save model
        best_val_loss = val_loss
        num_bad_epochs = 0  # Reset bad epoch count
    else:
        num_bad_epochs += 1

    # Stop training if validation loss doesn't improve for a set number of epochs
    if num_bad_epochs == patience:
        print(f'Early stopping triggered after epoch {epoch+1}')
        break

    torch.cuda.empty_cache()  # Clear cache at the end of each epoch
