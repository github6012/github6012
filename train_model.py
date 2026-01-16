
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from collections import Counter
import tqdm

# --- Configuration ---
config = {
    'train_seq_path': 'Train/train_sequences.fasta',
    'train_terms_path': 'Train/train_terms.tsv',
    'test_seq_path': 'Test/testsuperset.fasta',
    'model_save_path': 'protein_function_model.pth',
    'submission_path': 'submission.tsv',
    'max_len': 512,       # Reduced for speed
    'vocab_size': 26,     # Size of amino acid vocabulary (approx)
    'embedding_dim': 128,
    'num_classes': 500,   # Top N most frequent GO terms to predict
    'batch_size': 32,
    'epochs': 1,          # Reduced for demonstration
    'learning_rate': 0.001,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'quick_run': True     # Flag to use subset of data
}

print(f"Using device: {config['device']}")

# --- 1. Data Loading & Preprocessing ---

def parse_fasta(file_path):
    sequences = []
    ids = []
    with open(file_path, 'r') as f:
        entry_id = None
        seq = []
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if entry_id:
                    sequences.append(''.join(seq))
                    ids.append(entry_id)
                parts = line.split('|')
                if len(parts) >= 2:
                    entry_id = parts[1]
                else:
                    entry_id = line[1:].split()[0]
                seq = []
            else:
                seq.append(line)
        if entry_id:
            sequences.append(''.join(seq))
            ids.append(entry_id)
    return ids, sequences

# Amino acid vocabulary
# Standard: ACDEFGHIKLMNPQRSTVWY
# Map all to 1-20, pad=0, unknown=21
AA_MAP = {
    'A': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'K': 9, 'L': 10,
    'M': 11, 'N': 12, 'P': 13, 'Q': 14, 'R': 15, 'S': 16, 'T': 17, 'V': 18, 'W': 19, 'Y': 20
}
# Others map to 21
UNK_IDX = 21

def encode_sequence(seq, max_len):
    # Convert chars to indices
    indices = [AA_MAP.get(aa, UNK_IDX) for aa in seq]
    # Truncate or pad
    if len(indices) > max_len:
        indices = indices[:max_len]
    else:
        indices = indices + [0] * (max_len - len(indices))
    return np.array(indices, dtype=np.int64)

# --- 2. Dataset Class ---

class ProteinDataset(Dataset):
    def __init__(self, ids, sequences, labels=None, num_classes=None):
        self.ids = ids
        self.sequences = sequences
        self.labels = labels # Multi-hot encoded labels (numpy array)
        
    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, idx):
        seq_indices = encode_sequence(self.sequences[idx], config['max_len'])
        
        item = {
            'id': self.ids[idx],
            'sequence': torch.tensor(seq_indices, dtype=torch.long)
        }
        
        if self.labels is not None:
            item['label'] = torch.tensor(self.labels[idx], dtype=torch.float32)
            
        return item

# --- 3. Model Definition ---

class ProteinCNN(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, num_classes):
        super(ProteinCNN, self).__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim, padding_idx=0)
        self.conv1 = nn.Conv1d(in_channels=embedding_dim, out_channels=256, kernel_size=9, padding=4)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveMaxPool1d(1) # Global Max Pooling
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        # x: [batch_size, seq_len]
        x = self.embedding(x) # [batch_size, seq_len, emb_dim]
        x = x.permute(0, 2, 1) # [batch_size, emb_dim, seq_len] for Conv1d
        
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x).squeeze(-1) # [batch_size, 256]
        
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x) # Logits
        return x

# --- 4. Main Execution ---

def main():
    # A. Load Data
    print("Loading raw data...")
    train_ids, train_seqs = parse_fasta(config['train_seq_path'])
    train_terms = pd.read_csv(config['train_terms_path'], sep='\t')
    
    # B. Filter Terms (Top N)
    print(f"Selecting top {config['num_classes']} most frequent GO terms...")
    term_counts = train_terms['term'].value_counts()
    top_terms = term_counts.head(config['num_classes']).index.tolist()
    term_to_idx = {term: i for i, term in enumerate(top_terms)}
    idx_to_term = {i: term for term, i in term_to_idx.items()}
    
    # C. Create Labels Matrix
    # We need a map from EntryID to label vector
    # Filter train_terms to only keep rows with top terms
    filtered_terms = train_terms[train_terms['term'].isin(top_terms)]
    
    print("Creating label matrix...")
    # Group by ID
    id_to_terms = filtered_terms.groupby('EntryID')['term'].apply(list).to_dict()
    
    # Align labels with train_ids
    # Note: Some sequences in fasta might not have terms (though our exploration said all do)
    # Some sequences might not have terms in the *top N* set.
    
    final_ids = []
    final_seqs = []
    final_labels = []
    
    for pid, seq in zip(train_ids, train_seqs):
        if pid in id_to_terms:
            terms = id_to_terms[pid]
            label_vec = np.zeros(config['num_classes'])
            for t in terms:
                label_vec[term_to_idx[t]] = 1
            
            final_ids.append(pid)
            final_seqs.append(seq)
            final_labels.append(label_vec)
    
    print(f"Samples after filtering: {len(final_ids)}")

    if config.get('debug', False):
        print("DEBUG MODE: Limiting to 2000 samples and 1 epoch")
        final_ids = final_ids[:2000]
        final_seqs = final_seqs[:2000]
        final_labels = final_labels[:2000]
        config['epochs'] = 1
    
    # Quick run limit
    if config.get('quick_run'):
        print("Quick run mode: Limiting dataset size to 10,000 samples.")
        final_ids = final_ids[:10000]
        final_seqs = final_seqs[:10000]
        final_labels = final_labels[:10000]

    # D. Train/Val Split
    X_train_ids, X_val_ids, X_train_seqs, X_val_seqs, y_train, y_val = train_test_split(
        final_ids, final_seqs, final_labels, test_size=0.1, random_state=42
    )
    
    # E. Datasets & Loaders
    train_dataset = ProteinDataset(X_train_ids, X_train_seqs, y_train)
    val_dataset = ProteinDataset(X_val_ids, X_val_seqs, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    
    # F. Initialize Model
    model = ProteinCNN(num_embeddings=22, embedding_dim=config['embedding_dim'], num_classes=config['num_classes'])
    model = model.to(config['device'])
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    # G. Training Loop
    print("Starting training...")
    for epoch in range(config['epochs']):
        model.train()
        train_loss = 0
        for batch in tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']}"):
            inputs = batch['sequence'].to(config['device'])
            targets = batch['label'].to(config['device'])
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch['sequence'].to(config['device'])
                targets = batch['label'].to(config['device'])
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
    
    # H. Save Model
    torch.save(model.state_dict(), config['model_save_path'])
    print(f"Model saved to {config['model_save_path']}")
    
    # I. Save Term Mapping (needed for inference)
    import json
    with open('term_mapping.json', 'w') as f:
        json.dump(idx_to_term, f)
        
    # --- Prediction on Test Set ---
    print("Generating predictions for test set...")
    test_ids, test_seqs = parse_fasta(config['test_seq_path'])
    test_dataset = ProteinDataset(test_ids, test_seqs, labels=None) # No labels
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
    
    predictions = []
    
    model.eval()
    with torch.no_grad():
        for batch in tqdm.tqdm(test_loader, desc="Predicting"):
            ids = batch['id']
            inputs = batch['sequence'].to(config['device'])
            outputs = model(inputs)
            probs = torch.sigmoid(outputs).cpu().numpy()
            
            for i, pid in enumerate(ids):
                # For each protein, we have 500 probabilities
                # We need to format it as: ProteinID, GO_ID, Score
                # To save space, maybe just keep top k or all?
                # The submission format usually requires specific terms.
                # For this demo, let's just save the top 10 predictions per protein to a file.
                
                p_probs = probs[i]
                # Get indices of top 10
                top_k_indices = np.argsort(p_probs)[-20:][::-1]
                
                for idx in top_k_indices:
                    term = idx_to_term[idx]
                    score = p_probs[idx]
                    predictions.append(f"{pid}\t{term}\t{score:.4f}")

    with open(config['submission_path'], 'w') as f:
        f.write("EntryID\tterm\tscore\n")
        f.write('\n'.join(predictions))
        
    print(f"Submission file saved to {config['submission_path']}")

if __name__ == "__main__":
    main()
