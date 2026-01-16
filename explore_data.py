
import pandas as pd
from collections import Counter

def parse_fasta(file_path):
    sequences = {}
    with open(file_path, 'r') as f:
        entry_id = None
        seq = []
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if entry_id:
                    sequences[entry_id] = ''.join(seq)
                # Extract EntryID from header (e.g., >sp|A0A0C5B5G6|MOTSC_HUMAN -> A0A0C5B5G6)
                # Note: The format seems to be >sp|EntryID|... or just >EntryID
                # Let's verify the header format from the previous `read` output:
                # >sp|A0A0C5B5G6|MOTSC_HUMAN ...
                parts = line.split('|')
                if len(parts) >= 2:
                    entry_id = parts[1]
                else:
                    entry_id = line[1:].split()[0] # Fallback
                seq = []
            else:
                seq.append(line)
        if entry_id:
            sequences[entry_id] = ''.join(seq)
    return sequences

print("Loading sequences...")
train_seqs = parse_fasta('Train/train_sequences.fasta')
print(f"Total training sequences: {len(train_seqs)}")

print("Loading terms...")
train_terms = pd.read_csv('Train/train_terms.tsv', sep='\t')
print(f"Total annotations: {len(train_terms)}")
print(f"Unique proteins with annotations: {train_terms['EntryID'].nunique()}")
print(f"Unique GO terms: {train_terms['term'].nunique()}")

# Term frequency
term_counts = train_terms['term'].value_counts()
print("\nTop 10 most frequent terms:")
print(term_counts.head(10))

print("\nTerm frequency stats:")
print(term_counts.describe())

# Check intersection
seq_ids = set(train_seqs.keys())
term_ids = set(train_terms['EntryID'].unique())
intersection = seq_ids.intersection(term_ids)
print(f"\nProteins in fasta: {len(seq_ids)}")
print(f"Proteins with terms: {len(term_ids)}")
print(f"Proteins in both: {len(intersection)}")
