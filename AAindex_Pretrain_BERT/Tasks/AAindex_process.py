from Bio import SeqIO
import pandas as pd
from sklearn.model_selection import train_test_split
import pandas as pd
from scipy.stats import zscore
import gzip
# calculate the property of the peptides
def calculate_features(aaindex_df, peptide):
    # initinal the feature vector
    num_features = len(aaindex_df.index)
    feature_vector = [0.0] * num_features

    # enumerate the peptide sequence 
    for aa in peptide:
        if aa in aaindex_df.columns:
            aa_features = aaindex_df.loc[:,aa]
            feature_vector = [a + b for a, b in zip(feature_vector, aa_features)]

    # Normalized eigenvalue vector
    peptide_length = len(peptide)
    if peptide_length > 0:
        feature_vector = [round(val / peptide_length, 3) for val in feature_vector]

    return feature_vector
def z_score(df):
    # copy the dataframe
    df_std = df.copy()
    # apply the z-score method
    for column in df_std.columns:
        df_std[column] = (df_std[column] - df_std[column].mean()) / df_std[column].std()

    return df_std

# read AAindex file
aaindex_df = pd.read_csv('../data/aaindex1.csv',index_col=0)
aaindex_df = aaindex_df.dropna(axis=0)
fasta_file_path = '../data/uniref_identity_0_5_AND_length_TO_60_2023_10_09.fasta'
# Parse fasta files, obtain peptide sequences and filter out non-natural amino acid sequences
peptide_sequences = {}
with gzip.open(fasta_file_path,'rt') as handle:
    for record in SeqIO.parse(handle, "fasta"):
        sequence = str(record.seq)
        if all(aa in 'ACDEFGHIKLMNPQRSTVWY' for aa in sequence):  # Ensure that the sequence contains only natural amino acids
            peptide_sequences[record.id] = sequence
# Calculate the feature value of each peptide sequence and save it to a CSV file
output_data = []
for seq_id, sequence in peptide_sequences.items():
    peptide_features = calculate_features(aaindex_df, sequence)
    output_data.append([sequence] + peptide_features)
# Save the results to a CSV file
output_csv = '../data/output_filtered_features.csv'
header = ['Sequence'] + [f'Feature_{i+1}' for i in range(len(output_data[0])-1)]
output_df = pd.DataFrame(output_data, columns=header)
output_df.to_csv(output_csv, index=False)

random_state =42
data = pd.read_csv('../data/output_filtered_features.csv')
#data_n = data.apply(zscore)
data.iloc[:, 1:] = data.iloc[:, 1:].apply(zscore)
data.iloc[:, 1:] = data.iloc[:, 1:].round(3)
train_set, test_set = train_test_split(data,test_size=0.1,random_state=42)
train_set.to_csv('../data/train_set.csv', index=False)
test_set.to_csv('../data/test_set.csv', index=False)