# This is a sample Python script.
import random
from itertools import product
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch
from models import DNALinear, DNACnn
from training import *

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



class SeqDatasetOHE(Dataset):
    '''
    Dataset for one-hot-encoded sequences
    '''

    def __init__(self,
                 df,
                 seq_col='seq',
                 target_col='score'
                 ):
        # extract the DNA from the appropriate column in the df
        self.seqs = df[seq_col].to_list()
        self.seq_len = len(self.seqs[0])

        # one-hot encode sequences, then stack in a torch tensor
        self.ohe_seqs = torch.stack([torch.tensor(one_hot_encode(x)) for x in self.seqs])

        # +------------------+
        # | Get the Y labels |
        # +------------------+
        self.labels = torch.tensor(df[target_col].to_list()).unsqueeze(1)

        pass

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        # Given an index, return a tuple of an X with it's associated Y
        # This is called inside DataLoader
        seq = self.ohe_seqs[idx]
        label = self.labels[idx]

        return seq, label


## Here is how I constructed DataLoaders from Datasets.
def build_dataloaders(train_df,
                      test_df,
                      seq_col='seq',
                      target_col='score',
                      batch_size=128,
                      shuffle=True
                      ):
    '''
    Given a train and test df with some batch construction
    details, put them into custom SeqDatasetOHE() objects.
    Give the Datasets to the DataLoaders and return.
    '''

    # create Datasets
    train_ds = SeqDatasetOHE(train_df, seq_col=seq_col, target_col=target_col)
    test_ds = SeqDatasetOHE(test_df, seq_col=seq_col, target_col=target_col)

    # Put DataSets into DataLoaders
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle)
    test_dl = DataLoader(test_ds, batch_size=batch_size)

    return train_dl, test_dl


# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def kmers(k):
    "Generate a list of all kmers for a given k"
    return [''.join(x) for x in product(['A', 'C', 'G', 'T'], repeat=k)]


random_variable_dict = {
    "A": 20,
    "C": 17,
    "G": 14,
    "T": 11
}


def score_sequence_motif(seqs):
    '''
    Calculate the scores for a list of sequences based on
    the above random_variable_dict
    '''
    data: list[list] = []
    for seq in seqs:
        # get the average score by nucleotide
        score = np.mean([random_variable_dict[base] for base in seq])
        if 'TAT' in seq:
            score += 10
        if 'GCG' in seq:
            score -= 10
        data.append([seq, score])
    df = pd.DataFrame(data, columns=['seq', 'score'])
    return df


def one_hot_encode(seq):
    """
    Given a DNA sequence, return its one-hot encoding
    """
    # Make sure seq has only allowed bases
    allowed = set("ACTGN")
    if not set(seq).issubset(allowed):
        invalid = set(seq) - allowed
        raise ValueError(f"Sequence contains chars not in allowed DNA alphabet (ACGTN): {invalid}")

    # Dictionary returning one-hot encoding for each nucleotide
    nuc_d = {'A': [1.0, 0.0, 0.0, 0.0],
             'C': [0.0, 1.0, 0.0, 0.0],
             'G': [0.0, 0.0, 1.0, 0.0],
             'T': [0.0, 0.0, 0.0, 1.0],
             'N': [0.0, 0.0, 0.0, 0.0]}

    # Create array from nucleotide sequence
    vec = np.array([nuc_d[x] for x in seq])

    return vec


def quick_split(df, split_frac=0.8):
    '''
    Given a df of samples, randomly split indices between
    train and test at the desired fraction
    '''
    cols = df.columns  # original columns, use to clean up reindexed cols
    df = df.reset_index()

    # shuffle indices
    idxs = list(range(df.shape[0]))
    random.shuffle(idxs)

    # split shuffled index list by split_frac
    split = int(len(idxs) * split_frac)
    train_idxs = idxs[:split]
    test_idxs = idxs[split:]

    # split dfs and return
    train_df = df[df.index.isin(train_idxs)]
    test_df = df[df.index.isin(test_idxs)]

    return train_df[cols], test_df[cols]


def quick_loss_plot(data_label_list, loss_type="MSE Loss", sparse_n=0):
    '''
    For each train/test loss trajectory, plot loss by epoch
    '''
    for i, (train_data, test_data, label) in enumerate(data_label_list):
        plt.plot(train_data, linestyle='--', color=f"C{i}", label=f"{label} Train")
        plt.plot(test_data, color=f"C{i}", label=f"{label} Val", linewidth=3.0)

    plt.legend()
    plt.ylabel(loss_type)
    plt.xlabel("Epoch")
    plt.legend(bbox_to_anchor=(1, 1), loc='upper left')
    plt.show()

def run_cnn_model():
    # DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seq_len = len(train_df['seq'].values[0])

    # create Linear model object
    model_cnn = DNACnn(seq_len)
    model_cnn.to(DEVICE)  # put on GPU

    # run the model with default settings!
    cnn_train_losses, cnn_val_losses = run_model(
        train_dl,
        val_dl,
        model_cnn,
        DEVICE
    )

    cnn_data_label = (cnn_train_losses, cnn_val_losses, "CNN")
    return cnn_data_label, model_cnn

def run_linear_model():
    # use GPU if available
    # DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # get the sequence length from the first seq in the df
    seq_len = len(train_df['seq'].values[0])

    # create Linear model object
    model_lin = DNALinear(seq_len)
    model_lin.to(DEVICE)  # put on GPU

    # run the model with default settings!
    lin_train_losses, lin_val_losses = run_model(
        train_dl,
        val_dl,
        model_lin,
        DEVICE
    )

    lin_data_label = (lin_train_losses, lin_val_losses, "Lin")
    return lin_data_label, model_lin





def quick_seq_pred(model, desc, seqs, oracle):
    '''
    Given a model and some sequences, get the model's predictions
    for those sequences and compare to the oracle (true) output
    '''
    print(f"__{desc}__")
    for dna in seqs:
        s = torch.tensor(one_hot_encode(dna)).unsqueeze(0).to(DEVICE)
        pred = model(s.float())
        actual = oracle[dna]
        diff = pred.item() - actual
        print(f"{dna}: pred:{pred.item():.3f} actual:{actual:.3f} ({diff:.3f})")


def quick_8mer_pred(model, oracle):  # model interpretation?
    seqs1 = ("poly-X seqs", ['AAAAAAAA', 'CCCCCCCC', 'GGGGGGGG', 'TTTTTTTT'])
    seqs2 = ("other seqs", ['AACCAACA', 'CCGGTGAG', 'GGGTAAGG', 'TTTCGTTT'])
    seqsTAT = ("with TAT motif", ['TATAAAAA', 'CCTATCCC', 'GTATGGGG', 'TTTATTTT'])
    seqsGCG = ("with GCG motif", ['AAGCGAAA', 'CGCGCCCC', 'GGGCGGGG', 'TTGCGTTT'])
    TATGCG = ("both TAT and GCG", ['ATATGCGA', 'TGCGTATT'])

    for desc, seqs in [seqs1, seqs2, seqsTAT, seqsGCG, TATGCG]:
        quick_seq_pred(model, desc, seqs, oracle)
        print()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    seqs8 = kmers(8)
    print('Total 8mers:', len(seqs8))
    # prints: Total 8mers: 65536
    mer8 = score_sequence_motif(seqs8)

    a8 = one_hot_encode("AAAAAAAA")
    print("AAAAAAAA:\n", a8)

    s = one_hot_encode("AGGTACCT")
    print("AGGTACCT:\n", s)
    print("shape:", s.shape)

    full_train_df, test_df = quick_split(mer8)
    train_df, val_df = quick_split(full_train_df)

    print("Train:", train_df.shape)
    print("Val:", val_df.shape)
    print("Test:", test_df.shape)

    # prints:
    # Train: (41942, 2)
    # Val: (10486, 2)
    # Test: (13108, 2)

    train_dl, val_dl = build_dataloaders(train_df, val_df)
    lin_data_label, model_lin = run_linear_model()
    cnn_data_label, model_cnn = run_cnn_model()
    quick_loss_plot([lin_data_label, cnn_data_label])

    # oracle dict of true score for each seq
    oracle = dict(mer8[['seq', 'score']].values)

    quick_8mer_pred(model_lin, oracle)

    pass
