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
from test import *
from visualize import *

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


def parity_pred(models, seqs, oracle, alt=False):
  '''Given some sequences, get the model's predictions '''
  dfs = {}  # key: model name, value: parity_df

  for model_name, model in models:
    print(f"Running {model_name}")
    data = []
    for dna in seqs:
      s = torch.tensor(one_hot_encode(dna)).unsqueeze(0).to(DEVICE)
      actual = oracle[dna]
      pred = model(s.float())
      data.append([dna, actual, pred.item()])
    df = pd.DataFrame(data, columns=['seq', 'truth', 'pred'])
    r2 = r2_score(df['truth'], df['pred'])
    dfs[model_name] = (r2, df)

    # plot parity plot
    if alt:  # make an altair plot
      alt_parity_plot(model_name, df, r2)
    else:
      parity_plot(model_name, df, r2)


def get_conv_output_for_seq(seq, conv_layer):
  '''
  Given an input sequeunce and a convolutional layer,
  get the output tensor containing the conv filter
  activations along each position in the sequence
  '''

  # format seq for input to conv layer (OHE, reshape)
  seq = torch.tensor(one_hot_encode(seq)).unsqueeze(0).permute(0, 2, 1).to(DEVICE)
  # run seq through conv layer
  with torch.no_grad():  # don't want as part of gradient graph
    # apply learned filters to input seq
    res = conv_layer(seq.float())
    return res[0]
  pass


def get_filter_activations(seqs, conv_layer, act_thresh=0):
  '''
    Given a set of input sequences and a trained convolutional layer,
    determine the subsequences for which each filter in the conv layer
    activate most strongly.

    1.) Run seq inputs through conv layer.
    2.) Loop through filter activations of the resulting tensor, saving the
            position where filter activations were > act_thresh.
    3.) Compile a count matrix for each filter by accumulating subsequences which
            activate the filter above the threshold act_thresh
    '''
  # initialize dict of pwms for each filter in the conv layer
  # pwm shape: 4 nucleotides X filter width, initialize to 0.0s
  num_filters = conv_layer.out_channels
  filt_width = conv_layer.kernel_size[0]
  filter_pwms = dict((i, torch.zeros(4, filt_width)) for i in range(num_filters))

  print("Num filters", num_filters)
  print("filt_width", filt_width)

  # loop through a set of sequences and collect subseqs where each filter activated
  for seq in seqs:
    # get a tensor of each conv filter activation along the input seq
    res = get_conv_output_for_seq(seq, conv_layer)

    # for each filter and it's activation vector
    for filt_id, act_vec in enumerate(res):
      # collect the indices where the activation level
      # was above the threshold
      act_idxs = torch.where(act_vec > act_thresh)[0]
      activated_positions = [x.item() for x in act_idxs]

      # use activated indicies to extract the actual DNA
      # subsequences that caused filter to activate
      for pos in activated_positions:
        subseq = seq[pos:pos + filt_width]
        # print("subseq",pos, subseq)
        # transpose OHE to match PWM orientation
        subseq_tensor = torch.tensor(one_hot_encode(subseq)).T

        # add this subseq to the pwm count for this filter
        filter_pwms[filt_id] += subseq_tensor

  return filter_pwms


def view_filters_and_logos(model_weights, filter_activations, num_cols=8):
  '''
    Given some convolutional model weights and filter activation PWMs,
    visualize the heatmap and motif logo pairs in a simple grid
    '''
  model_weights = model_weights[0].squeeze(1)
  print(model_weights.shape)

  # make sure the model weights agree with the number of filters
  assert (model_weights.shape[0] == len(filter_activations))

  num_filts = len(filter_activations)
  num_rows = int(np.ceil(num_filts / num_cols)) * 2 + 1
  # ^ not sure why +1 is needed... complained otherwise

  plt.figure(figsize=(20, 17))

  j = 0  # use to make sure a filter and it's logo end up vertically paired
  for i, filter in enumerate(model_weights):
    if (i) % num_cols == 0:
      j += num_cols

    # display raw filter
    ax1 = plt.subplot(num_rows, num_cols, i + j + 1)
    ax1.imshow(filter.cpu().detach(), cmap='gray')
    ax1.set_yticks(np.arange(4))
    ax1.set_yticklabels(['A', 'C', 'G', 'T'])
    ax1.set_xticks(np.arange(model_weights.shape[2]))
    ax1.set_title(f"Filter {i}")

    # display sequence logo
    ax2 = plt.subplot(num_rows, num_cols, i + j + 1 + num_cols)
    filt_df = pd.DataFrame(filter_activations[i].T.numpy(), columns=['A', 'C', 'G', 'T'])
    filt_df_info = logomaker.transform_matrix(filt_df, from_type='counts', to_type='information')
    logo = logomaker.Logo(filt_df_info, ax=ax2)
    ax2.set_ylim(0, 2)
    ax2.set_title(f"Filter {i}")

  plt.tight_layout()
  plt.show()


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

  # generate plots
  seqs = test_df['seq'].values
  models = [
    ("Linear", model_lin),
    ("CNN", model_cnn)
  ]
  parity_pred(models, seqs, oracle)

  conv_layers, model_weights, bias_weights = get_conv_layers_from_model(model_cnn)
  view_filters(model_weights)

  # just use some seqs from test_df to activate filters
  some_seqs = random.choices(seqs, k=3000)

  filter_activations = get_filter_activations(some_seqs, conv_layers[0], act_thresh=1)
  view_filters_and_logos(model_weights, filter_activations)

  # prints:
  # Num filters 32
  # filt_width 3
  # torch.Size([32, 4, 3])
  pass
