import torch.nn as nn


class DNALinear(nn.Module):
  def __init__(self, seq_len, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.seq_len = seq_len
    self.lin = nn.Linear(in_features=4 * seq_len, out_features=1)
    pass

  def forward(self, xb):
    # reshape to flatten sequence dimension
    xb = xb.view(xb.shape[0], self.seq_len * 4)
    # Linear wraps up the weights/bias dot product operations
    out = self.lin(xb)
    return out
    pass


# basic cnn model
class DNACnn(nn.Module):
  def __init__(self, seq_len, num_filters=32, kernel_size=3, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.seq_len = seq_len

    self.conv_net = nn.Sequential(
      # 4 is for the 4 nucleotides
      nn.Conv1d(in_channels=4, out_channels=num_filters, kernel_size=kernel_size),
      nn.ReLU(inplace=True),
      nn.Flatten(),
      nn.Linear(num_filters * (seq_len - kernel_size + 1), 1)
    )
    pass

  def forward(self, xb):
    # permute to put channel in correct order
    # (batch_size x 4channel x seq_len)
    xb = xb.permute(0, 2, 1)

    # print(xb.shape)
    out = self.conv_net(xb)
    return out