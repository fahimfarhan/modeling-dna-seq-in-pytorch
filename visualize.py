import logomaker
import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt


def get_conv_layers_from_model(model):
  '''
  Given a trained model, extract its convolutional layers
  '''
  model_children = list(model.children())

  # counter to keep count of the conv layers
  model_weights = []  # we will save the conv layer weights in this list
  conv_layers = []  # we will save the actual conv layers in this list
  bias_weights = []
  counter = 0

  for i in range(len(model_children)):
    # get model type of Conv1d
    if type(model_children[i]) == nn.Conv1d:
      counter += 1
      model_weights.append(model_children[i].weight)
      conv_layers.append(model_children[i])
      bias_weights.append(model_children[i].bias)

    # also check sequential objects' children for conv1d
    elif type(model_children[i]) == nn.Sequential:
      for child in model_children[i]:
        if type(child) == nn.Conv1d:
          counter += 1
          model_weights.append(child.weight)
          conv_layers.append(child)
          bias_weights.append(child.bias)
  print(f"Total convolutional layers: {counter}")
  return conv_layers, model_weights, bias_weights


def view_filters(model_weights, num_cols=8):
  model_weights = model_weights[0]
  num_filt = model_weights.shape[0]
  filt_width = model_weights[0].shape[1]
  num_rows = int(np.ceil(num_filt / num_cols))

  # visualize the first conv layer filters
  plt.figure(figsize=(20, 17))

  for i, filter in enumerate(model_weights):
    ax = plt.subplot(num_rows, num_cols, i + 1)
    ax.imshow(filter.cpu().detach(), cmap='gray')
    ax.set_yticks(np.arange(4))
    ax.set_yticklabels(['A', 'C', 'G', 'T'])
    ax.set_xticks(np.arange(filt_width))
    ax.set_title(f"Filter {i}")

  plt.tight_layout()
  plt.show()
  pass


