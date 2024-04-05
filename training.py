import numpy as np
import torch


def loss_batch(model, loss_func, xb, yb, opt=None, verbose=False):
  '''
  Apply loss function to a batch of inputs. If no optimizer
  is provided, skip the back prop step.
  '''
  if verbose:
    print('loss batch ****')
    print("xb shape:", xb.shape)
    print("yb shape:", yb.shape)
    print("yb shape:", yb.squeeze(1).shape)
    # print("yb",yb)
  # get the batch output from the model given your input batch
  # ** This is the model's prediction for the y labels! **
  xb_out = model(xb.float())

  if verbose:
    print("model out pre loss", xb_out.shape)
    # print('xb_out', xb_out)
    print("xb_out:", xb_out.shape)
    print("yb:", yb.shape)
    print("yb.long:", yb.long().shape)

  loss = loss_func(xb_out, yb.float()) # for MSE / regression
  # __FOOTNOTE 2__

  if opt is not None:
    loss.backward()
    opt.step()
    opt.zero_grad()
  return loss.item(), len(xb)

def train_step(model, train_dl, loss_func, device, opt):
  model.train()
  tl = [] # training loss
  ns = [] # batch size, n

  for xb, yb in train_dl:
    xb, yb = xb.to(device), yb.to(device)
    t, n = loss_batch(model, loss_func, xb, yb, opt)

    # collect train loss and batch sizes
    tl.append(t)
    ns.append(n)
    # average the losses over all batches
    train_loss = np.sum(np.multiply(tl, ns)) / np.sum(ns)

    return train_loss

def val_step(model, val_dl, loss_func, device):
  model.eval()
  with torch.no_grad():
    vl = []  # val losses
    ns = []  # batch sizes, n
    # loop through validation DataLoader
    for xb, yb in val_dl:
      # put on GPU
      xb, yb = xb.to(device), yb.to(device)

      # Do NOT provide opt here, so backprop does not happen
      v, n = loss_batch(model, loss_func, xb, yb)

      # collect val loss and batch sizes
      vl.append(v)
      ns.append(n)

    # average the losses over all batches
    val_loss = np.sum(np.multiply(vl, ns)) / np.sum(ns)

    return val_loss

def fit(epochs, model, loss_func, opt, train_dl, val_dl,device,patience=1000):
  # keep track of losses
  train_losses = []
  val_losses = []

  # loop through epochs
  for epoch in range(epochs):
    # take a training step
    train_loss = train_step(model, train_dl, loss_func, device, opt)
    train_losses.append(train_loss)

    # take a validation step
    val_loss = val_step(model, val_dl, loss_func, device)
    val_losses.append(val_loss)

    print(f"E{epoch} | train loss: {train_loss:.3f} | val loss: {val_loss:.3f}")

  return train_losses, val_losses


def run_model(train_dl, val_dl, model, device,
              lr=0.01, epochs=50,
              lossf=None, opt=None
              ):
  '''
  Given train and val DataLoaders and a NN model, fit the mode to the training
  data. By default, use MSE loss and an SGD optimizer
  '''
  # define optimizer
  if opt:
    optimizer = opt
  else:  # if no opt provided, just use SGD
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

  # define loss function
  if lossf:
    loss_func = lossf
  else:  # if no loss function provided, just use MSE
    loss_func = torch.nn.MSELoss()

  # run the training loop
  train_losses, val_losses = fit(
    epochs,
    model,
    loss_func,
    optimizer,
    train_dl,
    val_dl,
    device)

  return train_losses, val_losses
