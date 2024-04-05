import altair as alt
import pandas as pd
from matplotlib import pyplot as plt
from rich.jupyter import display
from sklearn.metrics import r2_score
import torch

def parity_plot(model_name, df, r2):
  '''
  Given a dataframe of samples with their true and predicted values,
  make a scatterplot.
  '''
  plt.scatter(df['truth'].values, df['pred'].values, alpha=0.2)
  # y=x line
  xpoints = ypoints = plt.xlim()
  plt.plot(xpoints, ypoints, linestyle='--', color='k', lw=2, scalex=False, scaley=False)
  plt.ylim(xpoints)
  plt.ylabel("Predicted Score", fontsize=14)
  plt.xlabel("Actual Score", fontsize=14)
  plt.title(f"{model_name} (r2:{r2:.3f})", fontsize=20)
  plt.show()
  pass


def alt_parity_plot(model, df, r2):
  '''
  Make an interactive parity plot with altair
  '''
  chart = alt.Chart(df).mark_circle(size=100, opacity=0.4).encode(
    alt.X('truth:Q'),
    alt.Y('pred:Q'),
    tooltip=['seq:N']
  ).properties(
    title=f'{model} (r2:{r2:.3f})'
  ).interactive()

  chart.save(f'alt_out/parity_plot_{model}.html')
  display(chart)

