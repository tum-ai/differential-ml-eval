import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# import csv from train_error_files
df = pd.read_csv('train_error_files/train_error_dml.csv')
print(df.head())
values = df['dml_loss'] / df['ml_loss']
# create log scale plot
plt.yscale('log')
plt.xlabel('Epochs')
plt.ylabel('DML Loss / ML Loss')
# set x bounds
plt.xlim(0, 1000)
plt.title('DML Loss / ML Loss over Epochs (log scale)')
# add background grid
plt.plot(values)
# set resolution
plt.savefig('plots/dml_loss_over_epochs.png', dpi=1000)
plt.show()
# save plot
