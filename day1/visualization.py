%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import pandas as pd

# Plot the loss
log_df = pd.read_csv('gans_logs/loss.log', sep=',')

ax = log_df['d_error'].rolling(200).mean().plot(legend=True)
log_df['g_error'].rolling(200).mean().plot(ax=ax, legend=True)
ax.set_ylabel('loss')
ax.set_xlabel('update iterations')

# Plot a few images sampled from the generator
fig, axes = plt.subplots(figsize=(8,8), nrows=5, ncols=16)

for i, n in enumerate([10, 50, 100, 150, 200]):
    fn = 'gans_logs/generated_img.epoch{}.npy'.format(n)

    test_images = np.load(fn)
    test_images = test_images.squeeze(1)

    for idx in range(16):
        axes[i, idx].imshow(test_images[idx], cmap='gray')
        axes[i, idx].axis('off')
