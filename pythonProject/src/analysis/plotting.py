from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from joypy import joyplot
from math import prod
from matplotlib import cm


def ridge_plot(sim_data: Path, fig_outpath: Path) -> None:
    """
    Create a ridge plot representing the state of the attributes of the
    simulation.
    :param sim_data:
    :param fig_outpath:
    :return:
    """
    df = pd.read_csv(sim_data, index_col='epoch')
    df = df.drop(columns=[df.columns[-1]])

    attrs = df.columns
    num_epochs = len(df.columns)
    now_classes = len(df.index)
    vals = np.array(df).T

    means = vals.max(axis=1).reshape((13, 1))

    norm_vals = vals / (means + 0.000000001) / 1.5

    new_df = pd.DataFrame({
        'epoch': np.tile(np.arange(now_classes), num_epochs),
        'cls': np.repeat(attrs, now_classes),
        'val': norm_vals.reshape((prod(df.shape)))
    })

    fig_width = 500 / num_epochs

    joyplot(new_df, by='cls',
            column='val',
            kind='values',
            x_range=np.arange(100),
            figsize=(fig_width, 10),
            colormap=cm.get_cmap('terrain'))

    plt.savefig(fig_outpath)
