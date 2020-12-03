import os
from itertools import combinations
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def process_categoricals(dat):
    catdict = {}
    for cat in dat:
        if cat not in catdict:
            catdict[cat] = len(catdict)
    dat = [catdict[k] for k in dat]

    ticks = list(catdict.keys())
    return dat, len(ticks), ticks


def plot_dfs(df, fname):
    print(list(df))
    varnames = list(df)
    varnames.remove("V-Error")
    pool = list(combinations(varnames, r=2))

    size = int(np.ceil(np.sqrt(len(pool))))
    fig, axs = plt.subplots(nrows=size, ncols=size, figsize=(3*size, 3*size))

    plt.set_cmap('cool')

    def jitter(arr, strength=0.02):
        stdev = strength * (max(arr) - min(arr))
        return arr + np.random.randn(len(arr)) * stdev

    for idx, comb in enumerate(pool):
        x = df[comb[0]].to_numpy()
        y = df[comb[1]].to_numpy()
        z = df["V-Error"].to_numpy()

        if not os.path.exists(f"../sweeps/{fname}"):
            os.makedirs(f"../sweeps/{fname}")

        if type(x[0]).__name__ in ['str', 'bool_']:
            x, nx, xticks = process_categoricals(dat=x)
            axs[idx // size, idx % size].set_xticks(np.arange(0, nx))
            axs[idx // size, idx % size].set_xticklabels(xticks)
            extentx = (min(x)-.5, max(x)+.5)
            add_x_jiiter = True
        elif type(x[0]).__name__ == 'int64':
            xticks = np.arange(np.min(x), np.max(x)+1)
            extentx = (min(x)-.5, max(x)+.5)
            add_x_jiiter = True
            nx = len(xticks)
            if len(xticks) <= 20:
                axs[idx // size, idx % size].set_xticks(np.min(x)
                                                        + np.arange(0, nx))
                axs[idx // size, idx % size].set_xticklabels(xticks)
        else:  # Float
            extentx = (min(x), max(x))
            add_x_jiiter = False

            nx = 100

        if type(y[0]).__name__ in ['str', 'bool_']:
            y, ny, yticks = process_categoricals(dat=y)
            axs[idx // size, idx % size].set_yticks(np.arange(0, ny))
            axs[idx // size, idx % size].set_yticklabels(yticks)
            extenty = (min(y)-.5, max(y)+.5)
            add_y_jiiter = True
        elif type(y[0]).__name__ == 'int64':
            yticks = np.arange(np.min(y), np.max(y)+1)
            extenty = (min(y)-.5, max(y)+.5)
            add_y_jiiter = True
            ny = len(yticks)
            if len(yticks) <= 20:
                axs[idx // size, idx % size].set_yticks(np.min(y)
                                                        + np.arange(0, ny))
                axs[idx // size, idx % size].set_yticklabels(yticks)
        else:  # Float
            extenty = (min(y), max(y))
            add_y_jiiter = False
            ny = 100

        grid_x, grid_y = np.mgrid[np.min(x):np.max(x):complex(nx, 0),
                                  np.min(y):np.max(y):complex(ny, 0)]

        grid_z0 = griddata(np.stack((x, y),
                                    axis=1),
                           z,
                           (grid_x, grid_y),
                           method='linear')

        axs[idx // size, idx % size].set_xlabel(comb[0])
        axs[idx // size, idx % size].set_ylabel(comb[1])
        axs[idx // size, idx % size].set_title(f"{comb[1]} vs {comb[0]}")

        axs[idx // size, idx % size].scatter(x=jitter(x) if add_x_jiiter else x,
                                             y=jitter(y) if add_y_jiiter else y,
                                             c='black', marker='+', alpha=0.35)
        im = axs[idx // size, idx % size].imshow(
            grid_z0.T,
            extent=(extentx[0], extentx[1], extenty[0], extenty[1]),
            origin='lower',
            interpolation='none',
            vmin=np.min(z),
            vmax=np.max(z),)
        axs[idx // size, idx % size].set_aspect('auto')

    # Remove empty axes
    for idx in range(len(pool), size**2):
        axs[idx // size, idx % size].axis('off')

    plt.tight_layout()
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)

    plt.savefig(f"../sweeps/{fname}/all.pdf",
                bbox_inches='tight')
    plt.close()


def plot_regressions(df, fname):
    valid_varnames = [n for n in df]
    valid_varnames.remove("V-Error")

    size = int(np.ceil(np.sqrt(len(valid_varnames))))
    _, axs = plt.subplots(nrows=size, ncols=size, figsize=(3*size, 3*size))

    for idx, var in enumerate(valid_varnames):

        x = df[var].to_numpy()
        y = df["V-Error"].to_numpy()

        if type(x[0]).__name__ in ['str', 'bool_']:
            x, nx, xticks = process_categoricals(dat=x)
            axs[idx // size, idx % size].set_xticks(np.arange(0, nx))
            axs[idx // size, idx % size].set_xticklabels(xticks)
            axs[idx // size, idx % size].set_xlim(np.min(x)-0.3, np.max(x)+0.3)
        elif type(x[0]).__name__ == 'int64':
            xticks = np.arange(np.min(x), np.max(x)+1)
            nx = len(xticks)
            if len(xticks) <= 20:
                axs[idx // size, idx % size].set_xticks(np.min(x)
                                                        + np.arange(0, nx))
                axs[idx // size, idx % size].set_xticklabels(xticks)
                axs[idx // size, idx % size].set_xlim(np.min(x) - 0.3,
                                                      np.max(x) + 0.3)
        else:  # Float
            nx = 100

        if idx % size:
            axs[idx // size, idx % size].get_yaxis().set_visible(False)
        else:
            axs[idx // size, idx % size].set_ylabel("V-Error")
        sns.regplot(ax=axs[idx // size, idx % size],
                    x=x,
                    y=y,
                    marker='+',
                    order=1,
                    color='black')
        axs[idx // size, idx % size].set_xlabel(var, fontsize=14)
        # axs[idx // size, idx % size].set_title(var)

    # Remove empty axes
    for idx in range(len(valid_varnames), size**2):
        axs[idx // size, idx % size].axis('off')

    plt.suptitle("Regression analysis of Bayesian HPO", fontsize=2*size**2)
    plt.subplots_adjust(wspace=0, hspace=0.4)

    plt.savefig(f"../sweeps/{fname}/scatters.pdf",
                bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    for subdir, _, files in os.walk(f"../sweeps/"):
        for filename in files:
            if not filename.endswith('.csv'):
                continue
            filepath = subdir + os.sep + filename
            plot_dfs(df=pd.read_csv(filepath),
                     fname=filename[:-4])

            plot_regressions(df=pd.read_csv(filepath),
                             fname=filename[:-4])
