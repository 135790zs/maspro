import numpy as np
import os
import csv
import pandas as pd
from scipy.interpolate import griddata
import matplotlib.pyplot as plt

varnames = ["Repeats", "traub_trick", "V-Error"]

def process_categoricals(dat):
    catdict = {}
    for cat in dat:
        if cat not in catdict:
            catdict[cat] = len(catdict)
    dat = [catdict[k] for k in dat]

    ticks = [''] * len(catdict)
    for k, v in catdict.items():
        ticks[v] = k
    return dat, len(ticks), ticks

def plot_df(df, fname):
    try:
        x = df[varnames[0]].to_numpy()
        y = df[varnames[1]].to_numpy()
        z = df["V-Error"].to_numpy()
    except KeyError:
        return

    if type(x[0]).__name__ in ['str', 'int64', 'bool_']:
        x, nx, xticks = process_categoricals(dat=x)
        plt.xticks(np.arange(0, nx), xticks)
    else:
        nx = 100

    if type(y[0]).__name__ in ['str', 'int64', 'bool_']:
        y, ny, yticks = process_categoricals(dat=y)
        plt.yticks(np.arange(0, ny), yticks)
    else:
        ny = 100

    grid_x, grid_y = np.mgrid[np.min(x):np.max(x):complex(nx, 0),
                              np.min(y):np.max(y):complex(ny, 0)]

    grid_z0 = griddata(np.stack((x, y),
                                axis=1),
                       z,
                       (grid_x, grid_y),
                       method='linear')

    fig = plt.figure()
    ax = fig.add_subplot(111)

    plt.xlabel(varnames[0])
    plt.ylabel(varnames[1])
    plt.set_cmap('cool')

    plt.scatter(x=x, y=y, c='black')
    plt.imshow(grid_z0.T,
               extent=(np.min(x),np.max(x),np.min(y),np.max(y)),
               origin='lower',
               interpolation='none')
    ax.set_aspect('auto', adjustable='box')
    plt.colorbar()

    plt.savefig(f"../sweeps/{fname}-{varnames[0]}-{varnames[1]}.pdf",
                bbox_inches='tight')
    plt.close()


for subdir, _, files in os.walk(f"../sweeps/"):
    for filename in files:
        if not filename.endswith('.csv'):
            continue
        filepath = subdir + os.sep + filename
        df = pd.read_csv(filepath)

        plot_df(df, fname=filename)
