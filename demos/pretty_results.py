import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from matplotlib import rcParams as rc

rc['mathtext.fontset'] = 'stix'
rc['font.family'] = 'STIXGeneral'
sns.set(context='paper', style='whitegrid', rc={"ytick.left" : True})

NREC = 1
COMPARE_NEURONS = False
figsize = (2.5, 2.5)

testscores = {
	'ALIF': {1: (58.4, 2.233), 2: (64.5, 2.558), 3: (74.1, 3.345)},
	'STDP-ALIF': {1: (48.3, 1.732), 2: (65.3, 2.643), 3: (88.3, 4.779)},
	'Izhikevich': {1: (93.5, 8.863), 2: (88.2, 4.259), 3: (88.5, 4.161)}
}

tf_conv = {  # title
	'Error (reg).csv': '$E_{{reg}}$',
	'Percentage wrong.csv': 'Percentage misclassified',
	'Cross-entropy.csv': 'Cross-entropy error',
	'Mean Hz.csv': 'Mean frequency (Hz)',
}
tf_conv_fname = {  #fname
	'Error (reg).csv': 'regerr',
	'Percentage wrong.csv': 'percwrong',
	'Cross-entropy.csv': 'crossentropy',
	'Mean Hz.csv': 'hz',
}

types = {'ALIF', 'STDP-ALIF', 'Izhikevich'}
d = {}

if COMPARE_NEURONS:

	colors = ['mediumturquoise', 'goldenrod', 'maroon']

	# Read all data for selected vault dirs (indicated by BEST)
	for k in types:
		fpath = f"../vault/{k} 1 layer/"
		dirs = os.listdir(fpath)
		dirs = [vdir for vdir in dirs if vdir.startswith('BEST')]
		fpath = fpath + dirs[0] + '/tracker/'
		trackerfiles = os.listdir(fpath)
		for tf in trackerfiles:
			if tf not in d:
				d[tf] = {}
			d[tf][k] = pd.read_csv(fpath + tf)
	axidx = 0

	for tf in d.keys():
		fig, ax = plt.subplots(figsize=figsize)
		ax.set(title=tf_conv[tf])
		for k_idx, k in enumerate(types):
			raw = d[tf][k]['val'][d[tf][k]['val'] != -1]
			smooth = raw.rolling(window=12).mean()

			if tf_conv[tf] in ["Mean frequency (Hz)"]:
				sns.lineplot(ax=ax, data=raw, alpha=1, label=k, color=colors[k_idx])
			else:
				sns.lineplot(ax=ax, data=raw, alpha=0.2, color=colors[k_idx])
				sns.lineplot(ax=ax, data=smooth, label=k, color=colors[k_idx])

			# Plot test star
			if tf_conv[tf] == 'Cross-entropy error':
				plt.scatter(x=smooth.idxmin(), y=testscores[k][1][1], color=colors[k_idx], marker='*')
			elif tf_conv[tf] == 'Percentage misclassified':
				plt.scatter(x=smooth.idxmin(), y=testscores[k][1][0], color=colors[k_idx], marker='*')

		if tf_conv[tf] in ['Cross-entropy error', '$E_{{reg}}$']:
			ax.set(yscale='log')
		if tf_conv[tf] in ["Mean frequency (Hz)"]:
			ax.set(xscale='log')
		if tf_conv[tf] == 'Cross-entropy error':
			ax.set(ylim=(1, None))


		ax.set(xlabel="Iterations")
		ax.set(ylabel=tf_conv[tf])
		plt.savefig(f"../texfiles/thesis/gfx/{tf_conv_fname[tf]}.pdf",
	                bbox_inches='tight')
		print(f"Saved {tf_conv[tf]}")
		plt.close()
else:

	nlayers = 3
	colors = ['green', 'darkorange', 'darkblue']

	# Read all data for selected vault dirs (indicated by BEST)
	for k in types:
		for nlayer in range(1, nlayers+1):
			fpath = f"../vault/{k} {nlayer} layer/"
			dirs = os.listdir(fpath)
			dirs = [vdir for vdir in dirs if vdir.startswith('BEST')]
			fpath = fpath + dirs[0] + '/tracker/'
			trackerfiles = os.listdir(fpath)
			for tf in trackerfiles:
				if tf not in d:
					d[tf] = {t: {} for t in types}

				d[tf][k][nlayer] = pd.read_csv(fpath + tf)
	axidx = 0

	for tf in d.keys():
		for k_idx, k in enumerate(types):
			fig, ax = plt.subplots(figsize=figsize)
			ax.set(title=f"{tf_conv[tf]} ({k})")
			for nlayer in range(1, nlayers+1):
				raw = d[tf][k][nlayer]['val'][d[tf][k][nlayer]['val'] != -1]
				smooth = raw.rolling(window=12).mean()
				sns.lineplot(ax=ax, data=raw, alpha=0.2, color=colors[nlayer-1])
				sns.lineplot(ax=ax, data=smooth, label=f"{nlayer} layers", color=colors[nlayer-1])
				# Plot test star
				if tf_conv[tf] == 'Cross-entropy error':
					plt.scatter(x=smooth.idxmin(), y=testscores[k][nlayer][1], color=colors[nlayer-1], marker='*')
				elif tf_conv[tf] == 'Percentage misclassified':
					plt.scatter(x=smooth.idxmin(), y=testscores[k][nlayer][0], color=colors[nlayer-1], marker='*')
			if tf_conv[tf] in ['Cross-entropy error', '$E_{{reg}}$']:
				ax.set(yscale='log')
			if tf_conv[tf] == 'Cross-entropy error':
				ax.set(ylim=(1, None))
			ax.set(xlabel="Iterations")
			ax.set(ylabel=tf_conv[tf])
			plt.savefig(f"../texfiles/thesis/gfx/ml-{tf_conv_fname[tf]}-{k}.pdf",
		                bbox_inches='tight')
			print(f"Saved {tf_conv[tf]}-{k}")
			plt.close()
