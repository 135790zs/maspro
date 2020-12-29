import matplotlib
import matplotlib.pyplot as plt
import numpy as np
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

a0 = 0.53836
a1 = 0.46164
res = 400
N = 200

x = np.linspace(0, N, res)

y = a0 - a1 * np.cos((2*np.pi*x)/N)
plt.figure(figsize=(5, 5))
plt.plot(y, color='black', linewidth=2)
plt.grid()
plt.xlabel("Samples", fontsize=20)
plt.ylabel("Amplitude", fontsize=20)
plt.title("Hamming window", fontsize=20)
plt.savefig("../vis/hamming.pdf")
plt.clf()
