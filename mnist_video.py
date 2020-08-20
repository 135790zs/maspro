from torchvision.datasets import MNIST
from matplotlib.widgets import Slider
import numpy as np

mnist = MNIST(root='./data', train=True, download=True, transform=None)
nframes = 20
img = mnist[1][0]
img.show()
img = np.asarray(img)
print(img)

vid = np.zeros(shape=(nframes, img.shape[0], img.shape[1]))
print(vid.shape)
