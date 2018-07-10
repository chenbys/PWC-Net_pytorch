import torch
import torch.nn.functional as F
import imageio
import matplotlib.pyplot as plt
import numpy as np

x = imageio.imread('example/006.jpg')
x = np.array(x)
x = x[np.newaxis, :, :, :].transpose(0, 3, 1, 2)
x = torch.Tensor(x)
x2 = F.upsample(x, scale_factor=2, mode='bilinear')

x2s = np.transpose(x2[0], (1, 2, 0))
plt.imshow(x2s)
plt.show()
a = 1
