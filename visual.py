import numpy as np

adv = np.load('./adv_8.npy').item()

adv_image = adv['image'][5]

print(adv_image.shape)

import matplotlib.pyplot as plt

plt.imshow(adv_image/255)
plt.show()


