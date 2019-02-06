import numpy as np
import matplotlib.pyplot as plt
import os 
from scipy.signal import savgol_filter


#data step size
x = np.arange(0,100000,50)

#cropping window size
window_size = 22

#load data
file_id = os.path.join('./plot_data_{}.npy').format(window_size)

data = np.load(file_id).item()

#data processing
train_acc = data['train_acc']
train_acc = savgol_filter(train_acc, 51, 3)

train_adv_acc = data['train_adv_acc']
train_adv_acc = savgol_filter(train_adv_acc, 51, 3)

test_acc = data['test_acc']
test_acc = savgol_filter(test_acc, 51, 3)

test_adv_acc = data['test_adv_acc']
test_adv_acc = savgol_filter(test_adv_acc, 51, 3)

#plot
fig1 = plt.figure(1)
plt.plot(x,train_acc)

fig2 = plt.figure(2)
plt.plot(x,train_adv_acc)

fig3 = plt.figure(3)
plt.plot(x,test_acc)

fig4 = plt.figure(4)
plt.plot(x,test_adv_acc)

#set label and title
fig1.suptitle('training clean accuracy', fontsize=18)
fig2.suptitle('training adversarial accuracy', fontsize=18)
fig3.suptitle('testing clean accuracy', fontsize=18)
fig4.suptitle('training adversarial accuracy', fontsize=18)

plt.xlabel('iteration', fontsize=18)
plt.ylabel('accuracy', fontsize=16)

# saving
file_id1 = os.path.join('./image/train_acc_{}.png').format(window_size)
file_id2 = os.path.join('./image/train_adv_acc_{}.png').format(window_size)
file_id3 = os.path.join('./image/test_acc_{}.png').format(window_size)
file_id4 = os.path.join('./image/test_adv_acc_{}.png').format(window_size)


fig1.savefig(file_id1)
fig2.savefig(file_id2)
fig3.savefig(file_id3)
fig4.savefig(file_id4)


