from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
import tensorflow as tf
print_tensors_in_checkpoint_file(file_name = './checkpoint/noatt_ckpt', tensor_name='', all_tensors=False)
#print_tensors_in_checkpoint_file(file_name='/home/doi6/Documents/Guangyu/test1/lenet.ckpt', tensor_name='', all_tensors=False)

