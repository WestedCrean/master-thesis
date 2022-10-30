import torch
import tensorflow as tf

print("CHECKING FOR PYTORCH")
if torch.cuda.is_available():
    print("There are %d GPU(s) available." % torch.cuda.device_count())
    print("We will use the GPU:", torch.cuda.get_device_name(0))

else:
    print("No GPU available, using the CPU instead.")

print("\n\nCHECKING FOR TENSORFLOW")
print(tf.config.list_physical_devices("GPU"))
