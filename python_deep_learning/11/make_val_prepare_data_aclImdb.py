from tensorflow import keras
import os, pathlib, shutil, random
from keras import layers


base_dir = pathlib.Path("/home/binbin/dl/python_deep_learning/aclImdb")
val_dir = base_dir / "val"
train_dir = base_dir / "train"

for category in ("neg", "pos"):
    if  not os.path.isdir(val_dir) :
        os.makedirs(val_dir / category)  
        files = os.listdir(train_dir / category)
        random.Random(1337).shuffle(files) 
        num_val_samples = int(0.2 * len(files))
        val_files = files[-num_val_samples:]
        for fname in val_files:
            shutil.move(train_dir / category / fname,
                        val_dir / category / fname)

batch_size = 32
# tf.data
train_ds = keras.utils.text_dataset_from_directory(
    "/home/binbin/dl/python_deep_learning/aclImdb/train", batch_size=batch_size
)
val_ds = keras.utils.text_dataset_from_directory(
    "/home/binbin/dl/python_deep_learning/aclImdb/val", batch_size=batch_size
)
test_ds = keras.utils.text_dataset_from_directory(
    "/home/binbin/dl/python_deep_learning/aclImdb/test", batch_size=batch_size
)

print("train_ds")
for inputs, targets in train_ds:
    print("input shape:", inputs.shape)
    print("input shape:", inputs.dtype)
    print("targets shape:", targets.shape)
    print("targets shape:", targets.dtype)
    print("input[0]:", targets.shape)
    print("targets[0]:", targets.dtype)
    break

print("val_ds")
for inputs, targets in val_ds:
    print("input shape:", inputs.shape)
    print("input shape:", inputs.dtype)
    print("targets shape:", targets.shape)
    print("targets shape:", targets.dtype)
    print("input[0]:", targets.shape)
    print("targets[0]:", targets.dtype)
    break

print("test_ds")
for inputs, targets in test_ds:
    print("input shape:", inputs.shape)
    print("input shape:", inputs.dtype)
    print("targets shape:", targets.shape)
    print("targets shape:", targets.dtype)
    print("input[0]:", targets.shape)
    print("targets[0]:", targets.dtype)
    break