from tensorflow import keras
import os, pathlib, shutil, random
from keras import layers

batch_size = 32
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
        
train_ds = keras.utils.text_dataset_from_directory(
    "/home/binbin/dl/python_deep_learning/aclImdb/train", batch_size=batch_size
)
val_ds = keras.utils.text_dataset_from_directory(
    "/home/binbin/dl/python_deep_learning/aclImdb/val", batch_size=batch_size
)
test_ds = keras.utils.text_dataset_from_directory(
    "/home/binbin/dl/python_deep_learning/aclImdb/test", batch_size=batch_size
)

max_length = 600
max_tokens = 20000
text_vectorization = layers.TextVectorization(
    max_tokens=max_tokens,
    output_mode="int",
    output_sequence_length=max_length,
)

text_only_train_ds = train_ds.map(lambda x, y: x)
text_vectorization.adapt(text_only_train_ds)

int_train_ds = train_ds.map(
    lambda x, y: (text_vectorization(x), y),
    num_parallel_calls=4)
int_val_ds = val_ds.map(
    lambda x, y: (text_vectorization(x), y),
    num_parallel_calls=4)
int_test_ds = test_ds.map(
    lambda x, y: (text_vectorization(x), y),
    num_parallel_calls=4)
