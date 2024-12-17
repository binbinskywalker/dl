import re
import string
import tensorflow as tf
from keras.layers import TextVectorization


def custom_standardization_fn(string_tensor):
    lowercase_string = tf.strings.lower(string_tensor)  ←----将字符串转换为小写字母
    return tf.strings.regex_replace(  ←----将标点符号替换为空字符串
        lowercase_string, f"[{re.escape(string.punctuation)}]", "")

def custom_split_fn(string_tensor):
    return tf.strings.split(string_tensor)  ←----利用空格对字符串进行拆分

text_vectorization = TextVectorization(
    output_mode="int",
    standardize=custom_standardization_fn,
    split=custom_split_fn,
)