

text_file = "/home/binbin/dl/python_deep_learning/spa-eng/spa.txt"
with open(text_file) as f:
    lines = f.read().split("\n")[:-1]
text_pairs = []
for line in lines:
    english, spanish = line.split("\t") 
    spanish = "[start] " + spanish + " [end]"
    text_pairs.append((english, spanish))