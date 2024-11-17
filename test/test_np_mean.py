import numpy as np

a = np.array([[1,2],[3,4]])
print(a)

mean = np.mean(a)
print(mean)

print(np.mean(a, axis = 0))
print(np.mean(a, axis = 1))
