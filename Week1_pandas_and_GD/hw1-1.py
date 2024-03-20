#2018112571 김수성
import numpy as np

def L1_norm(vector):
    return sum([element for element in vector])

def L2_norm(vector):
    return sum([element * element for element in vector])


row_vector = np.random.randint(0, 100, 100)

print(row_vector)

L1 = L1_norm(row_vector)
print("my L1 norm: %.2f" % L1)
L1 = np.linalg.norm(row_vector, 1)
print("numoy L1 norm: %.2f" % L1)


L2 = L2_norm(row_vector)
print("")
print("my L2 norm: %.2f" % L2)
L1 = np.linalg.norm(row_vector, 2)
print("numpy L2 noem: %.2f" % L2)
