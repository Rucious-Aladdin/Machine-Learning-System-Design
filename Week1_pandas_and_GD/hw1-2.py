#2018112571 김수성
import numpy as np

def my_Hardmard_product(A, B):
    k1, k2 = A.shape
    
    answer = np.zeros_like(A)

    for i in range(k1):
        for j in range(k2):
            answer[i][j] = A[i][j] * B[i][j]
            
    return answer

k = int(input("k입력: "))    
A = np.random.randint(1, 10, size = (k, k))
B = np.random.randint(1, 10, size = (k, k))

numpy_hardmard = A * B
my_hardmard = my_Hardmard_product(A, B)
print("A: ")
print(A)
print("B: ")
print(B)
print("numpy hardmard:")
print(numpy_hardmard)
print("my hardmard:")
print(my_hardmard)