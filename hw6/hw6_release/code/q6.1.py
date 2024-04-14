import numpy as np

# q1: Trace of a 5x5 matrix
A = np.random.rand(5, 5)
trace_A_regular = np.trace(A)
trace_A_einsum = np.einsum('ii', A)
diff_trace = np.linalg.norm(trace_A_regular - trace_A_einsum)

# q2: Matrix product of two 5x5 matrices
B = np.random.rand(5, 5)
product_AB_regular = np.dot(A, B)
product_AB_einsum = np.einsum('ij,jk->ik', A, B)
diff_product_AB = np.linalg.norm(product_AB_regular - product_AB_einsum)

# q3: Batchwise matrix product of shapes (3, 4, 5) and (3, 5, 6)
C = np.random.rand(3, 4, 5)
D = np.random.rand(3, 5, 6)
batch_product_CD_regular = np.matmul(C, D)
batch_product_CD_einsum = np.einsum('ijk,ikl->ijl', C, D)
diff_batch_product_CD = np.linalg.norm(batch_product_CD_regular - batch_product_CD_einsum)

print(trace_A_regular, trace_A_einsum, diff_trace)
print(product_AB_regular, product_AB_einsum, diff_product_AB)
#print(batch_product_CD_regular, batch_product_CD_einsum, diff_batch_product_CD)
