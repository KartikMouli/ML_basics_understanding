import time
import numpy as np
a = np.array([1, 2, 3, 4, ])
print(a)

#vectorization
a = np.random.rand(1000000)
b = np.random.rand(1000000)

tic = time.time()
c = np.dot(a,b)
toc = time.time()

print("vect: "+(str((toc-tic)*1000))+"ms")

#avoid using loops whenever possible for saving time on large data
#loop  
c=0
tic=time.time()
for i in range(1000000):
    c+=a[i]*b[i]
toc=time.time()

print("loop: "+(str((toc-tic)*1000))+"ms")