import numpy as np
from numpy.linalg import *
import sympy as sp

X,Y,Z = sp.symbols(['x','y','z'])

# A
# Defining the vectors used in the question
A = np.array([1,2,3]).reshape(-1,1)
B = np.array([4,5,6]).reshape(-1,1)
C = np.array([-1,1,3]).reshape(-1,1)
# Defining the x,y,z axis
x = np.array([1,0,0]).reshape(-1,1)
y = np.array([0,1,0]).reshape(-1,1)
z = np.array([0,0,1]).reshape(-1,1)

## 1
print(2*A-B)

## 2
print(norm(A))
print(np.arccos((sum(A*x)/norm(A))))

## 3
print(A/norm(A))

## 4
print([x for x in A/norm(A)])

## 5
print(np.dot(A.transpose(),B))
print(np.dot(B.transpose(),A))

## 6
print(np.arccos(np.dot(A.transpose(),B)/(norm(A)*norm(B))))

## 7
print(np.array([-A[1],A[0],0]).reshape(-1,1))

## 8
print(np.cross(A.ravel(),B.ravel()))
print(np.cross(B.ravel(),A.ravel()))

## 9 
print(np.cross(A.ravel(),B.ravel()))

## 10


## 11
print(np.matmul(A.transpose(),B))
print(np.matmul(A,B.transpose()))

# B

A = np.array([[1,2,3],[4,-2,3],[0,5,-1]])
B = np.array([[1,2,1],[2,1,-4],[3,-2,1]])
C = np.array([[1,2,3],[4,5,6],[-1,1,3]])

## 1
print(2*A-B)

## 2
print(np.matmul(A,B))
print(np.matmul(B,A))

## 3
print(np.matmul(A,B).transpose())
print(np.matmul(B.transpose(),A.transpose()))

## 4
print(linalg.det(A))
print("%.2f"%linalg.det(C))

## 5

## 6
print(linalg.inv(A))
print(linalg.inv(B))

# C
A = np.array([[1,2],[3,2]])
B = np.array([[2,-2],[-2,5]])

## 1
w,v = linalg.eig(A)
print(w)
print(v)

## 2
print(np.dot(np.dot(linalg.inv(v),A),v))

## 3
print(np.dot(v.transpose()[0],v.transpose()[1]))

## 4

w,v = linalg.eig(B)
print(np.dot(v.transpose()[0],v.transpose()[1]))


## D
f = X**2 + 3
g = X**2 + Y**2

## 1
print(sp.diff(f,X))
print(sp.diff(f,X,2))

## 2
print(sp.diff(g,X))
print(sp.diff(g,Y))

## 3
print(sp.Matrix(((sp.diff(g,X)),(sp.diff(g,Y)))))
