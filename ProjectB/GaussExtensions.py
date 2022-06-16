# -*- coding: utf-8 -*-
"""
@Project: LinalgDat2022
@File: GaussExtensions.py

@Description: Project B Gauss extensions

"""

import math
import sys
from xml.etree.ElementTree import tostring

sys.path.append('../Core')
from Vector import Vector
from Matrix import Matrix


def AugmentRight(A: Matrix, v: Vector) -> Matrix:
    """
    Create an augmented matrix from a matrix and a vector.

    This function creates a new matrix by concatenating matrix A
    and vector v. See page 12 in "Linear Algebra for Engineers and
    Scientists", K. Hardy.

    Parameters:
         A: M-by-N Matrix
         v: M-size Vector
    Returns:
        the M-by-(N+1) matrix (A|v)
    """
    M = A.M_Rows
    N = A.N_Cols
    if v.size() != M:
        raise ValueError("number of rows of A and length of v differ.")

    B = Matrix(M, N + 1)
    for i in range(M):
        for j in range(N):
            B[i, j] = A[i, j]
        B[i, N] = v[i]
    return B


def ElementaryRowReplacement(A: Matrix, i: int, m: float, j: int) -> Matrix:
    """
    Replace row i of A by row i of A + m times row j of A.

    Parameters:
        A: M-by-N Matrix
        i: int, index of the row to be replaced
        m: float, the multiple of row j to be added to row i
        j: int, index or replacing row.

    Returns:
        A modified in-place after row replacement.
    """
    N = A.N_Cols
    for c in range(N):
        A[i,c] = A[i,c] + m*(A[j,c])
    return A



def ElementaryRowInterchange(A: Matrix, i: int, j : int) -> Matrix:
    """
    Interchange row i and row j of A.

    Parameters:
        A: M-by-N Matrix
        i: int, index of the first row to be interchanged
        j: int, index the second row to be interchanged.

    Returns:
        A modified in-place after row interchange
    """
    N = A.N_Cols
    for c in range(N):
        temp = A[i,c]
        A[i,c] = A[j,c]
        A[j,c] = temp
    return A




def ElementaryRowScaling(A: Matrix, i: int, c: float) -> Matrix:
    """
    Replace row i of A c * row i of A.

    Parameters:
        A: M-by-N Matrix
        i: int, index of the row to be replaced
        c: float, the scaling factor

    Returns:
        A modified in-place after row scaling.
    """
    N = A.N_Cols
    for k in range(N):
        A[i,k] = A[i,k]*c
    return A


def ForwardReduction(A: Matrix) -> Matrix:
    """
    Forward reduction of matrix A.

    This function performs the forward reduction of A provided in the
    assignment text to achieve row echelon form of a given (augmented)
    matrix.

    Parameters:
        A:  M-by-N augmented matrix
    returns
        M-by-N matrix which is the row-echelon form of A (performed in-place,
        i.e., A is modified directly).
    """
    M = A.M_Rows
    N = A.N_Cols
    pr = 0 #pivot row
    pc = 0 #pivot col
    def step_1_2(A,r, c): #makes step one and two
        b = False
        pc = c
        pr_before = r
        if not(b):
            for j in range(c,N):
                for i in range(r,M):
                    if A[i,j] != 0:
                        pc = j
                        pr_before = i
                        b = True
                        break
                if b:
                    break
                pc +=1
        A = ElementaryRowInterchange(A, pr_before, r)
        pr = r
        for i in range(r+1,M):
            if A[i,pc] != 0:
                A = ElementaryRowReplacement(A, i, -((A[i,pc])/A[pr,pc]),pr)
        return A
    
    #step3
    for c in range(N):
        A = step_1_2(A, pr, pc)
        if pr != M-1 and pc != N-1:
            pr+=1
            pc+=1
    return A




def BackwardReduction(A: Matrix) -> Matrix:
    """
    Backward reduction of matrix A.

    This function performs the forward reduction of A provided in the
    assignment text given a matrix A in row echelon form.

    Parameters:
        A:  M-by-N augmented matrix in row-echelon form
    returns
        M-by-N matrix which is the reduced form of A (performed in-place,
        i.e., A is modified directly).
    """

    M = A.M_Rows
    N = A.N_Cols
    pr,pc = M-1,0
    empty = 0
    def step1 (A,pr, empty):
        found = False
        pivot = 1
        for i in range(pr)[::-1]:
            found_now = False
            if not(found):
                for j in range(N):
                    if abs(A[i,j]) > 10e-08:
                        found = True
                        found_now = True
                        pivot = A[i,j]
                        pr,pc = i,j
                        A = ElementaryRowScaling(A, i,1/pivot)
                        break
                if not(found):
                    empty += 1

                
            if found and not(found_now):
                A = ElementaryRowReplacement(A,i,-A[i,pc], pr)
        return A
    
    for r in range(M+1)[::-1]:
        A = step1(A,r,empty)
    return A
            

        






def GaussElimination(A: Matrix, v: Vector) -> Vector:
    """
    Perform Gauss elimination to solve for Ax = v.

    This function performs Gauss elimination on a linear system given
    in matrix form by a coefficient matrix and a right-hand-side vector.
    It is assumed that the corresponding linear system is consistent and
    has exactly one solution.

    Hint: combine AugmentRight, ForwardReduction and BackwardReduction!

    Parameters:
         A: M-by_N coefficient matrix of the system
         v: N-size vector v, right-hand-side of the system.
    Return:
         M-size solution vector of the system.
    """
    Aug = AugmentRight(A,v)
    reduced = BackwardReduction(ForwardReduction(Aug))
    M = A.M_Rows
    N = A.N_Cols
    vec = Vector(M)
    for i in range(M):
        vec[i] = reduced[i,N]
    return vec

