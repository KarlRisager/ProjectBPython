"""
@project: LinalgDat 2022
@file: Matrix.py

@description: A class which implements some simple matrix structure
with not much more, as more complicated parts are the assignments the
students must hand in. Row major mode.

Note that I use type hints when convenient and for readability.

@author: François Lauze, DIKU.
@date: Mars 2022.
"""

import copy
import math
from Vector import Vector
import numbers


class MatrixException(Exception):
    """Exception class for matrix operations"""
    pass


class Matrix:
    """
    A simple matrix class with float entries.

    The class provides ability to create a new matrix, read from
    an existing one, from a 2D array, and export to a 2D array.
    In also includes a deep copy.
    """

    # used for string conversion / printing
    openDelimiter = '['
    closeDelimiter = ']'
    separator = ','
    offset = 0
    precision = 5

    def __init__(self, m: int, n: int):
        """Create a matrix (m, n) filled with zeros."""
        self.m_ = m
        self.n_ = n
        self.size_ = self.m_ * self.n_
        self.data = [0.0] * self.size_

    def __copy__(self):
        return copy.deepcopy(self)

    def __getitem__(self, index: tuple) -> float:
        """Retrieve entry (i,j)."""
        i, j = index
        offset = i * self.n_ + j
        return self.data[offset]

    def __setitem__(self, index: tuple, value: float) -> None:
        """Fill entry (i,j)."""
        i, j = index
        offset = i * self.n_ + j
        self.data[offset] = value

    @property
    def M_Rows(self) -> int:
        """Return the number of lines."""
        return self.m_

    @property
    def N_Cols(self) -> int:
        """Return the number of columns."""
        return self.n_

    @property
    def Size(self) -> int:
        """Return the number entries."""
        return self.size_

    def asArray(self) -> list:
        """Returns the matrix as a Python 2D array."""
        return [[self.data[i * self.n_ + j] for j in range(self.n_)]
                for i in range(self.m_)]

    def Row(self, i: int) -> Vector:
        """Return row i as row vector."""
        return Vector.fromArray(self.data[i * self.n_: (i + 1) * self.n_], mode='row')

    def Column(self, j: int) -> Vector:
        """Return column j as  column vector."""
        c = Vector(self.m_)
        for i in range(self.m_):
            c.data[i] = self.data[i * self.n_ + j]
        return c

    @staticmethod
    def fromArray(array):
        """
        Create a Matrix from a 2D array.

        array must be a rectangular array: it must possess element access via
        nested brackets i.e. A[i][j], and all A[i] object must have the same length.
        Elements must be of numerical type. Otherwise, an exception is raised.
        """
        N_rows = len(array)
        N_cols = len(array[0])
        for i in range(N_rows):
            if len(array[i]) != N_cols:
                raise MatrixException("Array is not rectangular.")
        M = Matrix(N_rows, N_cols)
        for i in range(N_rows):
            for j in range(N_cols):
                if isinstance(array[i][j], numbers.Number):
                    M[i, j] = array[i][j]
                else:
                    raise MatrixException("Elements of array must be numeric!")
        return M

    @staticmethod
    def IdentityMatrix(n: int):
        """Return a (n, n) identity matrix."""
        I = Matrix(n, n)
        for i in range(n):
            I[i, i] = 1.0
        return I

    @staticmethod
    def Hilbert(n: int):
        """Return a so-called Hilbert matrix (see Wikipedia)."""
        H = Matrix(n, n)
        for i in range(n):
            for j in range(n):
                H[i, j] = 1.0/(i + j + 1.0)
        return H

    def toString(self, openDelimiter=None, closeDelimiter=None, separator=None,
                 offset=None, precision=None) -> str:
        """Convert the matrix to a string."""
        if openDelimiter is None:
            openDelimiter = Matrix.openDelimiter
        if closeDelimiter is None:
            closeDelimiter = Matrix.closeDelimiter
        if separator is None:
            separator = Matrix.separator
        if offset is None:
            offset = Matrix.offset
        if precision is None:
            precision = Matrix.precision

        lsep = separator + " "
        # A couple of helper routines...

        def integerPartWidth(x: float) -> int:
            sign = 1 if x < 0 else 0
            return len(f'{math.floor(x)}') + sign

        def makeLine(i: int, width: int) -> str:
            return lsep.join([f'{self.data[i * self.n_ + j]:{width}.{precision}f}'
                              for j in range(self.n_)])

        def str_repeat(c: str, n: int) -> str:
            return ''.join([c] * n)

        # find the desired width
        int_width = 0
        for i in range(self.size_):
            x_int_width = integerPartWidth(self.data[i])
            if x_int_width > int_width:
                int_width = x_int_width

        # the + 1 is for the decimal point
        width = int_width + 1

        # convert lines
        lines = [makeLine(i, width) for i in range(self.m_)]
        # Add upper-level separators and  offset.

        top_start = str_repeat(" ", offset) + openDelimiter
        line_start = str_repeat(" ", offset + 1)

        A = ""
        for i in range(self.m_):
            first = i == 0
            last = i == self.m_ - 1
            A += top_start if first else line_start
            A += openDelimiter + lines[i] + closeDelimiter
            A += "]\n" if last else ",\n"
        return A

    def __repr__(self) -> str:
        """Convert content to a string representation."""
        return self.toString()
