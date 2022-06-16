"""
@project: LinalgDat 2022
@file: Vector.py

@description: A class which implements some simple vector structure
in a way very similar to the F# and old C# implementations.



@author: François Lauze, DIKU.
@date: Mars 2022.
"""

import math
import copy
import numbers


class VectorException(Exception):
    """Exception class for Vector operations."""
    pass


class Vector:
    """A simple vector class doing not much more than a list."""

    # used for string conversion / printing
    openDelimiter = "["
    closeDelimiter = ']'
    separator = ','
    offset = 0
    precision = 5

    def __init__(self, n: int, mode='column'):
        """
        Initialise a vector.

        If n is an integer, create a zero-filled vector of length n.
        if n is a numerical 1D list, create a vector with same length and
        same content (cast to float).
        """
        self.n_ = n
        self.data = [0.0] * self.n_
        self.mode = mode

    def __len__(self) -> int:
        """Length of the vector."""
        return self.n_

    def size(self) -> int:
        """same as len, for 'compatibility' with Matrix.size()."""
        return self.n_

    def __copy__(self):
        return copy.deepcopy(self)

    def __getitem__(self, i: int) -> float:
        """Get element i of the vector."""
        return self.data[i]

    def __setitem__(self, i: int, value: float) -> None:
        """Set element i of the vector."""
        if not isinstance(value, numbers.Number):
            raise ValueError(f'The value to be assigned must be of numeric type, not {type(value)}.')
        self.data[i] = value

    @property
    def mode(self) -> str:
        """Return the vector mode: row or column."""
        return self.mode_

    @mode.setter
    def mode(self, new_mode: str) -> None:
        """Get the vector mode, 'row' or 'column'."""
        if new_mode not in ('row', 'column'):
            raise TypeError(f'Mode specification {new_mode} invalid. It should be "row" or "column".')
        self.mode_ = new_mode

    def internalMul(self, s):
        """Multiplication by a scalar or a vector as used in the next two routines."""
        if isinstance(s, numbers.Number):
            # scalar multiplication
            y = self.__copy__()
            for i in range(self.n_):
                y[i] *= s
        elif isinstance(s, Vector):
            # Hadamard product
            if (len(s) != len(s)) or (self.mode_ != s.mode_):
                raise ValueError('Vectors with different lengths or different mode cannot be point-wise multiplied.')
            y = self.__copy__()
            for i in range(self.n_):
                y[i] *= s[i]
        else:
            raise TypeError(f'Only numeric and Vector objects are allowed in multiplication, but an object of type {type(s)} was used.')
        return y

    def __mul__(self, s):
        """
        Right-multiply self, either by a scalar or a vector.

        When s is a scalar, this a right-multiplication by a scalar,
        very similar to a left-multiplication. When other is a Vector, of
        the same length as self, this is a point-wise (Hadamard) product.

        This allows the infix notation
        y = x * s
        """
        return self.internalMul(s)

    def __rmul__(self, s):
        """
        Left-multiply self, either by a scalar or a vector.

        When s is a scalar, this a right-multiplication by a scalar,
        very similar to a left-multiplication. When other is a Vector, of
        the same length as self, this is a point-wise (Hadamard) product.

        This allows the infix notation
        y = s * x
        """
        return self.internalMul(s)

    def __add__(self, y):
        """
        Add two vectors together if they have the same length.

        This allows the infix notation:
        z = x + y
        """
        if type(y) is not Vector:
            return TypeError("Can only add a vector to another vector.")
        if len(y) != self.n_:
            return TypeError("To add vectors together, they must have the same length.")
        z = self.__copy__()
        for i in range(len(y)):
            z[i] += y[i]
        return z

    def __sub__(self, y):
        """
        Subtract other from self if they have the same length.

        This allows the infix notation:
        z = x - y
        """
        if type(y) is not Vector:
            return TypeError("Can only add a vector to another vector.")
        if len(y) != self.n_:
            return TypeError("To add vectors together, they must have the same length.")
        z = self.__copy__()
        for i in range(len(y)):
            z[i] -= y[i]
        return z

    @staticmethod
    def fromArray(array, mode='column'):
        """
        Create a Vector from a 1D array.

        array must be 1D and numeric, otherwise an exception is raised.
        """
        n = len(array)
        v = Vector(n, mode=mode)
        for i in range(n):
            if isinstance(array[i], numbers.Number):
                v[i] = float(array[i])
            else:
                raise VectorException('elements of array must be numeric!')
        return v

    @staticmethod
    def ones(n, mode='column'):
        """Returns a vector filled with 1."""
        x = Vector(n, mode=mode)
        for i in range(n):
            x[i] = 1.0
        return x

    def toString(self, openDelimiter=None, closeDelimiter=None, separator=None,
                 offset=None, precision=None) -> str:
        """Convert the vector to a string for printing."""
        # The horizontal as a classic list display, for the column,
        # I use delimiters at each line.
        if openDelimiter is None:
            openDelimiter = Vector.openDelimiter
        if closeDelimiter is None:
            closeDelimiter = Vector.closeDelimiter
        if separator is None:
            separator = Vector.separator
        if offset is None:
            offset = Vector.offset
        if precision is None:
            precision = Vector.precision

        lsep = self.separator + " "

        def integerPartWidth(x: float) -> int:
            sign = 1 if x < 0 else 0
            return len(f'{math.floor(x)}') + sign

        def makeLine(width: int) -> str:
            return lsep.join([f'{self.data[i]:{width}.{precision}f}'
                              for i in range(self.n_)])

        def str_repeat(c: str, n: int) -> str:
            return "".join([c] * n)

        width = 0
        for i in range(self.n_):
            x_width = integerPartWidth(self.data[i])
            if x_width > width:
                width = x_width
        # for the decimal point, in likely case there is one
        width += 1

        if self.mode_ == 'row':
            A = str_repeat(" ", offset) + openDelimiter + makeLine(width) + closeDelimiter
            return A
        else:
            A = ""
            for i in range(self.n_):
                A += str_repeat(" ", self.offset) + openDelimiter + \
                     f'{self.data[i]:{width}.{precision}f}' + closeDelimiter
                if i != self.n_:
                    A += "\n"
            return A

    def __str__(self):
        return self.toString()
