"""
@Project: LinalgDat 2022
@File: TestProjectB.py

@Description: Routines for testing implementations for the second LinalgDat
programming project.

A few routines are there to check validity of some vectors/matrix objects.
The tests have the same structure:
1) Check that teh implemented function runs
2) When applicable, test that the result has the expected size/dimension
3) Check that the result has the expected value
Only when all these checks are successful does the test method return True.

This an adaptation of my previous F# and C# code
"""

from sys import (
    path,
    modules
)
path.append('../Core')
from Vector import Vector
from Matrix import Matrix

import GaussExtensions
from data_projectB import (
    ERRArgs,
    ERIArgs,
    ERSArgs,
    FRArgs,
    BRArgs,
    GEArgs
)

__author__ = "François Lauze, University of Copenhagen"
__date__ = "04/27/22"
__version__ = "0.0.1"

Tolerance = 1e-3 # keep it large because of rounding/precision error when emitting data


class ProjectBException(Exception):
    """Dummy exception class for the project."""
    pass


def l1_distance(v: Vector, w: Vector) -> float:
    """
    l1-distance between two vectors of the same length.

    Internal use, as the lengths are actually not compared.
    """
    l1 = 0.0
    for i in range(len(v)):
        l1 += abs(v[i] - w[i])
    return l1


def compareVectorDimensions(v: Vector, w: Vector) -> bool:
    """Return True if v and w have same length, False otherwise."""
    return len(v) == len(w)


def compareVectors(v: Vector, w: Vector) -> bool:
    """
    Return True if they have the same length and L1-distance is less than Tolerance.

    Somewhat obfuscated version...
    """
    if compareVectorDimensions(v, w):
        return l1_distance(v, w) < Tolerance
    else:
        return False


def compareMatrixDimensions(A: Matrix, B: Matrix) -> bool:
    """Compare matrix dimensions."""
    return (A.N_Cols == B.N_Cols) and (A.M_Rows == B.M_Rows)


def compareMatrices(A: Matrix, B: Matrix) -> bool:
    """Compare matrices up to tolerance via l1-distance."""
    if compareMatrixDimensions(A, B):
        l1_norm = 0.0
        for i in range(A.N_Cols):
            l1_norm += l1_distance(A.Row(i), B.Row(i))
        return l1_norm < Tolerance
    else:
        raise ProjectBException("Matrix dimensions mismatch.")


def outMessage(taskName: str, subTaskName: str, status: bool) -> str:
    """Display a message followed by [PASSED] or [FAILED]."""
    passed = '[PASSED]'
    failed = '[FAILED]'
    s = f'{taskName} {subTaskName}'
    return f'{s:60} {passed if status else failed}'


# noinspection PyBroadException
def TestERR(A: Matrix, p: tuple, f: float, expected: Matrix) -> tuple:
    """Test that ElementaryRowSReplacement returns a correct result."""
    taskName = 'ElementaryRowReplacement(Matrix, int, float, int)'
    status = True
    resultStr = f'\nTests for the {taskName} function'
    resultStr += '\n==============================================================================='
    i, j = p
    try:
        B = GaussExtensions.ElementaryRowReplacement(A, i, f, j)
        try:
            if not compareMatrices(B, expected):
                resultStr += f'\n{outMessage(taskName, "Values", False)}'
                status = False
                # TODO: dump the expected result and some context.
            else:
                resultStr += f'\n{outMessage(taskName, "Dims", True)}'
                resultStr += f'\n{outMessage(taskName, "Values", True)}'
        except:
            resultStr += f'\n{outMessage(taskName, "Dims", False)}'
            status = False
    except Exception as e:
        resultStr += f'\n{outMessage(taskName, "Run", False)}'
        status = False

    if status:
        resultStr += f'\n{outMessage(taskName, "All", True)}'
    resultStr += f'\nEnd of test for the {taskName} function.'
    resultStr += '\n-------------------------------------------------------------------------------\n'
    return taskName, status, resultStr


def TestERI(A: Matrix, p: tuple, expected: Matrix) -> tuple:
    """Test that ElementaryRowInterchange returns a correct result."""
    taskName = 'ElementaryRowInterchange(Matrix, int, int)'
    status = True
    resultStr = f'\nTests for the {taskName} function'
    resultStr += '\n==============================================================================='
    i, j = p
    try:
        B = GaussExtensions.ElementaryRowInterchange(A, i, j)
        try:
            if not compareMatrices(B, expected):
                resultStr += f'\n{outMessage(taskName, "Values", False)}'
                status = False
                # TODO: dump the expected result and some context.
            else:
                resultStr += f'\n{outMessage(taskName, "Dims", True)}'
                resultStr += f'\n{outMessage(taskName, "Values", True)}'
        except:
            resultStr += f'\n{outMessage(taskName, "Dims", False)}'
            status = False
    except Exception as e:
        resultStr += f'\n{outMessage(taskName, "Run", False)}'
        status = False

    if status:
        resultStr += f'\n{outMessage(taskName, "All", True)}'
    resultStr += f'\nEnd of test for the {taskName} function.'
    resultStr += '\n-------------------------------------------------------------------------------\n'
    return taskName, status, resultStr


def TestERS(A: Matrix, i: int, c: float, expected: Matrix) -> tuple:
    """Test that ElementaryRowScaling returns a correct result."""
    taskName = 'ElementaryRowScaling(Matrix, int, float)'
    status = True
    resultStr = f'\nTests for the {taskName} function'
    resultStr += '\n==============================================================================='
    try:
        B = GaussExtensions.ElementaryRowScaling(A, i, c)
        try:
            if not compareMatrices(B, expected):
                resultStr += f'\n{outMessage(taskName, "Values", False)}'
                status = False
                # TODO: dump the expected result and some context.
            else:
                resultStr += f'\n{outMessage(taskName, "Dims", True)}'
                resultStr += f'\n{outMessage(taskName, "Values", True)}'
        except:
            resultStr += f'\n{outMessage(taskName, "Dims", False)}'
            status = False
    except Exception as e:
        resultStr += f'\n{outMessage(taskName, "Run", False)}'
        status = False

    if status:
        resultStr += f'\n{outMessage(taskName, "All", True)}'
    resultStr += f'\nEnd of test for the {taskName} function.'
    resultStr += '\n-------------------------------------------------------------------------------\n'
    return taskName, status, resultStr


def TestFR(A: Matrix, expected: Matrix) -> tuple:
    """Test that ForwardReduction returns a correct result."""
    taskName = 'ForwardReduction(Matrix)'
    status = True
    resultStr = f'\nTests for the {taskName} function'
    resultStr += '\n==============================================================================='
    try:
        B = GaussExtensions.ForwardReduction(A)
        try:
            if not compareMatrices(B, expected):
                resultStr += f'\n{outMessage(taskName, "Values", False)}'
                status = False
                # TODO: dump the expected result and some context.
            else:
                resultStr += f'\n{outMessage(taskName, "Dims", True)}'
                resultStr += f'\n{outMessage(taskName, "Values", True)}'
        except:
            resultStr += f'\n{outMessage(taskName, "Dims", False)}'
            status = False
    except Exception as e:
        resultStr += f'\n{outMessage(taskName, "Run", False)}'
        status = False

    if status:
        resultStr += f'\n{outMessage(taskName, "All", True)}'
    resultStr += f'\nEnd of test for the {taskName} function.'
    resultStr += '\n-------------------------------------------------------------------------------\n'
    return taskName, status, resultStr


def TestBR(A: Matrix, expected: Matrix) -> tuple:
    """Test that BackwardReduction returns a correct result."""
    taskName = 'BackwardReduction(Matrix)'
    status = True
    resultStr = f'\nTests for the {taskName} function'
    resultStr += '\n==============================================================================='
    try:
        B = GaussExtensions.BackwardReduction(A)
        try:
            if not compareMatrices(B, expected):
                resultStr += f'\n{outMessage(taskName, "Values", False)}'
                status = False
                # TODO: dump the expected result and some context.
            else:
                resultStr += f'\n{outMessage(taskName, "Dims", True)}'
                resultStr += f'\n{outMessage(taskName, "Values", True)}'
        except:
            resultStr += f'\n{outMessage(taskName, "Dims", False)}'
            status = False
    except Exception as e:
        resultStr += f'\n{outMessage(taskName, "Run", False)}'
        status = False

    if status:
        resultStr += f'\n{outMessage(taskName, "All", True)}'
    resultStr += f'\nEnd of test for the {taskName} function.'
    resultStr += '\n-------------------------------------------------------------------------------\n'
    return taskName, status, resultStr


def TestGE(A: Matrix, v: Vector, expected: Vector) -> tuple:
    """Test that GaussElimination returns a correct result."""
    taskName = 'GaussElimination(Matrix, Vector)'
    status = True
    resultStr = f'\nTests for the {taskName} function'
    resultStr += '\n==============================================================================='
    try:
        w = GaussExtensions.GaussElimination(A, v)
        try:
            if not compareVectors(w, expected):
                resultStr += f'\n{outMessage(taskName, "Values", False)}'
                status = False
                # TODO: dump the expected result and some context.
            else:
                resultStr += f'\n{outMessage(taskName, "Dims", True)}'
                resultStr += f'\n{outMessage(taskName, "Values", True)}'
        except:
            resultStr += f'\n{outMessage(taskName, "Dims", False)}'
            status = False
    except Exception as e:
        resultStr += f'\n{outMessage(taskName, "Run", False)}'
        status = False

    if status:
        resultStr += f'\n{outMessage(taskName, "All", True)}'
    resultStr += f'\nEnd of test for the {taskName} function.'
    resultStr += '\n-------------------------------------------------------------------------------\n'
    return taskName, status, resultStr


def runTestERR(matrices: list, indices: list, scalars: list, expected_matrices: list) -> tuple:
    """Run elementary row replacement tests and collect output."""
    n = len(matrices)
    passed = 0
    taskName = ''
    for A, p, f, E in zip(matrices, indices, scalars, expected_matrices):
        taskName, status, resultString = TestERR(A, p, f, E)
        passed += int(status)
        print(resultString)
    return taskName, passed, n


def runTestERI(matrices: list, indices: list, expected_matrices: list) -> tuple:
    """Run elementary row interchange tests and collect output."""
    n = len(matrices)
    passed = 0
    taskName = ''
    for A, p, E in zip(matrices, indices, expected_matrices):
        taskName, status, resultString = TestERI(A, p, E)
        passed += int(status)
        print(resultString)
    return taskName, passed, n


def runTestERS(matrices: list, indices: list, scalars: list, expected_matrices: list) -> tuple:
    """Run elementary row scaling tests and collect output."""
    n = len(matrices)
    passed = 0
    taskName = ''
    for A, i, f, E in zip(matrices, indices, scalars, expected_matrices):
        taskName, status, resultString = TestERS(A, i, f, E)
        passed += int(status)
        print(resultString)
    return taskName, passed, n


def runTestFR(matrices: list, expected_matrices: list) -> tuple:
    """Run forward reduction tests and collect output."""
    n = len(matrices)
    passed = 0
    taskName = ''
    for A,  E in zip(matrices, expected_matrices):
        taskName, status, resultString = TestFR(A, E)
        passed += int(status)
        print(resultString)
    return taskName, passed, n


def runTestBR(matrices: list, expected_matrices: list) -> tuple:
    """Run backward reduction tests and collect output."""
    n = len(matrices)
    passed = 0
    taskName = ''
    for A,  E in zip(matrices, expected_matrices):
        taskName, status, resultString = TestBR(A, E)
        passed += int(status)
        print(resultString)
    return taskName, passed, n


def runTestGE(matrices: list, vectors: list, expected_vectors: list) -> tuple:
    """Run  Gauss Elimination tests and collect output."""
    n = len(matrices)
    passed = 0
    taskName = ''
    for M, v, Mv in zip(matrices, vectors, expected_vectors):
        taskName, status, resultString = TestGE(M, v, Mv)
        passed += int(status)
        print(resultString)
    return taskName, passed, n


def printSummaryInfo(taskName: str, passed: int, total: int):
    """Print a line with execution summary for a given task."""
    str1 = f'Test of {taskName} passed/total'
    print(f'{str1:<70} [{passed}/{total}]')


def runALL():
    results = [
        runTestERR(*ERRArgs),
        runTestERI(*ERIArgs),
        runTestERS(*ERSArgs),
        runTestFR(*FRArgs),
        runTestBR(*BRArgs),
        runTestGE(*GEArgs)
    ]

    print('=' * 80)
    for result in results:
        printSummaryInfo(*result)
    print('-' * 80)


if __name__ == '__main__':
    if 'numpy' in modules:
        print("Numpy was imported!!!!!!!")
    else:
        runALL()