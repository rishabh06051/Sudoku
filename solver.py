from collections import deque
class Solver:

    def __init__(self, sudoku):
        self.sudoku = sudoku

    def isvalid(self, sudoku):
        boxes = [[] for _ in range(9)]
        for r in range(9):
            row = []
            column = []
            for c in range(9):
                if sudoku[r][c] != '.':
                    row.append(sudoku[r][c])
                if sudoku[c][r] != '.':
                    column.append(sudoku[c][r])
                    boxes[(r // 3)*3 + c//3].append(sudoku[c][r])
            if len(row) != len(set(row)) or len(column) != len(set(column)):
                return False

        for row in boxes:
            if len(row) != len(set(row)):
                return False

        return True

    def isSolved(self, sudoku):
        boxes = [[] for _ in range(9)]
        for r in range(9):
            row = []
            column = []
            for c in range(9):
                if sudoku[r][c] == '.':
                    return False
                row.append(sudoku[r][c])
                column.append(sudoku[c][r])
                boxes[r // 3 + c % 3] = sudoku[c][r]
            if len(row) != len(set(row)) or len(column) != len(set(column)):
                return False

        for row in boxes:
            if len(row) != len(set(row)):
                return False

        return True


    def solution(self, sudoku):

        que = deque([sudoku])

        while que:
            current = que.pop()
            if self.isSolved(current):
                return current
            looped = False
            for r in range(9):
                for c in range(9):
                    if current[r][c] == '.':
                        for i in '123456789':
                            new_board = [row[:] for row in current]
                            new_board[r][c] = i
                            if self.isvalid(new_board):
                                que.append(new_board)
                        looped = True
                        break
                if looped:
                    break
        return None