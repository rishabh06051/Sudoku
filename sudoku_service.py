import time
from imageprocess import Imageprocess
from properties import file
from solver import Solver


def sudoko_service(inp_path, output_path):
    time1 = time.time()
    imageprocess = Imageprocess()
    img = imageprocess.read(file)

    gray = imageprocess.tograyscale(img)
    board, cells = imageprocess.cell_detection(gray)

    time_celldetection = time.time()
    print("start - cell detection", time_celldetection - time1)
    board = imageprocess.recognize_cells_easyocr(cells)

    filled = []
    for r in range(9):
        for c in range(9):
            if board[r][c] != '.':
                filled.append((r, c))

    time2 = time.time()
    print("Image processing time", time2 - time1)
    solver = Solver(board)
    print(board)
    if solver.isvalid(board):
        solved_sudoku = solver.solution(board)
        if solved_sudoku:
            print(solved_sudoku)
            imageprocess.render_sudoku_solution(solved_sudoku, filled)
        else:
            print("The sudoku is unsolvable")
    else:
        print("The Sudoku is not valid")
    time3 = time.time()
    print("Sudoke solving time", time3 - time2)
