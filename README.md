Sudoku Image Solver

Overview:
This project takes an image of a Sudoku puzzle, extracts the grid, and solves it using constraint-based logic. It is built as an end-to-end pipeline combining image processing, structured data extraction, and algorithmic solving.

Approach:
- Image Processing
- Convert input image to grayscale
- Apply thresholding and contour detection
- Detect and isolate the Sudoku grid
- Grid Extraction
- Segment the grid into individual cells
- Extract digits using image processing / OCR methods
- Convert extracted values into a structured 9x9 matrix
- Solving
- Apply backtracking-based constraint solving
- Enforce Sudoku rules across rows, columns, and subgrids
- Output
- Generate solved grid
- Display input and output through a simple web interface

Tech Stack
- Python
- Flask
- OpenCV
- NumPy
- pytesseract
- easyOCR

Future Work
- Improve digit recognition accuracy
- Handle noisy or low-quality images
- Extend to API-based deployment
- Optimize processing pipeline for speed
