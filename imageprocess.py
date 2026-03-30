import cv2 as cv
import easyocr
import numpy as np
import pytesseract
reader = easyocr.Reader(['en'], gpu=False)
pytesseract.pytesseract.tesseract_cmd = r"/opt/homebrew/bin/tesseract"



class Imageprocess:

    def read(self, pathfile):
        return cv.imread(pathfile)

    def tograyscale(self, img):
        return cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    def noise_reduction(self, img):
        return cv.medianBlur(img, ksize=3)

    def thresholding(self, img):
        return cv.threshold(img, 200, 255, cv.THRESH_BINARY_INV)[1]

    def morphing(self, img):
        return cv.morphologyEx(img, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_RECT, (3, 3)), iterations=1)

    def cell_detection(self, img):
        edges = cv.Canny(img, 50, 150)  # mild thresholds, tweak if needed
        cnts, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        big = max(cnts, key=cv.contourArea)
        peri = cv.arcLength(big, True)
        approx = cv.approxPolyDP(big, 0.02 * peri, True)  # expect 4 points
        assert len(approx) == 4, "Outer grid not found as a quadrilateral."

        src = self.order_quad(approx)
        SIDE = 900  # warp to 900x900; any multiple of 9
        dst = np.array([[0, 0], [SIDE - 1, 0], [SIDE - 1, SIDE - 1], [0, SIDE - 1]], dtype=np.float32)
        M = cv.getPerspectiveTransform(src, dst)
        board = cv.warpPerspective(img, M, (SIDE, SIDE))  # color warp (keep original detail)

        cell = SIDE // 9
        cells = []
        for r in range(9):
            row = []
            for c in range(9):
                y0, y1 = r * cell, (r + 1) * cell
                x0, x1 = c * cell, (c + 1) * cell
                row.append(board[y0:y1, x0:x1].copy())
            cells.append(row)
        return board, cells

    #def grid_collage(self, cells):
     #   return np.block(cells)  # stacks 9x9 into one image

    def order_quad(self, pts):
        pts = pts.reshape(4, 2).astype(np.float32)
        s = pts.sum(axis=1);
        d = np.diff(pts, axis=1).ravel()
        tl = pts[np.argmin(s)];
        br = pts[np.argmax(s)]
        tr = pts[np.argmin(d)];
        bl = pts[np.argmax(d)]
        return np.array([tl, tr, br, bl], dtype=np.float32)

    def prep_cell(self, cell):
        """Light preprocessing, no OCR."""
        if len(cell.shape) == 3:
            cell = cv.cvtColor(cell, cv.COLOR_BGR2GRAY)
        h, w = cell.shape
        frac = 0.12
        dx, dy = int(w * frac), int(h * frac)
        if h - 2 * dy > 5 and w - 2 * dx > 5:
            cell = cell[dy:h - dy, dx:w - dx]
        # keep small, ~96 px square is enough
        return cv.resize(cell, (96, 96), interpolation=cv.INTER_AREA)

    def recognize_cells_easyocr(self, cells):
        """
        Fast EasyOCR (recognition-only) with conservative 1↔7 handling and 2 protection.
        - Trust OCR unless conf < 0.55
        - Only flip 7->1 when '1' evidence is strong AND '7' is weak
        - Prevent 2->7 by checking top-vs-bottom ink balance
        """
        out = [['.' for _ in range(9)] for _ in range(9)]

        # --- helpers ---
        def _bin(im, block=15, C=2):
            return cv.adaptiveThreshold(im, 255, cv.ADAPTIVE_THRESH_MEAN_C,
                                        cv.THRESH_BINARY_INV, block, C)

        def _metrics(th):
            """Return shape cues used by heuristics."""
            h, w = th.shape
            if h == 0 or w == 0:
                return dict(aspect=1, top=0, centerline=0, diag=0, top_ratio=0.5, bot_ratio=0.5)
            # bounding box of biggest blob
            cnts, _ = cv.findContours(th, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            if not cnts:
                return dict(aspect=1, top=0, centerline=0, diag=0, top_ratio=0.5, bot_ratio=0.5)
            x, y, bw, bh = cv.boundingRect(max(cnts, key=cv.contourArea))
            aspect = bw / float(max(bh, 1))

            # top bar density
            top_band = th[: max(3, h // 8), :]
            top = (top_band.sum() / 255.0) / (top_band.size if top_band.size else 1)

            # vertical centerline density
            cx0 = max(0, x + bw // 2 - 1);
            cx1 = min(w, x + bw // 2 + 2)
            col = th[:, cx0:cx1]
            centerline = (col.sum() / 255.0) / (col.size if col.size else 1)

            # anti-diagonal (top-right -> bottom-left) density (the slash of '7')
            band_w = max(2, w // 12)
            mask = np.zeros_like(th, np.uint8)
            for i in range(h):
                j = int((w - 1) - (w - 1) * (i / max(h - 1, 1)))
                j0 = max(0, j - band_w // 2);
                j1 = min(w, j + band_w // 2 + 1)
                mask[i, j0:j1] = 255
            overlap = cv.bitwise_and(th, mask)
            diag = (overlap.sum() / 255.0) / (mask.sum() / 255.0 if mask.sum() else 1)

            # top vs bottom ink ratios (7 is top-heavy; 2 is bottom-heavy)
            top_half = th[:h // 2, :]
            bot_half = th[h // 2:, :]
            top_ratio = (top_half.sum() / 255.0) / (top_half.size if top_half.size else 1)
            bot_ratio = (bot_half.sum() / 255.0) / (bot_half.size if bot_half.size else 1)

            return dict(aspect=aspect, top=top, centerline=centerline, diag=diag,
                        top_ratio=top_ratio, bot_ratio=bot_ratio)

        def _parse_easy(res):
            # res for one image: [ (bbox, text, conf), ... ] or []
            if res and res[0]:
                raw = (res[0][1] or "").strip()
                conf = float(res[0][2]) if isinstance(res[0][2], (int, float)) else 0.0
                # normalize common confusions
                m = {'B': '8', '&': '8', '§': '8', 'Z': '2', 'A': '4', 'S': '5'}
                if raw in m: raw = m[raw]
                txt = "".join(ch for ch in raw if ch.isdigit())
                return txt, conf
            return "", 0.0

        for r in range(9):
            for c in range(9):
                im = self.prep_cell(cells[r][c])  # ~96x96 gray
                if im.ndim == 3: im = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
                if im.dtype != np.uint8: im = im.astype(np.uint8)

                th = _bin(im, 15, 2)
                # small vertical dilation helps thin '1' without creating top bars
                th = cv.dilate(th, cv.getStructuringElement(cv.MORPH_RECT, (1, 2)), 1)

                ink = (th.sum() / 255.0) / th.size
                if ink < 0.006:
                    out[r][c] = '.'
                    continue

                h, w = im.shape[:2]
                h_list = [[0, int(w), 0, int(h)]]  # recognizer-only box

                # Pass 1: recognizer
                res1 = reader.recognize(im, horizontal_list=h_list, free_list=[],
                                        detail=1, allowlist='0123456789B&§ZAS',
                                        decoder='greedy')
                txt1, conf1 = _parse_easy(res1)

                # Accept confident non-ambiguous
                if len(txt1) == 1 and conf1 >= 0.60 and txt1 not in ('1', '7'):
                    out[r][c] = txt1
                    continue

                # Compute metrics once
                M = _metrics(th)

                # Conservative flips for 1↔7 (only when OCR is weak)
                def strong_one(m):  # tall-thin, strong vertical, weak top bar
                    return (m['aspect'] < 0.30) and (m['centerline'] > 0.34) and (m['top'] < 0.06)

                def strong_seven(m):  # clear top bar + slash + not tall-thin + top-heavy
                    return (m['top'] >= 0.16) and (m['diag'] >= 0.14) and (m['aspect'] >= 0.50) and (
                                m['top_ratio'] > m['bot_ratio'] + 0.05)

                # If OCR said '7' with decent conf, do NOT flip unless '1' is very strong
                if txt1 == '7' and conf1 >= 0.60:
                    if strong_one(M) and not strong_seven(M):
                        out[r][c] = '1'
                    else:
                        out[r][c] = '7'
                    continue

                # If OCR said '1' with decent conf, keep it unless strong '7'
                if txt1 == '1' and conf1 >= 0.60:
                    if strong_seven(M) and not strong_one(M):
                        out[r][c] = '7'
                    else:
                        out[r][c] = '1'
                    continue

                # Low/blank: one small alt-preproc retry
                th2 = _bin(im, 19, 3)
                th2 = cv.erode(th2, cv.getStructuringElement(cv.MORPH_RECT, (2, 1)), 1)  # dampen top bar
                im2 = cv.bitwise_not(th2)
                res2 = reader.recognize(im2, horizontal_list=h_list, free_list=[],
                                        detail=1, allowlist='0123456789B&§ZAS',
                                        decoder='greedy')
                txt2, conf2 = _parse_easy(res2)

                # pick better of the two OCRs
                best_txt, best_conf = (txt2, conf2) if conf2 > conf1 else (txt1, conf1)

                # Final decision rules
                if len(best_txt) == 1 and best_conf >= 0.55:
                    # guard 2 vs 7: if best is '7' but bottom-heavy, prefer '2' if OCR suggested it anywhere
                    if best_txt == '7' and M['bot_ratio'] > M['top_ratio'] + 0.06:
                        # if either pass yielded a '2', use it; else keep '7'
                        if txt1 == '2' or txt2 == '2':
                            out[r][c] = '2'
                        else:
                            out[r][c] = '7'
                    else:
                        out[r][c] = best_txt
                    continue

                # Heuristic only when OCR weak
                if strong_one(M):
                    out[r][c] = '1'
                    continue
                if strong_seven(M):
                    out[r][c] = '7'
                    continue

                # 8 rescue (two holes)
                cnts, hier = cv.findContours(th, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)
                if cnts and hier is not None:
                    holes = sum(1 for i in range(len(cnts)) if hier[0][i][3] >= 0)
                    if holes >= 2:
                        out[r][c] = '8'
                        continue

                out[r][c] = '.'

        return out

    def render_sudoku_solution(self, solution, filled, size=900, out_file=None):
        """
        Render a clean Sudoku solution on a white grid.

        solution : 9x9 list of strings (digits '1'..'9' or '.')
        size     : image size in pixels (square)
        out_file : optional path to save PNG

        Returns: BGR image (numpy array)
        """
        # white background
        img = np.ones((size, size, 3), dtype=np.uint8) * 255
        cell = size // 9

        # draw grid
        for i in range(10):
            thickness = 3 if i % 3 == 0 else 1
            # vertical lines
            cv.line(img, (i * cell, 0), (i * cell, size), (0, 0, 0), thickness)
            # horizontal lines
            cv.line(img, (0, i * cell), (size, i * cell), (0, 0, 0), thickness)

        # put digits
        font = cv.FONT_HERSHEY_SIMPLEX
        font_scale = cell / 40.0 * 2.0  # scale relative to cell size
        thickness = max(2, cell // 20)

        for r in range(9):
            for c in range(9):
                if (r,c) in filled:
                    color = (0,0,255)
                else:
                    color = (0,0,0)
                v = str(solution[r][c])
                if v == '.':
                    continue
                (tw, th), base = cv.getTextSize(v, font, font_scale/2, thickness)
                x = c * cell + (cell - tw) // 2
                y = r * cell + (cell + th) // 2
                cv.putText(img, v, (x, y), font, font_scale/2, color, thickness, cv.LINE_AA)

        #if out_file:
         #   cv.imwrite(out_file, img)

        self.display(img)
        return img

    def display(self, img, str = "Display"):
        cv.imshow(str, img)
        cv.waitKey(0)
        cv.destroyWindow(str)







