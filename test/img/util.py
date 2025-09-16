import numpy as np


def make_chessboard_rgb_and_mask(H: int = 128, W: int = 128, cell: int = 32):
    img = np.zeros((H, W, 3), dtype=np.uint8)
    msk = np.zeros((H, W), dtype=np.uint8)
    for y in range(0, H, cell):
        for x in range(0, W, cell):
            val = 255 if ((y // cell + x // cell) % 2 == 0) else 0
            img[y:y + cell, x:x + cell, :] = val
            msk[y:y + cell, x:x + cell] = 1 if val == 255 else 0
    return img, msk
