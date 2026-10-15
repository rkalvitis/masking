"""Contact-sheet helpers: overlay mask contours on image crops."""
import numpy as np, cv2
from . import common as C

def overlay(rgb, mask, color=(255, 0, 0), alpha=0.35, thick=2):
    out = rgb.copy()
    m = mask.astype(bool)
    out[m] = (out[m] * (1 - alpha) + np.array(color) * alpha).astype(np.uint8)
    cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, cnts, -1, color, thick)
    return out

def crop_to(rgb, masks, pad=0.35, min_side=400):
    """Crop rgb (and masks) to the union bbox of masks with padding."""
    u = np.zeros(rgb.shape[:2], bool)
    for m in masks:
        u |= m
    b = C.bbox(u)
    if b is None:
        h, w = rgb.shape[:2]; b = (w//3, h//3, 2*w//3, 2*h//3)
    b = C.pad_box(b, rgb.shape[:2], pad_frac=pad, min_pad=min_side//4)
    b = C.square_box(b, rgb.shape[:2])
    x0, y0, x1, y1 = b
    return b, rgb[y0:y1, x0:x1], [m[y0:y1, x0:x1] for m in masks]

def label(img, text, scale=None):
    img = img.copy()
    h = img.shape[0]
    s = scale or max(0.5, h / 700)
    cv2.rectangle(img, (0, 0), (int(len(text) * 17 * s) + 10, int(32 * s)), (0, 0, 0), -1)
    cv2.putText(img, text, (5, int(24 * s)), cv2.FONT_HERSHEY_SIMPLEX, s * 0.9, (255, 255, 255), max(1, int(2 * s)))
    return img

def grid(tiles, ncol, tile_px=420):
    tiles = [cv2.resize(t, (tile_px, tile_px), interpolation=cv2.INTER_AREA) for t in tiles]
    while len(tiles) % ncol:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.concatenate(tiles[i:i + ncol], 1) for i in range(0, len(tiles), ncol)]
    return np.concatenate(rows, 0)
