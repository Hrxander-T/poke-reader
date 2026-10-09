"""
pokemmo_detect.py - find the card inside a full (uncropped) screenshot, then hand it to pokemmo_extract.

How it works: the four label banners (Ability / Nature / Region / Hidden Power) look the same on every
card. We match them at several sizes, so the best size also tells us the UI scale. The card box is then
placed relative to the labels, cut out, resized back to normal size, and passed to extract_card().

One-time setup (builds card_template.npz from any cropped card of yours):
    import cv2, pokemmo_detect as pd
    pd.make_template(cv2.imread(FOLDER + 'some cropped card.png'), FOLDER + 'card_template.npz')

Every time:
    tpl = pd.load_template(FOLDER + 'card_template.npz')
    results = pd.process_full_folder(FOLDER + 'full/', tpl)
"""
import os, glob, json
import cv2
import numpy as np
import pokemmo_extract as pe

KEYS = ['ability', 'nature', 'region', 'hidden_power']

# Game layout facts at UI scale 1.0 (measured on a real screenshot):
CARD_TOP_TO_ABILITY = 131.4   # px from the card's top border to the centre of the "Ability" label
TOP_MARGIN = 8                # how many px of margin to keep above the card border (header OCR needs some)
LEFT_MARGIN = 3               # same idea on the left side
CARD_LEFT_TO_TEXT = 24        # px from the card's left border to where the label text starts
CROP_W, CROP_H = 347, 348     # size of the crop we hand to the extractor, margins included


# --------------------------------------------------------------------------
# Template (built once from a cropped card)
# --------------------------------------------------------------------------
def _text_left(gray, y_center, thr=190):
    """x where the bright label text starts on the row of a label."""
    band = gray[int(y_center) - 6:int(y_center) + 6, :int(0.4 * gray.shape[1])]
    cols = np.nonzero(band.max(0) > thr)[0]
    if len(cols) == 0:
        raise ValueError('could not find the label text on the reference card')
    return int(cols.min())


def make_template(ref_img, out_path):
    lab = pe.locate_labels(ref_img)
    missing = [k for k in KEYS if k not in lab]
    if missing:
        raise ValueError(f'labels not found on the reference card: {missing} - try another cropped card')
    g = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
    H, W = g.shape
    tl = _text_left(g, lab['ability'])
    x0 = max(tl - 4, 0)
    x1 = min(x0 + 150, W)
    arrays = {'size': np.array([W, H]), 'ability_y': np.array([lab['ability']]), 'text_left': np.array([tl])}
    for k in KEYS:
        y0 = int(round(lab[k])) - 12
        arrays['t_' + k] = g[y0:y0 + 24, x0:x1]
        arrays['off_' + k] = np.array([x0, y0])
    np.savez(out_path, **arrays)
    print('template saved:', out_path, '| reference card size', W, 'x', H)


def load_template(path):
    d = np.load(path)
    return {k: d[k] for k in d.files}


# --------------------------------------------------------------------------
# Detection
# --------------------------------------------------------------------------
def _score_map(g, tpl, s):
    """comb[y, x] = average label match score if the card's top-left corner is at (x, y), at scale s."""
    maps = []
    for k in KEYS:
        t = tpl['t_' + k]
        ox, oy = tpl['off_' + k]
        ts = cv2.resize(t, None, fx=s, fy=s, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
        if ts.shape[0] >= g.shape[0] or ts.shape[1] >= g.shape[1]:
            return None
        m = np.nan_to_num(cv2.matchTemplate(g, ts, cv2.TM_CCOEFF_NORMED), nan=0.0, posinf=0.0, neginf=0.0)
        maps.append((m, int(round(ox * s)), int(round(oy * s))))
    hc = min(m.shape[0] - dy for m, dx, dy in maps)
    wc = min(m.shape[1] - dx for m, dx, dy in maps)
    if hc <= 0 or wc <= 0:
        return None
    return sum(m[dy:dy + hc, dx:dx + wc] for m, dx, dy in maps) / len(maps)


def find_card(full, tpl, min_score=0.6, fast_score=0.75, scales=None):
    """Return {'score', 'scale', 'box': (x, y, w, h), 'card': image resized to reference size} or None."""
    g = cv2.cvtColor(full, cv2.COLOR_BGR2GRAY)
    if scales is None:
        scales = [1.0] + [round(s, 2) for s in np.arange(0.5, 2.01, 0.05) if abs(s - 1.0) > 1e-6]
    best = None
    for s in scales:
        comb = _score_map(g, tpl, s)
        if comb is None:
            continue
        _, mx, _, loc = cv2.minMaxLoc(comb)
        if best is None or mx > best['score']:
            best = {'score': float(mx), 'scale': s, 'xy': loc}
        if s == 1.0 and mx >= fast_score:   # normal case: UI scale 1.0, no need to search other sizes
            break
    if best is None or best['score'] < min_score:
        return None
    s, (x, y) = best['scale'], best['xy']
    # (x, y) is where the reference card's top-left corner would be. Reference crops have different margins,
    # so we re-anchor on the real card border: a fixed distance from the label text.
    top = y + (float(tpl['ability_y'][0]) - CARD_TOP_TO_ABILITY - TOP_MARGIN) * s
    left = x + (float(tpl['text_left'][0]) - CARD_LEFT_TO_TEXT - LEFT_MARGIN) * s
    y0, x0 = max(int(round(top)), 0), max(int(round(left)), 0)
    w, h = int(round(CROP_W * s)), int(round(CROP_H * s))
    crop = full[y0:y0 + h, x0:x0 + w]
    if s != 1.0:
        crop = cv2.resize(crop, (CROP_W, CROP_H), interpolation=cv2.INTER_AREA if s > 1 else cv2.INTER_CUBIC)
    return {'score': round(best['score'], 3), 'scale': float(s), 'box': (x0, y0, w, h), 'card': crop}


# --------------------------------------------------------------------------
# Full screenshot -> card data
# --------------------------------------------------------------------------
def extract_from_screenshot(full, tpl, file=''):
    det = find_card(full, tpl)
    if det is None:
        return {'file': file, 'stats': {}, 'warnings': ['card not found'], 'detect': None}, None
    card = pe.extract_card(det['card'], file)
    card['detect'] = {'score': det['score'], 'scale': det['scale'], 'box': det['box']}
    return card, det['card']


def process_full_folder(folder, tpl, out_dir=None, verbose=True):
    out_dir = out_dir or folder
    found_dir = os.path.join(out_dir, 'found')
    os.makedirs(found_dir, exist_ok=True)
    paths = sorted(p for ext in ('png', 'jpg', 'jpeg') for p in glob.glob(os.path.join(folder, f'*.{ext}')))
    results = []
    for p in paths:
        img = cv2.imread(p)
        if img is None:
            continue
        name = os.path.basename(p)
        try:
            card, crop = extract_from_screenshot(img, tpl, name)
        except Exception as e:   # one bad image must not stop the batch
            card, crop = {'file': name, 'stats': {}, 'warnings': [f'error: {e!r}'], 'detect': None}, None
        if crop is not None:
            cv2.imwrite(os.path.join(found_dir, name), crop)   # so you can eyeball what was cut out
        results.append(card)
        if verbose:
            d = card.get('detect')
            print(f"[{img.shape[1]}x{img.shape[0]}] detect:", d if d else 'NOT FOUND')
            if card.get('name') or card.get('stats'):
                pe.print_card(card)
            else:
                print('   ', name, card['warnings'])
    with open(os.path.join(out_dir, 'full_results.jsonl'), 'w') as f:
        f.write('\n'.join(json.dumps(c, ensure_ascii=False, default=str) for c in results))
    good = [c for c in results if not c.get('warnings') and not c.get('corrected')]
    review = [c for c in results if c.get('warnings') or c.get('corrected')]
    pe._write_csv(os.path.join(out_dir, 'full_cards.csv'), good)
    pe._write_csv(os.path.join(out_dir, 'full_review.csv'), review)
    print(f'\n{len(results)} screenshots: {len(good)} clean -> full_cards.csv, {len(review)} to review -> full_review.csv')
    return results
