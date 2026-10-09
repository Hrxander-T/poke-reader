"""
pokemmo_reader.py - ONE entry point for everything.

Give it a cropped card OR a full screenshot (file path, raw bytes from an upload, or a numpy image) and
it returns the same result format either way.

    import pokemmo_reader as pr
    pr.load(FOLDER + 'pokemon_data.json', FOLDER + 'card_template.npz')    # once

    card = pr.read_image(FOLDER + 'some screenshot.png')     # path
    card = pr.read_image(uploaded_bytes, 'upload.png')       # bytes (web upload)
    print(card['status'], card['source'], card['name'], card['stats'])

    results = pr.process_folder(FOLDER + 'full/')            # a whole folder -> csv files

Result fields (besides the card data): 
    status : 'ok'        every check passed
             'review'    read, but a check failed or a value was auto-corrected (see 'warnings'/'corrected')
             'not_found' no card in the image
             'error'     the image could not be opened
    source : 'cropped' (image is the card itself) or 'screenshot' (card was cut out of a bigger image)
"""
import os, glob, json
import cv2
import numpy as np
import pokemmo_extract as pe
import pokemmo_detect as pd

_TPL = None


def load(data_path, template_path):
    """Load the Pokemon data file and the card template. Call once at startup."""
    global _TPL
    pe.init(data_path)
    _TPL = pd.load_template(template_path)


def _to_bgr(src):
    if isinstance(src, str):
        img = cv2.imread(src)
    elif isinstance(src, (bytes, bytearray)):
        img = cv2.imdecode(np.frombuffer(src, np.uint8), cv2.IMREAD_COLOR)
    else:
        img = src
    if img is None:
        return None
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


def _is_cropped(det, shape):
    """The found card sits in the top-left corner and the image is barely bigger than the card."""
    x, y, w, h = det['box']
    ih, iw = shape[:2]
    return x <= 12 and y <= 12 and iw <= w + 40 and ih <= h + 40


def _fail(file, status, msg):
    return {'file': file, 'status': status, 'source': None, 'detect': None, 'stats': {}, 'warnings': [msg]}


def read_image(src, file=''):
    if _TPL is None:
        raise RuntimeError('call pokemmo_reader.load(data_path, template_path) first')
    img = _to_bgr(src)
    if img is None:
        return _fail(file, 'error', 'could not open image')

    det = pd.find_card(img, _TPL)
    if det is not None:
        cropped = _is_cropped(det, img.shape)
        if cropped and abs(det['scale'] - 1.0) < 1e-6:
            card = pe.extract_card(img, file)               # normal-size cropped card: use it as is
        else:
            card = pe.extract_card(det['card'], file)       # cut out and resized to normal size
        card['source'] = 'cropped' if cropped else 'screenshot'
        card['detect'] = {'score': det['score'], 'scale': det['scale'], 'box': det['box']}
    elif len(pe.locate_labels(img)) >= 3:                   # finder missed it, but the labels are there
        card = pe.extract_card(img, file)
        card['source'], card['detect'] = 'cropped', None
    else:
        return _fail(file, 'not_found', 'card not found')

    card['status'] = 'ok' if not card['warnings'] and not card['corrected'] else 'review'
    return card


# --------------------------------------------------------------------------
# Helpers for the web layer and for batches
# --------------------------------------------------------------------------
def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def to_json(card, **kw):
    """JSON text of a result (numpy numbers converted)."""
    return json.dumps(card, default=_json_default, ensure_ascii=False, **kw)


def process_folder(folder, out_dir=None, verbose=True):
    """Read every image in a folder (cropped or full), write all_results.jsonl, all_cards.csv, all_review.csv."""
    out_dir = out_dir or folder
    paths = sorted(p for ext in ('png', 'jpg', 'jpeg') for p in glob.glob(os.path.join(folder, f'*.{ext}')))
    results = []
    for p in paths:
        name = os.path.basename(p)
        try:
            card = read_image(p, name)
        except Exception as e:   # one bad image must not stop the batch
            card = _fail(name, 'error', f'error: {e!r}')
        results.append(card)
        if verbose:
            print(f"[{card['status']}/{card['source']}]", end=' ')
            if card.get('name') or card.get('stats'):
                pe.print_card(card)
            else:
                print(name, card['warnings'])
    with open(os.path.join(out_dir, 'all_results.jsonl'), 'w') as f:
        f.write('\n'.join(to_json(c) for c in results))
    pe._write_csv(os.path.join(out_dir, 'all_cards.csv'), [c for c in results if c['status'] == 'ok'])
    pe._write_csv(os.path.join(out_dir, 'all_review.csv'), [c for c in results if c['status'] != 'ok'])
    n_ok = sum(c['status'] == 'ok' for c in results)
    print(f'\n{len(results)} images: {n_ok} ok -> all_cards.csv, {len(results) - n_ok} to review -> all_review.csv')
    return results
