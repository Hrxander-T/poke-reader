"""
pokemmo_extract.py - read a PokeMMO "preview" card screenshot into structured data.

Usage (Colab):
    import sys; sys.path.insert(0, FOLDER)
    import pokemmo_extract as pe
    pe.init(FOLDER + 'pokemon_data.json')   # local data file, built once by build_pokemon_data.py
    results = pe.process_folder(FOLDER)     # writes results.jsonl, cards.csv, review.csv

Single image:
    card = pe.extract_card(cv2.imread(path), 'name.png')
"""
import os, re, json, glob, csv
import cv2
import numpy as np
import pytesseract
from pytesseract import Output
from rapidfuzz import process, fuzz

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------
LABELS = {'ability': 'Ability', 'nature': 'Nature', 'region': 'Region', 'hidden_power': 'Hidden'}
NATURES = ['Hardy', 'Lonely', 'Brave', 'Adamant', 'Naughty', 'Bold', 'Docile', 'Relaxed', 'Impish', 'Lax',
           'Timid', 'Hasty', 'Serious', 'Jolly', 'Naive', 'Modest', 'Mild', 'Quiet', 'Bashful', 'Rash',
           'Calm', 'Gentle', 'Sassy', 'Careful', 'Quirky']
REGIONS = ['Kanto', 'Johto', 'Hoenn', 'Sinnoh', 'Unova']
HP_TYPES = ['Fighting', 'Flying', 'Poison', 'Ground', 'Rock', 'Bug', 'Ghost', 'Steel', 'Fire', 'Water',
            'Grass', 'Electric', 'Psychic', 'Ice', 'Dragon', 'Dark']
STAT_KEYS = ['ATK', 'DEF', 'SPD', 'SPATK', 'SPDEF', 'HP']
NATURE_FX = {  # nature: (boosted stat, lowered stat)
    'Lonely': ('ATK', 'DEF'), 'Brave': ('ATK', 'SPD'), 'Adamant': ('ATK', 'SPATK'), 'Naughty': ('ATK', 'SPDEF'),
    'Bold': ('DEF', 'ATK'), 'Relaxed': ('DEF', 'SPD'), 'Impish': ('DEF', 'SPATK'), 'Lax': ('DEF', 'SPDEF'),
    'Timid': ('SPD', 'ATK'), 'Hasty': ('SPD', 'DEF'), 'Jolly': ('SPD', 'SPATK'), 'Naive': ('SPD', 'SPDEF'),
    'Modest': ('SPATK', 'ATK'), 'Mild': ('SPATK', 'DEF'), 'Quiet': ('SPATK', 'SPD'), 'Rash': ('SPATK', 'SPDEF'),
    'Calm': ('SPDEF', 'ATK'), 'Gentle': ('SPDEF', 'DEF'), 'Sassy': ('SPDEF', 'SPD'), 'Careful': ('SPDEF', 'SPATK')}
NEUTRAL = {'Hardy', 'Docile', 'Serious', 'Bashful', 'Quirky'}
FX_TO_NATURE = {v: k for k, v in NATURE_FX.items()}

ABILITIES, POKEMON = [], []   # filled by init()


# --------------------------------------------------------------------------
# Local Pokemon data (built once by build_pokemon_data.py, no internet needed afterwards)
# --------------------------------------------------------------------------
DATA = {}   # display name -> {'types': [...], 'abilities': [...], 'base': {...}}


def init(data_path):
    """Load pokemon_data.json. Fills DATA, POKEMON (names) and ABILITIES (for fuzzy matching)."""
    global DATA, ABILITIES, POKEMON
    DATA = json.load(open(data_path))['pokemon']
    POKEMON = sorted(DATA)
    ABILITIES = sorted({a for v in DATA.values() for a in v['abilities']})


def pokemon_info(name):
    """Types, abilities and base stats for a Pokemon, or None if unknown."""
    return DATA.get(name)


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------
def snap(text, choices, cutoff=60):
    """Fuzzy-match OCR text to the nearest official name, or None."""
    if not text:
        return None
    r = process.extractOne(text, choices, scorer=fuzz.WRatio, score_cutoff=cutoff,
                           processor=lambda s: s.lower())
    return r[0] if r else None


def ocr_block(crop, psm=6, scale=4):
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    g = cv2.bitwise_not(g)
    return pytesseract.image_to_string(g, config=f'--psm {psm}').strip()


def ocr_line(crop):
    return ocr_block(crop, psm=7)


def stats_roi(img):
    h, w = img.shape[:2]
    return img[int(0.58 * h):, int(0.50 * w):]


# --------------------------------------------------------------------------
# Left panel: Ability / Nature / Region / Hidden Power (anchored on the labels)
# --------------------------------------------------------------------------
def locate_labels(img):
    """{field: y_center_of_label_in_original_pixels}"""
    h, w = img.shape[:2]
    y0, x1, S = int(0.28 * h), int(0.55 * w), 4
    g = cv2.cvtColor(img[y0:, :x1], cv2.COLOR_BGR2GRAY)
    g = cv2.bitwise_not(cv2.resize(g, None, fx=S, fy=S, interpolation=cv2.INTER_CUBIC))
    d = pytesseract.image_to_data(g, config='--psm 11', output_type=Output.DICT)
    found = {}
    for i, t in enumerate(d['text']):
        t = t.strip()
        if not t:
            continue
        for key, word in LABELS.items():
            if key not in found and fuzz.ratio(t.lower(), word.lower()) >= 75:
                found[key] = y0 + (d['top'][i] + d['height'][i] / 2) / S
    return found


def crop_values(img):
    h, w = img.shape[:2]
    crops = {}
    for key, yc in locate_labels(img).items():
        top, bot = int(yc + 0.025 * w), int(yc + 0.075 * w)   # value line sits ~0.05*w below its label
        crops[key] = img[max(top, 0):bot, int(0.04 * w):int(0.50 * w)]
    return crops


# --------------------------------------------------------------------------
# Header: name, level, gender
# --------------------------------------------------------------------------
def header_variants(img):
    h, w = img.shape[:2]
    for y0, y1 in ((0, .14), (0, .18), (.02, .20), (0, .24)):
        for x0, x1 in ((0, .68), (.08, .68), (0, 1.0)):
            crop = img[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)]
            g0 = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            for scale in (3, 5):
                g = cv2.resize(g0, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
                _, otsu = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                for im in (cv2.bitwise_not(g), g, cv2.bitwise_not(otsu), otsu):
                    im = cv2.copyMakeBorder(im, 12, 12, 12, 12, cv2.BORDER_REPLICATE)
                    yield pytesseract.image_to_string(im, config='--psm 7').strip()


def read_name_level(img, extra_after_name=12):
    """Name + level from the header. Once the name is found, the level gets at most
    `extra_after_name` more variants; the icon reader and the stat formula are the backups."""
    name, level, raws, since = None, None, [], 0
    for t in header_variants(img):
        raws.append(t)
        if level is None:
            m = re.search(r'[LIl1]v\.?\s*(\d{1,3})', t)
            if m and 1 <= int(m.group(1)) <= 100:
                level = int(m.group(1))
        if name is None:
            for tok in re.findall(r"[A-Za-z'\.\-]{4,}", t):
                r = process.extractOne(tok, POKEMON, scorer=fuzz.ratio,
                                       score_cutoff=82, processor=lambda s: s.lower())
                if r:
                    name = r[0]
                    break
        if name and level:
            break
        if name:
            since += 1
            if since >= extra_after_name:
                break
    return name, level, raws


def read_gender(img):
    """Blue icon = M, pink icon = F (counted by pixel colour)."""
    h, w = img.shape[:2]
    roi = img[0:int(0.20 * h), int(0.20 * w):int(0.62 * w)]
    hue, sat, val = cv2.split(cv2.cvtColor(roi, cv2.COLOR_BGR2HSV))
    colored = (sat > 90) & (val > 110)
    blue = int((colored & (hue >= 95) & (hue <= 130)).sum())
    pink = int((colored & (hue >= 140) & (hue <= 179)).sum())
    if max(blue, pink) < 6:
        return 'N', (blue, pink)      # no icon at all = genderless (e.g. Kyurem)
    return ('M' if blue > pink else 'F'), (blue, pink)


def shiny_pixels(img):
    """Shiny cards have a gold 'S' badge left of the name. Count strongly saturated yellow pixels in a small
    box around the badge (shiny: ~35-55 px, normal: 0-2 on all test cards). The box skips the card edge,
    and the strict colour ignores tan game-background tiles that can show through the margin."""
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img[int(0.05 * h):int(0.17 * h), int(0.05 * w):int(0.20 * w)], cv2.COLOR_BGR2HSV)
    gold = (hsv[..., 0] >= 15) & (hsv[..., 0] <= 35) & (hsv[..., 1] > 180) & (hsv[..., 2] > 180)
    return int(gold.sum())


def read_level_icon(img):
    """Fallback level reader: anchor on the gender icon, read the digits to its right."""
    h, w = img.shape[:2]
    hdr = img[0:int(.20 * h), :int(.75 * w)]
    hue, sat, val = cv2.split(cv2.cvtColor(hdr, cv2.COLOR_BGR2HSV))
    m = (sat > 90) & (val > 110) & (((hue >= 95) & (hue <= 130)) | (hue >= 140))
    ys, xs = np.nonzero(m)
    if len(xs) < 6:
        return None
    x0 = int(xs.max()) + 2
    g0 = cv2.cvtColor(img[0:int(.20 * h), x0:x0 + int(.17 * w)], cv2.COLOR_BGR2GRAY)
    for scale in (4, 6, 8):
        g = cv2.resize(g0, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        for im in (cv2.bitwise_not(g), g):
            im = cv2.copyMakeBorder(im, 12, 12, 12, 12, cv2.BORDER_REPLICATE)
            t = pytesseract.image_to_string(im, config='--psm 7 -c tessedit_char_whitelist=Lv.0123456789')
            m2 = re.search(r'(\d{1,3})', t)
            if m2 and 1 <= int(m2.group(1)) <= 100:
                return int(m2.group(1))
    return None


# --------------------------------------------------------------------------
# Stats table
# --------------------------------------------------------------------------
def parse_stats(lines):
    """Each row is stat(2-3 digits) IV(2) EV(3). Spaces get lost, so split from the right.
    HP rows have no stat value. Reading stops once a full row is complete."""
    out = {}
    for ln in lines:
        m = re.match(r'^\W*([A-Za-z]+)\W*(.*)$', ln)
        if not m:
            continue
        key = process.extractOne(m.group(1).upper(), STAT_KEYS, scorer=fuzz.ratio, score_cutoff=60)
        if not key:
            continue
        key = key[0]
        need = 5 if key == 'HP' else 7
        d = ''
        for tok in m.group(2).split():
            d += re.sub(r'\D', '', tok)
            if len(d) >= need:
                break
        if len(d) < 5:
            continue
        out[key] = {'stat': int(d[:-5]) if d[:-5] else None, 'iv': int(d[-5:-3]), 'ev': int(d[-3:])}
    return out


def read_stat_colors(img):
    """Colour of each stat label: green = boosted by nature, orange = lowered."""
    oy, ox, S = int(0.58 * img.shape[0]), int(0.50 * img.shape[1]), 4
    roi = img[oy:, ox:]
    g = cv2.bitwise_not(cv2.resize(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), None, fx=S, fy=S,
                                   interpolation=cv2.INTER_CUBIC))
    d = pytesseract.image_to_data(g, config='--psm 6', output_type=Output.DICT)
    colors = {}
    for i, t in enumerate(d['text']):
        word = re.sub(r'[^A-Za-z]', '', t).upper()
        if len(word) < 2:
            continue
        m = process.extractOne(word, STAT_KEYS[:5], scorer=fuzz.ratio, score_cutoff=60)
        if not m or m[0] in colors:
            continue
        x, y, bw, bh = [int(d[k][i] / S) for k in ('left', 'top', 'width', 'height')]
        px = roi[y:y + bh, x:x + bw].reshape(-1, 3).astype(int)
        if len(px) == 0:
            continue
        ink = (px.max(1) > 110) & ((px.max(1) - px.min(1)) > 40)
        if ink.sum() < 4:
            colors[m[0]] = 'neutral'
            continue
        b, gg, r = px[ink].mean(0)
        if gg > r + 25 and gg > b + 25:
            colors[m[0]] = 'up'
        elif r > gg + 25 and r > b + 60:
            colors[m[0]] = 'down'
        else:
            colors[m[0]] = 'neutral'
    return colors


def nature_check(nature, colors):
    if len(colors) < 5:
        return f'unclear ({len(colors)}/5 labels read)'
    ups = [k for k, v in colors.items() if v == 'up']
    downs = [k for k, v in colors.items() if v == 'down']
    if not ups and not downs:
        return 'ok' if nature in NEUTRAL else f'MISMATCH: colors say neutral, nature={nature}'
    if len(ups) == 1 and len(downs) == 1:
        implied = FX_TO_NATURE.get((ups[0], downs[0]))
        return 'ok' if implied == nature else f'MISMATCH: colors imply {implied}, nature={nature}'
    return 'unclear (colors ' + str(colors) + ')'


# --------------------------------------------------------------------------
# Game-formula checks (stat = floor((2*base + IV + EV/4) * level / 100) + 5, then nature)
# --------------------------------------------------------------------------
def expected(info, card, L):
    up, down = NATURE_FX.get(card['nature'], (None, None))
    out = {}
    for k, base in info['base'].items():
        v = card['stats'].get(k)
        if not v:
            continue
        core = (2 * base + v['iv'] + v['ev'] // 4) * L // 100
        out[k] = core + L + 10 if k == 'HP' else (core + 5) * (11 if k == up else 9 if k == down else 10) // 10
    return out


def n_match(info, card, L):
    exp, n = expected(info, card, L), 0
    for k, e in exp.items():
        if k != 'HP' and card['stats'][k]['stat'] == e:
            n += 1
    return n


def identify_by_stats(card):
    """Fallback when the header name is unreadable: find species + level whose formula stats match all 5
    stat values (and whose ability list contains the ability we read). Returns [(name, level), ...]."""
    if not card['nature'] or sum(1 for k, v in card['stats'].items() if k != 'HP' and v['stat']) < 5:
        return []
    hits = []
    for name, info in DATA.items():
        if card['ability'] and card['ability'] not in info['abilities']:
            continue
        for L in range(1, 101):
            if n_match(info, card, L) >= 5:
                hits.append((name, L))
    return hits


def _verify_with_formula(card, info, warns):
    """Infer a missing/wrong level, fill unreadable stats, fix single-digit misreads, check max HP."""
    L = card['level']
    best = n_match(info, card, L) if L else -1
    if best < 4:   # level looks wrong or missing: solve it from the stats
        cand = max(range(1, 101), key=lambda x: n_match(info, card, x))
        nc = n_match(info, card, cand)
        if nc >= 4 and nc > best:
            card['level'], card['level_inferred'] = cand, True
    L = card['level']
    if not L:
        return
    exp = expected(info, card, L)
    for k, e in exp.items():                      # fill stats the OCR could not read
        if k != 'HP' and card['stats'][k]['stat'] is None:
            card['stats'][k]['stat'] = e
            card['filled'].append(k)
    bad = [k for k, e in exp.items() if k != 'HP' and card['stats'][k]['stat'] != e]
    good = [k for k, e in exp.items() if k != 'HP' and card['stats'][k]['stat'] == e]
    if len(bad) == 1 and len(good) >= 4:          # one stat wrong, rest right: allow a 1-digit fix
        k = bad[0]
        read, e = str(card['stats'][k]['stat']), str(exp[k])
        if len(read) == len(e) and sum(a != b for a, b in zip(read, e)) == 1:
            card['stats'][k]['stat'] = exp[k]
            card['corrected'].append(f'{k}: {read} -> {e}')
            bad = []
    for k in bad:
        warns.append(f"{k} read {card['stats'][k]['stat']}, formula says {exp[k]}")
    if 'HP' in exp:
        card['hp_max'] = exp['HP']   # max HP is calculated, current HP is not read


# --------------------------------------------------------------------------
# One card -> one dict
# --------------------------------------------------------------------------
def extract_card(img, file=''):
    vals = {k: ocr_line(c) for k, c in crop_values(img).items()}
    name, level, raws = read_name_level(img)
    if level is None:
        level = read_level_icon(img)
    gender, gcounts = read_gender(img)
    stats_lines = ocr_block(stats_roi(img), psm=6).splitlines()
    colors = read_stat_colors(img)
    info = pokemon_info(name) if name else None

    card = {'file': file, 'name': name, 'level': level, 'gender': gender, 'shiny': None,
            'hp_max': None,
            'ability': snap(vals.get('ability'), ABILITIES),
            'nature': snap(vals.get('nature'), NATURES),
            'region': snap(vals.get('region'), REGIONS),
            'hidden_power': snap(vals.get('hidden_power'), HP_TYPES),
            'types': info['types'] if info else [],      # types come from the local data file, not OCR
            'stats': parse_stats(stats_lines),
            'filled': [], 'corrected': [], '_img': img}
    card['nature_check'] = nature_check(card['nature'], colors)
    card['shiny_px'] = shiny_pixels(img)
    card['shiny'] = card['shiny_px'] >= 20

    candidates = []
    if not card['name']:
        hits = identify_by_stats(card)
        names = sorted({n for n, _ in hits})
        if len(names) == 1:
            card['name'], card['name_from_stats'] = names[0], True
            levels = {L for _, L in hits}
            if len(levels) == 1:
                card['level'] = levels.pop()      # stats-derived level beats a garbled header read
            info = pokemon_info(card['name'])
            card['types'] = info['types']
        else:
            candidates = names[:5]

    warns = []
    if candidates:
        warns.append(f'name unread; stats fit several species: {candidates}')
    if info and card['stats']:
        _verify_with_formula(card, info, warns)
        if card['ability'] and card['ability'] not in info['abilities']:
            warns.append(f"ability {card['ability']} not in {info['abilities']}")
    elif not info:
        warns.append('name unknown: no type/formula check')
    if len(card['stats']) != 6:
        warns.append(f"stats rows found: {len(card['stats'])}/6")
    for f in ('name', 'level', 'gender', 'hp_max', 'ability', 'nature', 'region', 'hidden_power'):
        if not card.get(f):
            warns.append(f'{f} missing')
    for k, v in card['stats'].items():
        if v['iv'] > 31 or v['ev'] > 252:
            warns.append(f'{k} iv/ev out of range')
    if card['nature_check'].startswith('MISMATCH'):
        warns.append(card['nature_check'])
    card['warnings'] = warns
    if warns:
        card['debug'] = {'stats_raw': stats_lines, 'stat_colors': colors,
                         'gender_px(blue,pink)': gcounts,
                         'header_raw': ' || '.join(dict.fromkeys(raws))[:200]}
    card.pop('_img')
    return card


# --------------------------------------------------------------------------
# Batch: folder in, jsonl + cards.csv + review.csv out
# --------------------------------------------------------------------------
def _row(c):
    row = {'file': c.get('file'), 'name': c.get('name'), 'level': c.get('level'), 'gender': c.get('gender'), 'shiny': c.get('shiny'),
           'hp_max': c.get('hp_max'), 'types': '/'.join(c.get('types') or []),
           'ability': c.get('ability'), 'nature': c.get('nature'), 'region': c.get('region'),
           'hidden_power': c.get('hidden_power')}
    for k in STAT_KEYS:
        v = (c.get('stats') or {}).get(k, {})
        row[f'{k}_stat'], row[f'{k}_iv'], row[f'{k}_ev'] = v.get('stat'), v.get('iv'), v.get('ev')
    row['notes'] = '; '.join((c.get('warnings') or []) + [f'corrected {x}' for x in c.get('corrected') or []])
    return row


def _write_csv(path, cards):
    rows = [_row(c) for c in cards]
    if not rows:
        return
    with open(path, 'w', newline='', encoding='utf-8') as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)


def print_card(c):
    tag = (' (level inferred)' if c.get('level_inferred') else '') + (' (name from stats)' if c.get('name_from_stats') else '') + (f" SHINY({c.get('shiny_px')}px)" if c.get('shiny') else '')
    print(f"{c['file']} | {c['name']} Lv{c['level']}{tag} {c['gender']} HP max {c['hp_max']} | "
          f"{c['types']} | {c['ability']} | {c['nature']} | {c['region']} | HP-{c['hidden_power']}")
    if c.get('corrected'):
        print('    CORRECTED (please eyeball):', c['corrected'])
    if c.get('filled'):
        print('    filled from formula:', c['filled'])
    if c.get('warnings'):
        print('    WARN:', c['warnings'])
        print('    DEBUG:', c.get('debug'))


def process_folder(folder, out_dir=None, verbose=True):
    out_dir = out_dir or folder
    paths = sorted(p for ext in ('png', 'jpg', 'jpeg')
                   for p in glob.glob(os.path.join(folder, f'*.{ext}')))
    results = []
    for p in paths:
        img = cv2.imread(p)
        if img is None:
            continue
        try:
            card = extract_card(img, os.path.basename(p))
        except Exception as e:   # one bad image must not stop the batch
            card = {'file': os.path.basename(p), 'warnings': [f'error: {e!r}'], 'stats': {}}
        results.append(card)
        if verbose:
            if 'error' in ' '.join(card['warnings']) and 'name' not in card:
                print(card['file'], '| ERROR', card['warnings'])
            else:
                print_card(card)
    with open(os.path.join(out_dir, 'results.jsonl'), 'w') as f:
        f.write('\n'.join(json.dumps(c, ensure_ascii=False) for c in results))
    good = [c for c in results if not c.get('warnings') and not c.get('corrected')]
    review = [c for c in results if c.get('warnings') or c.get('corrected')]
    _write_csv(os.path.join(out_dir, 'cards.csv'), good)
    _write_csv(os.path.join(out_dir, 'review.csv'), review)
    print(f'\n{len(results)} cards: {len(good)} clean -> cards.csv, {len(review)} to review -> review.csv')
    return results
