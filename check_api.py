"""
check_api.py - run your frozen regression answers against the DEPLOYED API.
The server uses a different Tesseract build than Colab, so always do this after deploying.

    import json, check_api
    check_api.check('https://YOUR-API-URL', FOLDER, FOLDER + 'gold.json')                    # cropped cards
    check_api.check('https://YOUR-API-URL', FOLDER + 'full/', FOLDER + 'gold_full.json')     # screenshots
"""
import os, glob, json, time
import requests

FIELDS = ['name', 'level', 'gender', 'shiny', 'hp_max', 'ability', 'nature', 'region',
          'hidden_power', 'types', 'stats']


def check(api_url, folder, gold_path, api_key=None):
    gold = json.load(open(gold_path))
    headers = {'X-API-Key': api_key} if api_key else {}
    checked = diffs = 0
    for p in sorted(glob.glob(os.path.join(folder, '*.png')) + glob.glob(os.path.join(folder, '*.jpg'))):
        name = os.path.basename(p)
        if name not in gold:
            continue
        t = time.time()
        r = requests.post(api_url.rstrip('/') + '/read', files={'file': (name, open(p, 'rb'))},
                          headers=headers, timeout=180)
        secs = round(time.time() - t, 1)
        if r.status_code != 200:
            print(name, '-> HTTP', r.status_code, r.text[:100]); diffs += 1; continue
        card, checked = r.json(), checked + 1
        for k in FIELDS:
            if card.get(k) != gold[name][k]:
                diffs += 1
                print(f'{name} | {k}: was {gold[name][k]} -> API {card.get(k)}')
        print(f'  {name}  {secs}s  status={card["status"]}')
    print(f'checked {checked} images, {diffs} differences')
