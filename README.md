---
title: PokeMMO Card Reader
emoji: 🔎
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
---

# PokeMMO card reader API

Upload a screenshot (a cropped card or a full screenshot) and get the card's data back as JSON.

## Files in this folder

| File | What it is |
|---|---|
| `app.py` | the web API |
| `pokemmo_reader.py`, `pokemmo_detect.py`, `pokemmo_extract.py` | the reading code |
| `Dockerfile`, `requirements.txt` | how the server is built |
| `check_api.py` | tests the deployed API against your frozen answers (not needed on the server) |
| **`pokemon_data.json`** | **you add this** (built by `build_pokemon_data.py`) |
| **`card_template.npz`** | **you add this** (built with `pd.make_template`) |

## Use it

```
POST /read     form field "file" = PNG or JPG
GET  /health
```

```javascript
const fd = new FormData();
fd.append("file", fileInput.files[0]);
const res = await fetch("https://YOUR-SPACE.hf.space/read", { method: "POST", body: fd });
const card = await res.json();
```

Main fields of the answer:

```
status      "ok" | "review" | "not_found" | "error"
source      "cropped" | "screenshot"
name, level, gender ("M"/"F"/"N"), shiny, hp_max, types, ability, nature, region, hidden_power
stats       { ATK: {stat, iv, ev}, DEF: {...}, SPD: {...}, SPATK: {...}, SPDEF: {...}, HP: {stat: null, iv, ev} }
warnings    list of problems; show the values as editable when this is not empty
corrected   values the formula fixed automatically
```

`status: "ok"` means every check passed. `"review"` means it read the card but something failed a check
or was auto-corrected, so let the user confirm the values.

## Settings (environment variables, all optional)

| Name | Meaning |
|---|---|
| `ALLOWED_ORIGINS` | your website's address, e.g. `https://my-site.com` (default: any site) |
| `API_KEY` | if set, calls must send the header `X-API-Key`. A key inside browser code is visible to anyone, so it only stops casual misuse. |
| `MAX_MB`, `MAX_PIXELS`, `MAX_CONCURRENT` | upload limits, and how many images are read at the same time |

## Deploy on Hugging Face Spaces (free, Docker)

1. Create an account, then **New Space**. Pick **Docker** (blank template) and keep it public or private.
2. Copy `pokemon_data.json` and `card_template.npz` into this folder.
3. In the Space, open **Files, Add file, Upload files**, and upload everything in this folder, including this README.
4. The Space builds by itself (a few minutes). When it says "Running", open
   `https://YOUR-USERNAME-YOUR-SPACE.hf.space/health`. It should show `{"ok": true, ...}`.
5. In the Space's **Settings**, add `ALLOWED_ORIGINS` as a variable (your site's address).
6. Run `check_api.check(...)` from your notebook (see `check_api.py`). The server's Tesseract version can differ
   from Colab's, so this confirms the answers match.

Free tiers change: check the current limits (sleeping when idle, CPU, memory) before you rely on it.

## Run it on your own computer

```
pip install -r requirements.txt     # and install Tesseract itself
uvicorn app:app --port 8000          # then open http://localhost:8000/docs
```
