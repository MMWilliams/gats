# QR codes — BIC Village · DEF CON 34

Regenerate with `python make_qr.py` (needs `segno`). All three are error-correction
level **H** (~30% recoverable), so they survive projector glare, an off-angle photo
from the back of the room, and a logo dropped in the center later.

| File | Points to | Use it on |
| --- | --- | --- |
| `qr-arxiv-paper.*` | `https://arxiv.org/abs/2607.08894` | Slide 26 (close) — the citation |
| `qr-gats-site.*` | `https://gats-site.vercel.app` | Slide 3 or 26 — explainer video + demo |
| `qr-portfolio.*` | `https://maureesewilliams.io` | Slide 26 — contact / everything else |

**SVG is the one to put in the deck** — it stays sharp at any projector resolution.
PNGs (~1200px) are the fallback for tools that won't take vector.

All three were verified by decoding the rendered PNGs back to their URLs, not just
generated — a QR that doesn't scan is a silent failure and you won't find out until
you're on stage.

## On stage

One QR per slide, minimum ~2 inches / 5 cm on the projected image. Three codes on one
slide means nobody scans any of them — if you want all three at the close, put the
portfolio code on the slide and let that page link to the other two.

## Note on the target list

The ask was "arxiv paper, personal portfolio, gats paper." The arXiv paper *is* the
GATS paper, so the third code points at the **GATS project site** (`gats-site.vercel.app`)
rather than duplicating the arXiv link. If the intent was a direct PDF instead, change
the `arxiv-paper` target to `https://arxiv.org/pdf/2607.08894` and rerun.
