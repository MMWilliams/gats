"""Generate slide-ready QR codes for the DEF CON 34 GATS talk.

Outputs, per target: a scalable SVG (use this in the deck) and a 1200px PNG
(fallback for tools that won't take SVG). Error correction is set to H (~30%
recoverable) so the codes still scan off a projector at an angle, from the back
of the room, or with a logo dropped in the middle later.

Run:  python make_qr.py
"""

import segno

OUT = "."

# (slug, url, what it's for on stage)
TARGETS = [
    (
        "arxiv-paper",
        "https://arxiv.org/abs/2607.08894",
        "GATS paper on arXiv (abstract page - lets people grab PDF or cite)",
    ),
    (
        "portfolio",
        "https://maureesewilliams.io",
        "Personal portfolio / blog",
    ),
    (
        "gats-site",
        "https://gats-site.vercel.app",
        "GATS project site (explainer video + demo)",
    ),
]

# Dark-deck friendly: near-black module on white quiet zone. A QR needs the
# light/dark contrast to scan, so we keep a white card rather than inverting
# onto the dark slide background.
DARK = "#0b0b0f"
LIGHT = "#ffffff"

for slug, url, purpose in TARGETS:
    qr = segno.make(url, error="h")

    qr.save(
        f"{OUT}/qr-{slug}.svg",
        scale=10,
        border=4,
        dark=DARK,
        light=LIGHT,
    )

    qr.save(
        f"{OUT}/qr-{slug}.png",
        scale=40,          # -> ~1200-1500px depending on version
        border=4,
        dark=DARK,
        light=LIGHT,
    )

    print(f"{slug:14s} v{qr.version:<3} {url}")
    print(f"{'':14s} {purpose}")
