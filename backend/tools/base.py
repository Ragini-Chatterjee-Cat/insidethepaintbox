"""
Shared singletons and constants for the artwork tool classes.

Purpose: every tool in this package needs the same Chroma collection,
the same embedding function, and the same series-URL/commission text —
this file is where those live once instead of being redefined per tool.

Imported by: every file in tools/ (search_artworks.py, get_artwork_details.py,
filter_by_series.py, recommend_similar.py, commission_info.py,
reveal_secret.py). Not called directly — these are module-level values,
resolved once on first import and reused for the life of the process.
"""
from rag import collection, embed_query  # noqa: F401 - re-exported for tools/*.py

# --- shared values ------------------------------------------------------

BASE_URL = "https://insidethepaintbox.netlify.app"
CONFIDENCE_THRESHOLD = 1.5  # Chroma distance above this = "not a real match"

# Filenames here must match frontend/pages/series/ exactly - Netlify's
# static hosting is case-sensitive, and 6 of these 8 were previously wrong
# (e.g. "thoughts.html" for a file actually named "Thoughts.html"), silently
# handing visitors a 404 whenever the chatbot linked a series page.
SERIES_URLS = {
    "Portraits":        f"{BASE_URL}/pages/series/Portraits.html",
    "Animal Portraits": f"{BASE_URL}/pages/series/portraits-animals.html",
    "Mythical":         f"{BASE_URL}/pages/series/Mythical.html",
    "Thoughts":         f"{BASE_URL}/pages/series/Thoughts.html",
    "Camera Series":    f"{BASE_URL}/pages/series/Cam.html",
    "Diary Entries":    f"{BASE_URL}/pages/series/diary-entries.html",
    "Fanart":           f"{BASE_URL}/pages/series/Fanart.html",
    "Cards":            f"{BASE_URL}/pages/series/Cards.html",
}

COMMISSION_INFO = """
Ragini Chatterjee accepts commissions for custom artwork.

To request a commission:
- Contact via the website contact form: https://insidethepaintbox.netlify.app/pages/contact.html
- Or reach out on Instagram: @ragini_chatterjee

Commission types available:
- Portraits (people, pets, characters)
- Custom illustrations in any of the existing series styles
- Cards for special occasions

Please include reference photos and details about what you'd like when reaching out.
Response time is typically within a few days.
"""
