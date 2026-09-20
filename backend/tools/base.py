"""Shared singletons for all artwork tools."""
from rag import collection, embed_query

BASE_URL = "https://insidethepaintbox.netlify.app"
CONFIDENCE_THRESHOLD = 1.5

SERIES_URLS = {
    "Portraits":        f"{BASE_URL}/pages/series/portraits.html",
    "Animal Portraits": f"{BASE_URL}/pages/series/portraits-animals.html",
    "Mythical":         f"{BASE_URL}/pages/series/mythical.html",
    "Thoughts":         f"{BASE_URL}/pages/series/thoughts.html",
    "Camera Series":    f"{BASE_URL}/pages/series/cam.html",
    "Diary Entries":    f"{BASE_URL}/pages/series/diary-entries.html",
    "Fanart":           f"{BASE_URL}/pages/series/fanart.html",
    "Cards":            f"{BASE_URL}/pages/series/cards.html",
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
