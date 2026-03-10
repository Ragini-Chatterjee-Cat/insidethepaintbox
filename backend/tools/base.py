"""Shared singletons for all artwork tools."""
from rag import collection, embedding_model

BASE_URL = "https://insidethepaintbox.netlify.app"
CONFIDENCE_THRESHOLD = 1.5

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
