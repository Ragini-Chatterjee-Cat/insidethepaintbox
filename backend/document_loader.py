"""
Document Loader for Inside the Paintbox
Extracts artwork information from HTML files
"""

from bs4 import BeautifulSoup
from pathlib import Path
from urllib.parse import unquote
import re


# Maps series names to the artwork filenames that belong to them.
# Built from the links on each series page in /pages/series/.
SERIES_ARTWORK_MAP = {
    "Portraits": [
        "peeping-through-the-ivy.html",
        "krishna.html",
        "in-memory.html",
        "nailea-devora.html",
        "aishwarya-rai.html",
        "sydney-sweeney.html",
        "neymar.html",
        "harry-styles.html",
        "Byun.html",
    ],
    "Animal Portraits": [
        "lucky-panda.html",
        "tiger-cub.html",
        "an-imp-puppy.html",
        "just-chilling.html",
        "rawr.html",
        "hippo.html",
    ],
    "Mythical": [
        "R.html",
        "myth.html",
        "athena.html",
        "Draconic.html",
        "from-the-ashes.html",
        "head-in-the-clouds.html",
        "icarus.html",
    ],
    "Thoughts": [
        "whine.html",
        "hurt.html",
        "Trapped.html",
        "Voices.html",
        "Flowers.html",
        "behind-the-tiger.html",
    ],
    "Camera Series": [
        "camera-series.html",
    ],
    "Diary Entries": [
        "coffee.html",
        "chocolate.html",
        "strawberries.html",
        "dreaming.html",
        "pawprints.html",
    ],
    "Fanart": [
        "only-murders.html",
        "anora.html",
        "fellow-travellers.html",
        "normal-people.html",
    ],
    "Cards": [
        "congratulations.html",
        "happy-birthday.html",
    ],
}

# Build reverse lookup: artwork filename -> series name
_ARTWORK_TO_SERIES = {}
for series_name, artworks in SERIES_ARTWORK_MAP.items():
    for artwork_file in artworks:
        _ARTWORK_TO_SERIES[artwork_file.lower()] = series_name


def get_artwork_series(html_path):
    """Look up which series an artwork belongs to, based on its filename."""
    filename = Path(html_path).name.lower()
    return _ARTWORK_TO_SERIES.get(filename, "")


# Pages that exist but aren't linked from anywhere on the site. Not surfaced
# by ordinary search or browsing — only revealed if a visitor explicitly
# asks about something secret/hidden (see tools/reveal_secret.py).
SECRET_ARTWORKS = {
    "bibbity.html",
    "bare.html",
    "saree.html",
}


def is_secret_artwork(html_path):
    """Whether this artwork should be hidden from ordinary search/browsing."""
    return Path(html_path).name.lower() in SECRET_ARTWORKS


def clean_text(text):
    """Clean extracted text by removing extra whitespace"""
    if not text:
        return ""
    # Remove extra whitespace and newlines
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def get_artwork_url(html_path, website_path):
    """Generate the URL for an artwork page based on file path"""
    # Resolve both paths to absolute so relative_to works reliably
    html_path = Path(html_path).resolve()
    website_path = Path(website_path).resolve()

    # Get relative path from website root
    try:
        relative_path = html_path.relative_to(website_path)
    except ValueError:
        relative_path = html_path.name

    # Convert to URL path (forward slashes, URL encoded)
    url_path = str(relative_path).replace("\\", "/")

    # Base URL - update this to your actual Netlify URL
    base_url = "https://insidethepaintbox.netlify.app"

    return f"{base_url}/{url_path}"


def load_artwork_from_html(html_path, website_path="../"):
    """Extract artwork information from a single HTML file"""
    try:
        with open(html_path, 'r', encoding='utf-8') as f:
            soup = BeautifulSoup(f.read(), 'html.parser')

        # Extract title from h1. This site nests <h6> (the subtitle/dimensions)
        # inside <h1>, so h1.get_text() would silently absorb the subtitle
        # into the title (e.g. "PEEPING THROUGH THE IVY CUSTOM PORTRAIT").
        # Pull the subtitle's text out first, then remove it from the tree
        # before reading the title, so each stays what it says it is.
        title_elem = soup.find('h1')
        nested_subtitle = title_elem.find('h6') if title_elem else None
        if nested_subtitle:
            subtitle = clean_text(nested_subtitle.get_text())
            nested_subtitle.extract()
        else:
            subtitle_elem = soup.find('h6')
            subtitle = clean_text(subtitle_elem.get_text()) if subtitle_elem else ""

        title = clean_text(title_elem.get_text()) if title_elem else ""

        # Extract description from paragraphs
        paragraphs = soup.find_all('p')
        description = " ".join([clean_text(p.get_text()) for p in paragraphs])

        # Extract any h2 headers (like "INSPIRATION")
        headers = soup.find_all('h2')
        section_titles = " ".join([clean_text(h.get_text()) for h in headers])

        # Locate the page's main image, if any, for multimodal embedding.
        # src may be URL-encoded (e.g. "%20" for a space in the filename).
        img_elem = soup.find('img')
        image_path = None
        if img_elem and img_elem.get('src'):
            candidate = (Path(html_path).parent / unquote(img_elem['src'])).resolve()
            if candidate.exists():
                image_path = str(candidate)

        # Generate URL for this artwork
        url = get_artwork_url(html_path, website_path)

        # Look up series membership
        series = get_artwork_series(html_path)
        series_line = f"\nSeries: {series}" if series else ""

        # Combine all content (include URL so chatbot knows it)
        full_content = f"""
Artwork: {title}
{subtitle}{series_line}
URL: {url}

{description}
        """.strip()

        return {
            "title": title,
            "subtitle": subtitle,
            "description": description,
            "series": series,
            "content": full_content,
            "source": str(html_path),
            "url": url,
            "image_path": image_path,
            "secret": is_secret_artwork(html_path),
        }
    except Exception as e:
        print(f"Error loading {html_path}: {e}")
        return None


def load_all_artworks(website_path):
    """Load all artwork documents from the website"""
    documents = []
    website_path = Path(website_path)

    # Look for artwork HTML files
    artwork_paths = [
        website_path / "artworks",
    ]

    for artwork_dir in artwork_paths:
        if artwork_dir.exists():
            for html_file in artwork_dir.glob("**/*.html"):
                doc = load_artwork_from_html(html_file, website_path)
                if doc and doc["content"]:
                    documents.append(doc)
                    print(f"Loaded: {doc['title']} -> {doc.get('url', 'no url')}")

    # Also load series pages for additional context
    series_path = website_path / "pages" / "series"
    if series_path.exists():
        for html_file in series_path.glob("*.html"):
            doc = load_artwork_from_html(html_file, website_path)
            if doc and doc["content"]:
                documents.append(doc)
                print(f"Loaded series: {doc['title']}")

    return documents


def load_about_page(website_path):
    """Load the about page for artist information"""
    about_path = Path(website_path) / "pages" / "about.html"
    if about_path.exists():
        doc = load_artwork_from_html(about_path, website_path)
        if doc:
            doc["title"] = "About the Artist"
            return doc
    return None


if __name__ == "__main__":
    # Test the loader
    docs = load_all_artworks("../frontend/")
    print(f"\nLoaded {len(docs)} documents")
    for doc in docs[:3]:
        print(f"\n--- {doc['title']} ---")
        print(doc['content'][:200] + "...")
