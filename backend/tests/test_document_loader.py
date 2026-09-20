"""Tests for document_loader.py - HTML parsing into indexable documents."""
from pathlib import Path

from document_loader import (
    clean_text,
    get_artwork_series,
    get_artwork_url,
    is_secret_artwork,
    load_artwork_from_html,
)

ARTWORK_HTML = """<!DOCTYPE html>
<html><head><title>x</title></head>
<body>
<h1>PEEPING THROUGH THE IVY
    <h6>CUSTOM PORTRAIT</h6>
</h1>
<img src="../../images/artworks/Portraits/Custom/Peeping.jpg" alt="Peeping" id="cub" />
<h2>-INSPIRATION-</h2>
<p>This is a portrait of the artist's mother.</p>
</body></html>
"""

MULTI_IMAGE_HTML = """<!DOCTYPE html>
<html><body>
<h1>HEAD IN THE CLOUDS
    <h6>MYTHICAL SERIES</h6>
</h1>
<img src="HIC1.jpeg" alt="1" />
<img src="HIC%201%20copy.jpeg" alt="2 encoded" />
<img src="missing.jpeg" alt="does not exist" />
<p>Ovid's Metamorphoses.</p>
</body></html>
"""


def _write(tmp_path: Path, name: str, html: str, image_names: list[str] = ()) -> Path:
    html_path = tmp_path / name
    html_path.write_text(html, encoding="utf-8")
    for img_name in image_names:
        (tmp_path / img_name).write_bytes(b"fake-image-bytes")
    return html_path


def test_clean_text_collapses_whitespace():
    assert clean_text("  a\n\n  b   c\t") == "a b c"
    assert clean_text(None) == ""
    assert clean_text("") == ""


def test_get_artwork_series_matches_known_filenames():
    assert get_artwork_series("some/path/icarus.html") == "Mythical"
    assert get_artwork_series("some/path/ICARUS.HTML".lower()) == "Mythical"
    assert get_artwork_series("some/path/not-a-real-artwork.html") == ""


def test_is_secret_artwork():
    assert is_secret_artwork("frontend/artworks/misc/Bibbity.html") is True
    assert is_secret_artwork("frontend/artworks/misc/bibbity.html") is True
    assert is_secret_artwork("frontend/artworks/mythical/icarus.html") is False


def test_get_artwork_url_builds_public_url(tmp_path):
    website_root = tmp_path / "frontend"
    html_path = website_root / "artworks" / "mythical" / "icarus.html"
    html_path.parent.mkdir(parents=True)
    html_path.write_text("<h1>ICARUS</h1>")

    url = get_artwork_url(html_path, website_root)
    assert url == "https://insidethepaintbox.netlify.app/artworks/mythical/icarus.html"


def test_title_does_not_absorb_nested_subtitle(tmp_path):
    """Regression test: <h6> is nested inside <h1> on every artwork page,
    which used to make the title silently absorb the subtitle text (e.g.
    "PEEPING THROUGH THE IVY CUSTOM PORTRAIT"). This was the root cause
    of a real reported bug where the chatbot described the wrong artwork."""
    html_path = _write(
        tmp_path, "peeping.html", ARTWORK_HTML,
        image_names=["Peeping.jpg"],
    )
    # image src is a relative path that needs the real directory structure
    (tmp_path / "images" / "artworks" / "Portraits" / "Custom").mkdir(parents=True)
    (tmp_path / "images" / "artworks" / "Portraits" / "Custom" / "Peeping.jpg").write_bytes(b"x")

    doc = load_artwork_from_html(html_path, tmp_path)

    assert doc["title"] == "PEEPING THROUGH THE IVY"
    assert doc["subtitle"] == "CUSTOM PORTRAIT"
    # The subtitle text must not also be duplicated inside the title.
    assert "CUSTOM PORTRAIT" not in doc["title"]


def test_load_artwork_collects_all_images_and_decodes_urls(tmp_path):
    """Regression test: previously only the first <img> was collected, and
    a %20-encoded src (a real case for several artwork files) failed to
    resolve because the path wasn't URL-decoded before checking existence."""
    html_path = tmp_path / "hic.html"
    html_path.write_text(MULTI_IMAGE_HTML, encoding="utf-8")
    (tmp_path / "HIC1.jpeg").write_bytes(b"x")
    (tmp_path / "HIC 1 copy.jpeg").write_bytes(b"x")  # real filename has a space
    # "missing.jpeg" deliberately not created

    doc = load_artwork_from_html(html_path, tmp_path)

    assert len(doc["image_paths"]) == 2
    assert any(p.endswith("HIC1.jpeg") for p in doc["image_paths"])
    assert any(p.endswith("HIC 1 copy.jpeg") for p in doc["image_paths"])
    assert not any("missing" in p for p in doc["image_paths"])


def test_load_artwork_returns_none_for_unreadable_file(tmp_path):
    missing_path = tmp_path / "does-not-exist.html"
    assert load_artwork_from_html(missing_path, tmp_path) is None


def test_root_relative_image_src_resolves_against_site_root(tmp_path):
    """Regression test: Path's / operator discards the left operand
    entirely when the right side looks absolute, so a site-root-relative
    src like "/images/foo.png" used to resolve to a filesystem-root path
    that never exists, silently dropping the image."""
    website_root = tmp_path / "frontend"
    (website_root / "images").mkdir(parents=True)
    (website_root / "images" / "foo.png").write_bytes(b"x")

    html_path = website_root / "artworks" / "misc" / "page.html"
    html_path.parent.mkdir(parents=True)
    html_path.write_text('<h1>TITLE</h1><img src="/images/foo.png">')

    doc = load_artwork_from_html(html_path, website_root)

    assert len(doc["image_paths"]) == 1
    assert doc["image_paths"][0].endswith("images/foo.png")
