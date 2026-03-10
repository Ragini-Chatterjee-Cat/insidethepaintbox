"""Artwork tools package for Inside the Paintbox."""
from .search_artworks import SearchArtworksTool
from .filter_by_series import FilterBySeriesTool
from .get_artwork_details import GetArtworkDetailsTool
from .commission_info import CommissionInfoTool
from .recommend_similar import RecommendSimilarTool

ARTWORK_TOOLS = [
    SearchArtworksTool(),
    FilterBySeriesTool(),
    GetArtworkDetailsTool(),
    CommissionInfoTool(),
    RecommendSimilarTool(),
]
