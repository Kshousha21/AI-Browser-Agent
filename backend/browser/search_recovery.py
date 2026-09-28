"""Generic empty-search guidance; never invent site-specific query URLs."""
def empty_search(page):
    return False  # No universal DOM evidence for an empty result set.

def validate_empty_search_retry(goal, page, call):
    return None

EMPTY_SEARCH_INSTRUCTION = "Use the observed search input with concise topic terms, then inspect the results."
