import pytest
from utils import preprocess_text, is_non_english

def test_preprocess_text_remove_html_and_refs():
    txt = "<p>Hello</p> [12] world"
    assert preprocess_text(txt) == "Hello world"

def test_preprocess_text_normalize_spaces():
    txt = "Hello   \n world"
    assert preprocess_text(txt) == "Hello world"

@pytest.mark.parametrize("text, expected", [
    ("This is English.", False),
    ("C'est un texte en français.", True),
])
def test_is_non_english(text, expected):
    assert is_non_english(text) is expected
