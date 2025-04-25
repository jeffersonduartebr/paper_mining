import pandas as pd
from inclusion import meets_inclusion

class DummyRow:
    def __init__(self, year, abstract_pp, document_type):
        self.year = year
        self.abstract_pp = abstract_pp
        self.document_type = document_type

def test_meets_inclusion_all_criteria(monkeypatch):
    # Publicado há menos de 15 anos
    monkeypatch.setattr("inclusion.classifier", lambda text, labels: {"labels":[], "scores":[]})
    monkeypatch.setattr("inclusion.is_non_english", lambda t: False)

    row = DummyRow(year=2015, abstract_pp="texto", document_type="journal")
    assert meets_inclusion(row, threshold=0.5)  # ajusta assinatura conforme seu código
