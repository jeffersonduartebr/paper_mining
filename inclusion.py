from utils import published_within_last_n_years
from transformers import pipeline

threshold = 0.7
def meets_inclusion(classifier, criterios_inclusao, sublabels_7,  row) -> bool:
    # 3) Publicado nos últimos 15 anos
    if not published_within_last_n_years(row["year"], 15):
        return False

    # 2 e 9) Artigos primários e revisados por pares
    tipo = row["document_type"].lower()
    if tipo not in {"journal article", "conference paper", "thesis", "dissertation"}:
        return False

    # 1, 4, 5, 6, 8 via zero-shot
    labels_main = [
        criterios_inclusao[1],
        criterios_inclusao[4],
        criterios_inclusao[5],
        criterios_inclusao[6],
        criterios_inclusao[8]
    ]
    out_main = classifier(row["abstract_pp"], labels_main)
    # exige que **todos** estejam acima do limiar
    for lbl, score in zip(out_main["labels"], out_main["scores"]):
        if score < threshold:
            return False

    # 7) pelo menos um dos sub-critérios
    out7 = classifier(row["abstract_pp"], sublabels_7)
    if not any(score >= threshold for score in out7["scores"]):
        return False

    return True