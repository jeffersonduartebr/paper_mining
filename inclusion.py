from utils import published_within_last_n_years
from transformers import pipeline

threshold = 0.7
import re
import pandas as pd
from datetime import datetime
from transformers import pipeline

def apply_inclusion_filter(
    dados: pd.DataFrame,
    classifier,
    criterios_inclusao: dict,
    sublabels_7: list,
    threshold: float = 0.7
) -> pd.DataFrame:
    """
    Adiciona ao DataFrame 'dados' a coluna 'include' indicando True se o registro
    satisfizer todos os critérios de inclusão (com exceção do sub-critério 7, que
    requer pelo menos um dos sublabels).
    
    Parâmetros:
    - dados: DataFrame com colunas 'year', 'document_type' e 'abstract_pp'.
    - classifier: pipeline de zero-shot classification.
    - criterios_inclusao: dict[int, str] mapeando índices de critérios a labels.
    - sublabels_7: lista de strings com os sublabels do critério 7.
    - threshold: valor mínimo de confiança para considerar positivo.
    
    Retorna:
    - DataFrame original com a coluna 'include' adicionada.
    """
    # Definições estáticas de critérios 2, 3 e 9
    limit_year = datetime.now().year - 5
    #valid_types = {"journal article", "conference paper"}

    # Máscara inicial: filtros por ano e tipo de documento
    mask = (
        (dados["publication year"] >= limit_year) #&
        #(dados["document_type"].str.lower().isin(valid_types))
    )

    # Inicializa coluna como False
    dados["include"] = False

    # Subconjunto elegível
    df_sub = dados.loc[mask]
    texts = df_sub["abstract_pp"].tolist()

    if texts:
        # Critérios principais: 1, 4, 5, 6, 8
        main_labels = [
            criterios_inclusao[1],
            criterios_inclusao[4],
            criterios_inclusao[5],
            criterios_inclusao[6],
            criterios_inclusao[8]
        ]
        outs_main = classifier(texts, main_labels, multi_label=True)
        ok_main = [
            all(score >= threshold for score in out["scores"])
            for out in outs_main
        ]

        # Sub-critério 7: pelo menos um dos sublabels
        outs7 = classifier(texts, sublabels_7, multi_label=True)
        ok_7 = [
            any(score >= threshold for score in out["scores"])
            for out in outs7
        ]

        # Combina resultados principais e sub-critério 7
        ok_global = [m and s for m, s in zip(ok_main, ok_7)]
        dados.loc[mask, "include"] = ok_global

    return dados

