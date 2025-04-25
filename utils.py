import re
import pandas as pd
from langdetect import detect
from transformers import pipeline
from datetime import datetime
import time

# ----------------------------
# Funções utilitárias
# ----------------------------
def preprocess_text(text: str) -> str:
    """Remove HTML, referências numéricas e normaliza espaços."""
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'\[\d+\]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

def published_within_last_n_years(year: int, n: int = 15) -> bool:
    """Verifica se 'year' está nos últimos n anos."""
    return year >= datetime.now().year - n

def is_non_english(text: str) -> bool:
    """Detecta se o texto NÃO está em inglês (critério QE5)."""
    try:
        return detect(text) != 'en'
    except:
        return False
    
def classify_abstract(criterios_exclusao: dict[str,str], classifier: pipeline, labels: list[str], text: str, threshold: float = 0.7) -> dict[str, bool]:
    """Retorna um dict {QEx: bool} para QE1, QE2, QE3, QE4, QE6."""
    out = classifier(text, labels)
    result: dict[str, bool] = {}
    for label, score in zip(out["labels"], out["scores"]):
        # encontra a chave QE correspondente
        key = next(k for k, v in criterios_exclusao.items() if v == label)
        result[key] = score >= threshold
    return result

def log(text: str, log_path: str) -> None:
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"{timestamp} - {text}\n")
    print(f"{timestamp} - {text}")

def load_and_filter_bases(directory: str) -> pd.DataFrame:
    print(f"Carregando bases de '{directory}'")
    dfs = []
    for fname in sorted(os.listdir(directory)):
        if fname.lower().endswith('.csv'):
            path = os.path.join(directory, fname)
            try:
                df = pd.read_csv(path)
                print(f"{fname}: {len(df)} registros")
                dfs.append(df)
            except Exception as e:
                print(f"Erro lendo {fname}: {e}")
    if not dfs:
        return pd.DataFrame()
    df = pd.concat(dfs, ignore_index=True)
    return df    