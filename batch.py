import random
import time
from utils import log
import ollama

def call_local_llm(messages, model: str, temperature: float):
    response = ollama.chat(
        model=model,
        messages=messages,
        options={"temperature": temperature},
        stream=False
    )
    return response.message.content.strip()

def process_batch_with_retry(
    batch_df, global_index, query, model, temperature,
    seconds_between_requests, log_path, retry_limit=5
):
    retry_count = 0
    delay = seconds_between_requests + random.uniform(0, 5)

    while retry_count < retry_limit:
        try:
            if model == "cogito:8b":
                time.sleep(delay)
                messages = [
                    {"role": "system", "content": (
                        "Enable deep thinking subroutine."
                    )}
                ]
                prompt = f"{query}\n\nYou are a research assistant who helps analyze scientific articles. Restrict yourself to answering the question with exclusively 'yes' or 'no'.\n\n"
            else:
                time.sleep(delay)
                messages = [
                    {"role": "system", "content": (
                        "You are a research assistant who helps analyze scientific articles."
                    )}
                ]
                prompt = f"{query}\n\nRestrict yourself to answering the question with exclusively 'yes' or 'no'.\n\n"

            for i, row in batch_df.iterrows():
                prompt += f"Abstract {i + 1}:\n{row['abstract']}\n\n"

            messages.append({"role": "user", "content": prompt})

            content = call_local_llm(messages, model=model, temperature=temperature)
            answers = content.splitlines()

            results = []
            coluna = "relevant_" + model.split(":")[0]
            for answer, (_, row) in zip(answers, batch_df.iterrows()):
                result = row.to_dict()
                clean = answer.strip().lower()
                result[coluna] = (clean == "yes")
                results.append(result)

            log(f"[{model}] Lote {global_index} OK", log_path)
            return results

        except Exception as e:
            retry_count += 1
            wait_time = 2 ** retry_count + random.uniform(0, 1)
            log(f"[{model}][ERRO] Lote {global_index}, tentativa {retry_count}: {e}", log_path)
            time.sleep(wait_time)

    log(f"[{model}][FALHA] Lote {global_index} excedeu tentativas", log_path)
    return []