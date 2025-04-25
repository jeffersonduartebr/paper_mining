import pytest
import time
from batch import process_batch_with_retry

class DummyError(Exception):
    pass

def fake_sleep(sec):
    # não pausa de verdade
    pass

def test_retry_then_success(monkeypatch):
    # força delay determinístico
    monkeypatch.setattr("random.uniform", lambda a, b: 0)
    monkeypatch.setattr(time, "sleep", fake_sleep)

    calls = {"n": 0}
    def fake_chat(model, messages):
        calls["n"] += 1
        if calls["n"] < 3:
            raise DummyError("temporário")
        return {"choices":[{"message":{"content":"OK"}}]}

    monkeypatch.setenv("OLLAMA_TOKEN", "dummy")
    monkeypatch.setattr("batch.ollama.chat", fake_chat)

    result = process_batch_with_retry(
        batch_df=None, global_index=0,
        query="Q", model="foo", temperature=0,
        seconds_between_requests=0.1, log_path="/tmp/log"
    )
    assert result == "OK"
    assert calls["n"] == 3
    