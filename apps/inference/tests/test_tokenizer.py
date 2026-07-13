from __future__ import annotations

from dullahan_inference.config import TokenizerConfig
from dullahan_inference.tokenizer import ModelTokenizer


class FakeEncoding:
    ids = [10, 20, 30]


class FakeTokenizer:
    def __init__(self) -> None:
        self.calls = []

    def encode(self, text: str, *, add_special_tokens: bool):
        self.calls.append((text, add_special_tokens))
        return FakeEncoding()


# Verifies exact token-ID counting while loading the external model tokenizer is mocked.
def test_model_tokenizer_counts_encoded_ids_and_reuses_instance(monkeypatch) -> None:
    fake = FakeTokenizer()
    loads = []

    def fake_hf_hub_download(*, repo_id: str, filename: str, token: bool):
        loads.append((repo_id, filename, token))
        return "/tmp/tokenizer.json"

    def fake_from_file(path: str):
        assert path == "/tmp/tokenizer.json"
        return fake

    monkeypatch.setattr(
        "dullahan_inference.tokenizer.hf_hub_download",
        fake_hf_hub_download,
    )
    monkeypatch.setattr(
        "dullahan_inference.tokenizer.Tokenizer.from_file",
        fake_from_file,
    )
    tokenizer = ModelTokenizer(
        TokenizerConfig(model="Qwen/test", add_special_tokens=False)
    )

    assert tokenizer.count("first") == 3
    assert tokenizer.count("second") == 3
    assert loads == [("Qwen/test", "tokenizer.json", False)]
    assert fake.calls == [("first", False), ("second", False)]
