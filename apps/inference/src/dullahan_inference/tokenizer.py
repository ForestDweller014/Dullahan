from __future__ import annotations

from threading import Lock

from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

from dullahan_inference.config import TokenizerConfig


class ModelTokenizerError(RuntimeError):
    pass


class ModelTokenizer:
    """Lazy, thread-safe access to the generation model's exact tokenizer."""

    def __init__(self, config: TokenizerConfig) -> None:
        self.config = config
        self._tokenizer: Tokenizer | None = None
        self._lock = Lock()

    def count(self, text: str) -> int:
        try:
            encoding = self._get_tokenizer().encode(
                text,
                add_special_tokens=self.config.add_special_tokens,
            )
        except Exception as exc:
            raise ModelTokenizerError(
                f"tokenizer {self.config.model!r} failed: {exc}"
            ) from exc
        return len(encoding.ids)

    def _get_tokenizer(self) -> Tokenizer:
        if self._tokenizer is not None:
            return self._tokenizer
        with self._lock:
            if self._tokenizer is None:
                tokenizer_path = hf_hub_download(
                    repo_id=self.config.model,
                    filename="tokenizer.json",
                    token=False,
                )
                self._tokenizer = Tokenizer.from_file(tokenizer_path)
        return self._tokenizer
