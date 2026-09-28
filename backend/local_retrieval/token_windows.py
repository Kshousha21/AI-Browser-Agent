"""Explicit bounded windows after complete tokenization; no tokenizer truncation."""
from dataclasses import dataclass

from .errors import RetrievalError


@dataclass(frozen=True, repr=False)
class TokenWindow:
    ids: list[int]
    attention_mask: list[int]
    type_ids: list[int]


def windows(tokenizer, texts: list[str], max_tokens: int = 128) -> list[tuple[int, TokenWindow]]:
    """Pinned XLM-R token framing: <s> content </s> then masked <pad>.

    Tokenizers 0.23 can stop pre-tokenization when truncation is enabled, so
    relying on Encoding.overflowing loses text. Tokenize the bounded full input
    without truncation/padding, then partition token IDs ourselves.
    """
    start, end, pad = [tokenizer.token_to_id(value) for value in ("<s>", "</s>", "<pad>")]
    if any(value is None for value in (start, end, pad)):
        raise RetrievalError("embedding_model_invalid")
    output = []
    width = max_tokens - 2
    for owner, encoding in enumerate(tokenizer.encode_batch(texts, add_special_tokens=False)):
        ids = encoding.ids
        if not ids or len(ids) > width * 256:
            raise RetrievalError("embedding_failed")
        for offset in range(0, len(ids), width):
            framed = [start, *ids[offset:offset + width], end]
            count = len(framed)
            output.append((owner, TokenWindow(framed + [pad] * (max_tokens - count),
                                              [1] * count + [0] * (max_tokens - count), [0] * max_tokens)))
    return output
