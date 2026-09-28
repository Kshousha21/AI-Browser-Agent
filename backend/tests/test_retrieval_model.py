"""Model-free tests of the production token framing, pooling and vector validation."""
import numpy as np
import pytest
from tokenizers import Tokenizer, models, pre_tokenizers

from local_retrieval import RetrievalError
from local_retrieval.local_model import OnnxEmbedder
from local_retrieval.token_windows import windows


def tokenizer():
    result = Tokenizer(models.WordLevel({"<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3,
                                       "optical": 4, "cooling": 5, "agriculture": 6}, unk_token="<unk>"))
    result.pre_tokenizer = pre_tokenizers.Whitespace()
    return result


def test_full_token_stream_survives_beyond_first_window_and_padding_is_masked():
    text = "optical " * 300 + "cooling " * 130
    result = windows(tokenizer(), [text, "agriculture"])
    first = [window for owner, window in result if owner == 0]
    assert len(first) == 4
    content = []
    for window in first:
        count = sum(window.attention_mask)
        assert len(window.ids) == len(window.attention_mask) == len(window.type_ids) == 128
        assert window.ids[0] == 1 and window.ids[count - 1] == 2
        assert all(value == 0 for value in window.ids[count:])
        content.extend(window.ids[1:count - 1])
    assert content == [4] * 300 + [5] * 130
    assert sum(result[-1][1].attention_mask) == 3


class FakeSession:
    """Synthetic hidden features, never production model weights or inference."""
    def __init__(self, bad=None):
        self.bad = bad
        self.calls = 0

    def run(self, outputs, inputs):
        self.calls += 1
        assert all(value.dtype == np.int64 for value in inputs.values())
        ids = inputs["input_ids"]
        result = np.zeros((*ids.shape, 384), dtype=np.float32)
        for index in range(7):
            result[:, :, index] = ids == index
        if self.bad == "nan": result[:, :, 0] = np.nan
        if self.bad == "shape": result = result[:, :, :3]
        if self.bad == "zero": result[:] = 0
        return [result]


def local_fake(tmp_path, bad=None):
    model = OnnxEmbedder(tmp_path / "no-model-files")
    model._tokenizer = tokenizer()
    model._session = FakeSession(bad)
    return model


def test_pooling_masks_padding_weights_all_tokens_and_preserves_tail(tmp_path):
    model = local_fake(tmp_path)
    short = np.asarray(model.encode(["optical"])[0])
    assert short[0] == 0  # Masked padding contributes nothing.
    assert np.allclose(short[[1, 2, 4]], 1 / np.sqrt(3))
    prefix = "optical " * 300
    vectors = np.asarray(model.encode([prefix + "cooling " * 130, prefix + "agriculture " * 130]))
    assert vectors[0, 5] > 0 and vectors[0, 6] == 0
    assert vectors[1, 6] > 0 and vectors[1, 5] == 0
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1)
    assert model._session.calls == 2


@pytest.mark.parametrize("bad", ["nan", "shape", "zero"])
def test_invalid_inference_output_fails_closed(tmp_path, bad):
    with pytest.raises(RetrievalError, match="^invalid_embedding$"):
        local_fake(tmp_path, bad).encode(["optical"])


@pytest.mark.parametrize("texts", [[], [""], [" "], ["a" * 16001], ["a"] * 33])
def test_real_embedder_rejects_bad_input_before_loading(tmp_path, texts):
    model = OnnxEmbedder(tmp_path / "absent")
    with pytest.raises(RetrievalError, match="^invalid_arguments$"):
        model.encode(texts)
    assert model._session is None
