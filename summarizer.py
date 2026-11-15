"""
Abstractive text summarization with a Hugging Face encoder-decoder (seq2seq) model.

Why not pipeline("summarization")?
    The summarization/translation pipelines were removed in transformers v5,
    so this project uses AutoTokenizer + AutoModelForSeq2SeqLM + model.generate()
    directly. It works on transformers v4 and v5.

Long texts:
    BART-style models read at most 1024 tokens. Longer texts are split into
    sentence-aligned chunks, every chunk is summarized, and the partial
    summaries are summarized once more (map-reduce).
"""
import argparse
import re
import sys

DEFAULT_MODEL = "facebook/bart-large-cnn"      # best quality, ~1.6 GB
LIGHT_MODEL = "sshleifer/distilbart-cnn-12-6"  # smaller and faster
CHUNK_TOKENS = 900                              # safety margin below 1024


# ---------- pure helpers (no model needed, easy to test) ----------
def split_sentences(text: str) -> list[str]:
    """Split text into sentences on '.', '!' and '?' followed by whitespace."""
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    return re.split(r"(?<=[.!?])\s+", text)


def pack_sentences(sentences: list[str], lengths: list[int], max_tokens: int) -> list[str]:
    """Greedily group sentences into chunks of at most `max_tokens` tokens.

    A single sentence longer than `max_tokens` becomes its own chunk
    (it is truncated later by the tokenizer).
    """
    chunks, current, size = [], [], 0
    for sentence, n in zip(sentences, lengths):
        if current and size + n > max_tokens:
            chunks.append(" ".join(current))
            current, size = [], 0
        current.append(sentence)
        size += n
    if current:
        chunks.append(" ".join(current))
    return chunks


# ---------- the model wrapper ----------
class Summarizer:
    def __init__(self, model_name: str = DEFAULT_MODEL, device: str | None = None):
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
        self.model.eval()  # inference mode (no dropout)
        self.max_input = min(self.tok.model_max_length, 1024)

    def count_tokens(self, text: str) -> int:
        return len(self.tok(text, add_special_tokens=False)["input_ids"])

    def _generate(self, text: str, max_new_tokens: int, min_new_tokens: int, num_beams: int) -> str:
        batch = self.tok(
            text, truncation=True, max_length=self.max_input, return_tensors="pt"
        ).to(self.device)
        n_in = batch["input_ids"].shape[1]
        # do not force a long summary out of a very short input
        min_new = min(min_new_tokens, max(5, n_in // 2))
        max_new = max(max_new_tokens, min_new + 1)
        with self.torch.no_grad():
            ids = self.model.generate(
                **batch,
                max_new_tokens=max_new,
                min_new_tokens=min_new,
                num_beams=num_beams,
                no_repeat_ngram_size=3,
                length_penalty=2.0,
                early_stopping=True,
            )
        return self.tok.decode(ids[0], skip_special_tokens=True).strip()

    def summarize(
        self,
        text: str,
        max_new_tokens: int = 130,
        min_new_tokens: int = 30,
        num_beams: int = 4,
    ) -> str:
        sentences = split_sentences(text)
        if not sentences:
            raise ValueError("Empty text.")
        text = " ".join(sentences)
        kwargs = dict(max_new_tokens=max_new_tokens, min_new_tokens=min_new_tokens, num_beams=num_beams)

        # short text: one pass
        if self.count_tokens(text) <= self.max_input - 2:
            return self._generate(text, **kwargs)

        # long text: summarize chunks, then summarize the summaries
        lengths = [self.count_tokens(s) for s in sentences]
        chunks = pack_sentences(sentences, lengths, CHUNK_TOKENS)
        partial = [self._generate(c, **kwargs) for c in chunks]
        joined = " ".join(partial)
        if len(partial) > 1 and self.count_tokens(joined) <= self.max_input - 2:
            return self._generate(joined, **kwargs)
        return joined


# ---------- command line ----------
def main() -> None:
    p = argparse.ArgumentParser(description="Summarize English text.")
    p.add_argument("--file", help="text file to summarize (default: read stdin)")
    p.add_argument("--model", default=DEFAULT_MODEL, help=f"default: {DEFAULT_MODEL}")
    p.add_argument("--max-tokens", type=int, default=130, help="max summary length in tokens")
    p.add_argument("--min-tokens", type=int, default=30, help="min summary length in tokens")
    p.add_argument("--beams", type=int, default=4, help="beam search width")
    args = p.parse_args()

    if args.file:
        with open(args.file, encoding="utf-8") as f:
            text = f.read()
    else:
        text = sys.stdin.read()

    s = Summarizer(args.model)
    print(s.summarize(text, args.max_tokens, args.min_tokens, args.beams))


if __name__ == "__main__":
    main()
