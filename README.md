# Text Summarizer (BART, Hugging Face Transformers)

Abstractive English text summarization with a pretrained BART encoder-decoder model.
Paste a text, get a short summary. Includes a command-line tool, a small Python API and a Gradio web demo.

## Features

- Abstractive summaries (the model writes new sentences, it does not just copy them)
- Two models: `facebook/bart-large-cnn` (best quality) and `sshleifer/distilbart-cnn-12-6` (smaller, faster)
- **Long-text support:** texts longer than the 1024-token model limit are split into sentence-aligned chunks, each chunk is summarized, then the partial summaries are summarized again (map-reduce)
- Adjustable summary length and beam-search width
- Works on transformers v4 and v5

## How it works

```
text -> sentences -> chunks (<= 900 tokens)
     -> tokenizer -> input_ids -> BART encoder -> BART decoder (model.generate, beam search)
     -> token ids -> tokenizer.decode -> summary
```

Note: `pipeline("summarization")` was removed in transformers v5, so this project calls
`AutoTokenizer`, `AutoModelForSeq2SeqLM` and `model.generate()` directly.

## Installation

Python 3.10+ is required.

```bash
git clone https://github.com/<your-username>/text-summarizer.git
cd text-summarizer
pip install -r requirements.txt
```

## Usage

**Command line**

```bash
python summarizer.py --file sample_article.txt
python summarizer.py --file sample_article.txt --model sshleifer/distilbart-cnn-12-6 --max-tokens 100
```

**Python**

```python
from summarizer import Summarizer

s = Summarizer("sshleifer/distilbart-cnn-12-6")
print(s.summarize(open("sample_article.txt").read()))
```

**Web demo**

```bash
python app.py
```

Then open the local URL printed in the terminal. In Google Colab the demo prints a public link.

**Google Colab**

```python
!git clone https://github.com/<your-username>/text-summarizer.git
%cd text-summarizer
!pip install -q -r requirements.txt
!python summarizer.py --file sample_article.txt
```

Use a GPU runtime (`Runtime -> Change runtime type -> T4 GPU`) for faster generation.

## Example

<!-- Replace this section with your real input/output and a screenshot of the demo -->

Input: `sample_article.txt`

Output:

```
(paste the summary your model produced here)
```

## Parameters

| Parameter | Meaning |
|---|---|
| `max_new_tokens` | maximum summary length (tokens) |
| `min_new_tokens` | minimum summary length (tokens) |
| `num_beams` | beam search width (higher = better, slower) |
| `no_repeat_ngram_size=3` | blocks repeated 3-word phrases (fixed in code) |

## Tests

```bash
pip install pytest
pytest
```

The tests cover the sentence splitting and chunking logic.

## Limitations

- English only. Both models were trained on CNN/DailyMail news, so they work best on news-style articles and less well on dialogue, code or very technical text.
- Abstractive models can state things that are not in the source text. Check important facts against the original.
- Very long texts are summarized in several passes, so details can be lost.

## Tech stack

Python, PyTorch, Hugging Face Transformers, Gradio

## License

MIT
