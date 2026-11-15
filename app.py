"""Gradio web demo: paste an English text, get a summary."""
import gradio as gr

from summarizer import DEFAULT_MODEL, LIGHT_MODEL, Summarizer

_models: dict[str, Summarizer] = {}


def get_model(name: str) -> Summarizer:
    if name not in _models:          # load each model only once
        _models[name] = Summarizer(name)
    return _models[name]


def run(text, model_name, max_new_tokens, min_new_tokens, num_beams):
    if not text or not text.strip():
        return "Please paste some text.", ""
    max_new_tokens, min_new_tokens, num_beams = int(max_new_tokens), int(min_new_tokens), int(num_beams)
    if min_new_tokens >= max_new_tokens:
        return "Min length must be smaller than max length.", ""
    summary = get_model(model_name).summarize(text, max_new_tokens, min_new_tokens, num_beams)
    n_in, n_out = len(text.split()), len(summary.split())
    stats = f"Input: {n_in} words | Summary: {n_out} words | Compression: {n_in / max(n_out, 1):.1f}x"
    return summary, stats


demo = gr.Interface(
    fn=run,
    inputs=[
        gr.Textbox(lines=14, label="Text (English)"),
        gr.Dropdown([DEFAULT_MODEL, LIGHT_MODEL], value=LIGHT_MODEL, label="Model"),
        gr.Slider(40, 300, value=130, step=10, label="Max summary tokens"),
        gr.Slider(10, 100, value=30, step=5, label="Min summary tokens"),
        gr.Slider(1, 8, value=4, step=1, label="Beams (more = better, slower)"),
    ],
    outputs=[gr.Textbox(label="Summary"), gr.Textbox(label="Stats")],
    title="Text Summarizer",
    description="Abstractive summarization with BART (Hugging Face Transformers). Best on news-style English text.",
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()
