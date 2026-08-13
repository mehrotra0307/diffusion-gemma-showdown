import os
import re
import threading
import queue
import difflib

import gradio as gr
import spaces
from google import genai
from transformers import DiffusionGemmaForBlockDiffusion, AutoProcessor, TextDiffusionStreamer

GEMMA4_MODEL_ID = "gemma-4-26b-a4b-it"
DIFFUSIONGEMMA_MODEL_ID = "google/diffusiongemma-26B-A4B-it"

_STOP = object()
_LEADING_ROLE_MARKERS = ("user", "model", "thought")
_ROLE_MARKER_PATTERN = re.compile(
    r"^(?:\s*(?:" + "|".join(_LEADING_ROLE_MARKERS) + r")\b[:\s]*)+",
    re.IGNORECASE,
)


def _strip_leading_role_markers(text: str) -> str:
    """Safety net: if a chat-template role marker (e.g. a stray 'model' or 'thought')
    survives at the very start of the decoded text, drop it. Matches the marker word
    regardless of what whitespace/punctuation follows it (space, newline, colon,
    stacked markers), so formatting variations can't slip past this like before."""
    return _ROLE_MARKER_PATTERN.sub("", text).lstrip()


class QueueDiffusionStreamer(TextDiffusionStreamer):
    """TextDiffusionStreamer is built to print ANSI colors to a terminal, not to
    hand data back to code — it isn't iterable at all. This subclass intercepts
    its two real callbacks (put_draft for in-progress denoising, on_finalized_text
    for confirmed text) and pushes plain strings into a queue instead, so we can
    read it like a normal streaming iterator from app.py. This is the version that
    already worked correctly and quickly before — no timeout wrapper on top of it,
    since that wrapper (added later, for Gemma 4) was the thing that broke."""

    def __init__(self, tokenizer, prompt_token_count: int = 0, **kwargs):
        super().__init__(tokenizer, **kwargs)
        self.text_queue = queue.Queue()
        self._confirmed = ""
        self.prompt_token_count = prompt_token_count

    def put_draft(self, value, **kwargs):
        if len(value.shape) > 1 and value.shape[0] > 1:
            raise ValueError("QueueDiffusionStreamer only supports batch size 1")
        elif len(value.shape) > 1:
            value = value[0]
        value = value[self.prompt_token_count:]
        draft_text = self.tokenizer.decode(value, skip_special_tokens=True)
        self.text_queue.put(_strip_leading_role_markers(draft_text))

    def on_finalized_text(self, text: str, stream_end: bool = False):
        self._confirmed += text
        self.text_queue.put(_strip_leading_role_markers(self._confirmed))
        if stream_end:
            self.text_queue.put(_STOP)

    def __iter__(self):
        return self

    def __next__(self):
        value = self.text_queue.get()
        if value is _STOP:
            raise StopIteration
        return value


FIX_INSTRUCTION = (
    "The paragraph below contains one or more factual or spelling errors. "
    "Return the corrected paragraph only, with no explanation, no preamble, "
    "and no changes other than what is needed to fix those errors."
)

# Loaded once when the Space starts up, kept warm — not reloaded per request.
diff_model = DiffusionGemmaForBlockDiffusion.from_pretrained(
    DIFFUSIONGEMMA_MODEL_ID, dtype="auto", device_map="auto",
)
diff_processor = AutoProcessor.from_pretrained(DIFFUSIONGEMMA_MODEL_ID)


def word_diff_html(original: str, fixed: str):
    orig_words = original.split()
    fixed_words = fixed.split()
    matcher = difflib.SequenceMatcher(a=orig_words, b=fixed_words)
    html_parts = []
    changed = 0
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            html_parts.append(" ".join(fixed_words[j1:j2]))
        else:
            changed += max(i2 - i1, j2 - j1)
            replaced = " ".join(fixed_words[j1:j2])
            if replaced:
                html_parts.append(f"<mark>{replaced}</mark>")
    pct = round(100 * changed / max(len(orig_words), 1), 1)
    return " ".join(html_parts), pct


def stream_gemma4(paragraph: str):
    """Yields (label, html) repeatedly as Gemma 4's answer streams in. No timeout
    wrapper this time — that was the piece that actually broke last time."""
    if not paragraph.strip():
        yield "### Gemma 4 (autoregressive)", ""
        return
    yield "### Gemma 4 (autoregressive) — thinking…", "<p><em>Waiting for a response…</em></p>"
    try:
        client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        partial = ""
        for chunk in client.models.generate_content_stream(
            model=GEMMA4_MODEL_ID,
            contents=f"{FIX_INSTRUCTION}\n\n{paragraph}",
        ):
            if chunk.text:
                partial += chunk.text
                html, pct = word_diff_html(paragraph, partial)
                yield f"### Gemma 4 (autoregressive) — {pct}% of words changed so far", html
    except Exception as e:
        yield "### Gemma 4 (autoregressive) — request failed", f"<p><em>{type(e).__name__}: {e}</em></p>"


@spaces.GPU(duration=60, size="xlarge")
def stream_diffusiongemma(paragraph: str):
    """Yields (label, html) repeatedly as DiffusionGemma denoises the block in place."""
    if not paragraph.strip():
        yield "### DiffusionGemma (block diffusion)", ""
        return
    yield "### DiffusionGemma (block diffusion) — thinking…", "<p><em>Waiting for a response…</em></p>"
    try:
        messages = [{"role": "user", "content": f"{FIX_INSTRUCTION}\n\n{paragraph}"}]
        inputs = diff_processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
        ).to(diff_model.device)
        prompt_token_count = inputs["input_ids"].shape[-1]

        streamer = QueueDiffusionStreamer(
            tokenizer=diff_processor.tokenizer,
            prompt_token_count=prompt_token_count,
            skip_prompt=True,
            skip_special_tokens=True,
        )
        generation_thread = threading.Thread(
            target=diff_model.generate,
            kwargs=dict(**inputs, max_new_tokens=160, streamer=streamer),
        )
        generation_thread.start()

        for partial in streamer:
            html, pct = word_diff_html(paragraph, partial)
            yield f"### DiffusionGemma (block diffusion) — {pct}% of words changed so far", html

        generation_thread.join()
    except Exception as e:
        yield "### DiffusionGemma (block diffusion) — request failed", f"<p><em>{type(e).__name__}: {e}</em></p>"


CUSTOM_CSS = """
mark {
    background-color: #ff8a00;
    color: #000000;
    font-weight: 700;
    padding: 0.05em 0.25em;
    border-radius: 3px;
}
"""

with gr.Blocks(title="Diffusion Gemma Showdown", css=CUSTOM_CSS) as demo:
    gr.Markdown(
        "# Diffusion Gemma Showdown\n"
        "Paste a paragraph with mistakes in it. Watch how an autoregressive model "
        "(Gemma 4) and a diffusion model (DiffusionGemma) each fix it live, side by side — "
        "and see exactly how many words each one actually changed."
    )
    input_box = gr.Textbox(
        label="Paragraph with a mistake in it",
        lines=6,
        placeholder="Paste a paragraph that contains one or more planted errors...",
    )
    run_button = gr.Button("Fix it with both models", variant="primary")

    with gr.Row():
        with gr.Column():
            gemma4_title = gr.Markdown("### Gemma 4 (autoregressive)")
            gemma4_output = gr.HTML()
        with gr.Column():
            diffusion_title = gr.Markdown("### DiffusionGemma (block diffusion)")
            diffusion_output = gr.HTML()

    # Two separate listeners on the same click, run concurrently (see demo.queue()
    # below) so each panel streams independently instead of waiting on the other.
    # show_progress="hidden" turns off Gradio's own spinner+seconds overlay — the
    # "thinking…" message yielded above is our own stand-in progress indicator.
    run_button.click(
        fn=stream_gemma4,
        inputs=[input_box],
        outputs=[gemma4_title, gemma4_output],
        show_progress="hidden",
    )
    run_button.click(
        fn=stream_diffusiongemma,
        inputs=[input_box],
        outputs=[diffusion_title, diffusion_output],
        show_progress="hidden",
    )

if __name__ == "__main__":
    # Let both listeners actually run at the same time instead of Gradio's
    # default of one queued event at a time app-wide.
    demo.queue(default_concurrency_limit=2)
    demo.launch()
