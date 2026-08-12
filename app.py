import os
import difflib

import gradio as gr
import spaces
from google import genai
from transformers import DiffusionGemmaForBlockDiffusion, AutoProcessor

GEMMA4_MODEL_ID = "gemma-4-26b-a4b-it"
DIFFUSIONGEMMA_MODEL_ID = "google/diffusiongemma-26B-A4B-it"

_LEADING_ROLE_MARKERS = ("user", "model", "thought")


def _strip_leading_role_markers(text: str) -> str:
    """Safety net: if a chat-template role marker (e.g. a stray 'model' or 'thought')
    survives at the very start of the decoded text, drop it. Only strips whole leading
    words that exactly match a known marker, so normal sentences are never touched."""
    text = text.lstrip()
    changed = True
    while changed:
        changed = False
        for marker in _LEADING_ROLE_MARKERS:
            if text == marker:
                text = ""
                changed = True
            elif text.startswith(marker + " "):
                text = text[len(marker):].lstrip()
                changed = True
    return text


FIX_INSTRUCTION = (
    "The paragraph below contains exactly one factual or spelling error. "
    "Return the corrected paragraph only, with no explanation, no preamble, "
    "and no changes other than what is needed to fix that one error."
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
    """Yields (label, html) repeatedly as Gemma 4's answer streams in, word by word."""
    if not paragraph.strip():
        yield "### Gemma 4 (autoregressive)", ""
        return
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


@spaces.GPU(duration=90, size="xlarge")
def fix_with_diffusiongemma(paragraph: str) -> str:
    """Plain, blocking call — no streaming, no background thread. This is the
    exact pattern that ran in ~10 seconds before streaming was added."""
    messages = [{"role": "user", "content": f"{FIX_INSTRUCTION}\n\n{paragraph}"}]
    inputs = diff_processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt",
    ).to(diff_model.device)
    prompt_token_count = inputs["input_ids"].shape[-1]

    output = diff_model.generate(**inputs, max_new_tokens=512)
    generated_tokens = output[0][prompt_token_count:]
    text = diff_processor.decode(generated_tokens, skip_special_tokens=True)
    return _strip_leading_role_markers(text.strip())


def run_diffusiongemma(paragraph: str):
    if not paragraph.strip():
        return "### DiffusionGemma (block diffusion)", ""
    try:
        fix = fix_with_diffusiongemma(paragraph)
        html, pct = word_diff_html(paragraph, fix)
        return f"### DiffusionGemma (block diffusion) — {pct}% of words changed", html
    except Exception as e:
        return "### DiffusionGemma (block diffusion) — request failed", f"<p><em>{type(e).__name__}: {e}</em></p>"


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
        "Paste a paragraph with a mistake in it. Watch how an autoregressive model "
        "(Gemma 4) and a diffusion model (DiffusionGemma) each fix it — and see exactly "
        "how many words each one actually changed."
    )
    input_box = gr.Textbox(
        label="Paragraph with a mistake in it",
        lines=6,
        placeholder="Paste a paragraph that contains one planted error...",
    )
    run_button = gr.Button("Fix it with both models", variant="primary")

    with gr.Row():
        with gr.Column():
            gemma4_title = gr.Markdown("### Gemma 4 (autoregressive)")
            gemma4_output = gr.HTML()
        with gr.Column():
            diffusion_title = gr.Markdown("### DiffusionGemma (block diffusion)")
            diffusion_output = gr.HTML()

    run_button.click(
        fn=stream_gemma4,
        inputs=[input_box],
        outputs=[gemma4_title, gemma4_output],
    )
    run_button.click(
        fn=run_diffusiongemma,
        inputs=[input_box],
        outputs=[diffusion_title, diffusion_output],
    )

if __name__ == "__main__":
    demo.launch()
