import os
import difflib
import threading

import gradio as gr
import spaces
from google import genai
from transformers import DiffusionGemmaForBlockDiffusion, AutoProcessor, TextDiffusionStreamer

GEMMA4_MODEL_ID = "gemma-4-26b-a4b-it"
DIFFUSIONGEMMA_MODEL_ID = "google/diffusiongemma-26B-A4B-it"

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
def stream_diffusiongemma(paragraph: str):
    """Yields (label, html) repeatedly as DiffusionGemma denoises the block in place."""
    if not paragraph.strip():
        yield "### DiffusionGemma (block diffusion)", ""
        return
    try:
        messages = [{"role": "user", "content": f"{FIX_INSTRUCTION}\n\n{paragraph}"}]
        inputs = diff_processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
        ).to(diff_model.device)

        streamer = TextDiffusionStreamer(tokenizer=diff_processor.tokenizer, skip_special_tokens=True)
        generation_thread = threading.Thread(
            target=diff_model.generate,
            kwargs=dict(**inputs, max_new_tokens=512, streamer=streamer),
        )
        generation_thread.start()

        for partial in streamer:
            html, pct = word_diff_html(paragraph, partial)
            yield f"### DiffusionGemma (block diffusion) — {pct}% of words changed so far", html

        generation_thread.join()
    except Exception as e:
        yield "### DiffusionGemma (block diffusion) — request failed", f"<p><em>{type(e).__name__}: {e}</em></p>"


with gr.Blocks(title="Diffusion Gemma Showdown") as demo:
    gr.Markdown(
        "# Diffusion Gemma Showdown\n"
        "Paste a paragraph with a mistake in it. Watch how an autoregressive model "
        "(Gemma 4) and a diffusion model (DiffusionGemma) each fix it live, side by side — "
        "and see exactly how many words each one actually changed."
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

    # Two separate listeners on the same click — Gradio runs both concurrently,
    # so each panel streams independently instead of waiting on the other.
    run_button.click(
        fn=stream_gemma4,
        inputs=[input_box],
        outputs=[gemma4_title, gemma4_output],
    )
    run_button.click(
        fn=stream_diffusiongemma,
        inputs=[input_box],
        outputs=[diffusion_title, diffusion_output],
    )

if __name__ == "__main__":
    demo.launch()
