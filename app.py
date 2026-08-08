import os
import difflib

import gradio as gr
import spaces
from google import genai
from transformers import DiffusionGemmaForBlockDiffusion, AutoProcessor

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


@spaces.GPU(duration=90, size="xlarge")
def fix_with_diffusiongemma(paragraph: str) -> str:
    messages = [{"role": "user", "content": f"{FIX_INSTRUCTION}\n\n{paragraph}"}]
    inputs = diff_processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt",
    ).to(diff_model.device)
    output = diff_model.generate(**inputs, max_new_tokens=512)
    text = diff_processor.decode(output[0], skip_special_tokens=True)
    return text.strip()


def fix_with_gemma4(paragraph: str) -> str:
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    response = client.models.generate_content(
        model=GEMMA4_MODEL_ID,
        contents=f"{FIX_INSTRUCTION}\n\n{paragraph}",
    )
    return response.text.strip()


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


def run_comparison(paragraph: str):
    gemma4_fix = fix_with_gemma4(paragraph)
    diffusion_fix = fix_with_diffusiongemma(paragraph)

    gemma4_html, gemma4_pct = word_diff_html(paragraph, gemma4_fix)
    diffusion_html, diffusion_pct = word_diff_html(paragraph, diffusion_fix)

    gemma4_label = f"### Gemma 4 (autoregressive) — {gemma4_pct}% of words changed"
    diffusion_label = f"### DiffusionGemma (block diffusion) — {diffusion_pct}% of words changed"

    return gemma4_label, gemma4_html, diffusion_label, diffusion_html


with gr.Blocks(title="Diffusion Gemma Showdown") as demo:
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
        fn=run_comparison,
        inputs=[input_box],
        outputs=[gemma4_title, gemma4_output, diffusion_title, diffusion_output],
    )

if __name__ == "__main__":
    demo.launch()
