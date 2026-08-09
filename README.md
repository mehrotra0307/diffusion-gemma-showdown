---
title: Diffusion Gemma Showdown
emoji: 🥊
colorFrom: blue
colorTo: purple
sdk: gradio
app_file: app.py
pinned: false
---

# Diffusion Gemma Showdown

Two AI models. Same broken paragraph. One has to rewrite everything after the mistake — the other, in theory, only touches the broken words.

This project compares Gemma 4 (autoregressive — writes left to right, one token at a time) against DiffusionGemma (denoises a block of text in parallel) by planting an error in the middle of several paragraphs and asking both models to fix it. The result is a side-by-side diff view plus a measured stat: what percentage of words each model actually changed to fix a single mistake.

Status: work in progress.

## What's here

- `app.py` — the Hugging Face Space: paste a paragraph, see both models fix it live, side by side.
- `test_cases.json` — the paragraphs used for the batch experiment, each with one planted error.
- `run_experiment.py` — runs every test case through both models and saves the results.
- `analyze.py` — computes word-level diffs and % of words changed per model.
- `results/` — saved experiment output.

## Why

Built as a small, honest experiment (not just a visual trick) to learn how diffusion-based language models actually differ from standard autoregressive ones.
