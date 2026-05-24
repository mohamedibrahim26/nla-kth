# Natural Language Autoencoders — small-scale reimplementation

> Work in progress. Reimplementing the natural language autoencoder from Anthropic's
> *Natural Language Autoencoders Produce Unsupervised Explanations of LLM Activations*
> on a small, recent open model, for the KTH PhD recruitment task.

The full write-up (choices, results, figures) will live here. For now this README
tracks the pipeline as it is built.

## Pipeline

1. **Harvest activations** — `src/harvest_activations.py` runs a frozen target model
   over text and saves per-token residual-stream activations.
   Colab runner: [`notebooks/01_harvest_activations.ipynb`](notebooks/01_harvest_activations.ipynb).
2. *(next)* Generate teacher summaries for warm-start.
3. *(next)* Train the verbalizer (activation → English) and reconstructor (English → activation).
4. *(next)* Evaluate: FVE, controls, ablations, faithfulness, confabulation.

## Setup

```bash
pip install -r requirements.txt
```

## Target model

`Qwen/Qwen2.5-0.5B-Instruct` — recent, small enough for a free Colab T4, read at
layer 16 of 24 (~2/3 depth). See [`src/config.py`](src/config.py) for all settings.
