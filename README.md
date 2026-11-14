# Controlling and Assessing Appropriate Persona Use in LLM-based Dialogue Generation

Code for **SCONPOS** (Self-CONtrastive Persona Overuse Suppression) and **PAS** (Persona
Appropriateness Score), proposed in *"Controlling and Assessing Appropriate Persona Use in
LLM-based Dialogue Generation"* by Jongkyung Shin<sup>1</sup>, Inkyu Lee<sup>2</sup>, and
Chiehyeon Lim<sup>1,2,3</sup> (<sup>1</sup>UNIST, <sup>2</sup>POSTECH, <sup>3</sup>POSCO
Holdings).

- **SCONPOS** mitigates *persona overuse* in persona-based dialogue generation (PDG) by
  extracting a steering vector from the difference between "all persona attributes given"
  and "only topic-relevant persona attributes given" prompts, and subtracting it from the
  model's hidden states at inference time — without any response generation or parameter
  updates.
- **PAS** is a Jensen-Shannon-distance-based metric that scores how well a response's topic
  distribution matches the dialogue-relevant persona topics implied by the partner's last
  utterance, penalizing both persona *overuse* and *underuse*.

This repository contains the pipeline needed to reproduce the paper's main results: dataset
construction/topic labeling, SCONPOS steering-vector extraction and inference, the ICL
baselines (Vanilla / CoT / Task Decomposition / Self-Refine), and PAS + the other automatic
metrics (Overuse, C-Score, P-Dist, P-F1, BLEU, ROUGE-L, BERTScore). `ablations/` contains the
driver scripts for three follow-up analyses reported in the appendix.

## Repository layout

```
config.py                    # central CLI argument definitions
main_test.py                  # SCONPOS: steering-vector extraction, inference (baselines +
                              # SCONPOS), and metric computation (Overuse/C-Score/P-Dist/P-F1/
                              # BLEU/ROUGE-L/BERTScore/PAS)
gpt_inference.py               # GPT-4.1 baseline generation via the OpenAI Batch API
processing.py                  # dataset construction + GPT-based topic/persona labeling
                              # (PersonaChat, MBTI-S2Conv)

evaluate_pas.py                # standalone PAS evaluation CLI (NumPy only; see "PAS" below)

utility/pas.py                # PAS implementation (persona_appropriateness_score, corpus_pas)
utility/utils.py              # other metrics: Overuse (ovs_eval), C-Score, P-Dist, P-F1, BLEU
prompts/
  PersonaChat/{self,partner}/{sub_prompt.py,main_prompt.py}
  MBTI_S2Conv/{self,partner}/{sub_prompt.py,main_prompt.py}
                              # prompt construction for vanilla / no-persona / CoT / task
                              # decomposition / self-refine settings, for both datasets

ablations/
  main_test_multi_layer.py               # extract the steering vector jointly across all
                                          # decoder layers ("all") or a 3-layer window around
                                          # the midpoint ("mid3"), instead of a single layer
  main_test_small_train_ratio.py         # extract the steering vector from a 5/10/25/50%
                                          # subsample of the contrastive training pairs
  main_test_certain_persona_matched.py   # control condition: contrast "All" against "Match"
                                          # with duplicated relevant attributes so the two
                                          # prompts have matched attribute count / length
                                          # ("All, Match+dup." in the appendix)

inference.sh, inference_gpt.sh, processing.sh   # example CLI invocations
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"
```

Python 3.10+ is recommended. GPU inference is expected for `main_test.py` (Llama-3.1-8B /
Qwen2.5-7B / DeepSeek-LLM-7B-chat via 🤗 Transformers).

### External assets you need to obtain separately

These are not included in this repository due to their size and license terms:

| Asset | Used by | Notes |
|---|---|---|
| `./GoogleNews-vectors-negative300.bin` | P-Dist metric (`utility/utils.py`) | [Google's pretrained word2vec vectors](https://code.google.com/archive/p/word2vec/) (~3.6GB) |
| `./nli/model/` | C-Score metric (`utility/utils.py`) | `bert-large` fine-tuned on the [Dialogue NLI dataset](https://wellecks.com/dialogue_nli/) (Welleck et al., 2019); train your own or substitute any 3-way (entail/neutral/contradict) NLI classifier saved in the HF `BertForSequenceClassification` format |
| Llama-3.1-8B-Instruct / Qwen2.5-7B-Instruct / deepseek-llm-7b-chat | `main_test.py` | pulled automatically from the Hugging Face Hub on first run |
| `gpt-4.1-2025-04-14` (via `OPENAI_API_KEY`) | `gpt_inference.py`, `processing.py`'s GPT-based topic/persona labeling | requires an OpenAI API key with Batch API access |

### Dataset preparation

The paper uses two datasets:

- **PersonaChat** ([Zhang et al., 2018](https://github.com/facebookresearch/ParlAI/tree/main/projects/personachat)), with the predefined topic taxonomy and topic-labeled persona
  attributes from PANDA ([Kim et al., 2024](https://aclanthology.org/2024.emnlp-main.669/);
  code: [jin62304/PANDA](https://github.com/jin62304/PANDA)).
  `processing.py`'s `turn_split_personachat()` expects a topic-labeled PANDA-style file at
  `./dataset/original/panda_sample.json`.
- **MBTI-S2Conv** ([Tu et al., 2023](https://arxiv.org/abs/2308.10278); code:
  [morecry/CharacterChat](https://github.com/morecry/CharacterChat)).
  `collect_all_persona_mbti()`, `sample_dataset_mbti()`, and `turn_split_mbti()` in
  `processing.py` expect the dataset's raw `mbti_1024_bank.json` and `mbti_s2conv.json` files
  under `./dataset/MBTI_S2Conv/original/`.

After the `turn_split_*` step, GPT-4o-based topic annotation for the partner's last utterance
and for generated responses is run via `processing.py`'s `persona_labeling*` functions (see
`processing.sh`); this reproduces the topic-labeling procedure described in Appendix A.1 of
the paper. MBTI-S2Conv instead reuses the dataset's own utterance-level topic labels.

## Running SCONPOS

`main_test.py` is driven by `config.py`'s CLI flags and dispatches on
`--test_task` ∈ `{extract_vector, eval, eval_ovs}`:

1. **Extract the steering vector** (Section 3.2 of the paper — response-free, built from the
   contrastive pair of "all persona attributes" vs. "only topic-matched persona attributes"
   prompts):

   ```bash
   python main_test.py --test_task extract_vector \
     --dataset PersonaChat --model_name llama3 \
     --target_layer_ratio 0.5 --token_position last --differnce_method mean
   ```

   This writes `./result/<dataset>/<model_name>/vector/target_layer_ratio=...,....pt`.

2. **Generate responses** — vanilla / no-persona / CoT / decomposition / self-refine baselines
   use `--output_path self/default/all_persona_input` (etc., see `FUNC_MAP_*` in
   `main_test.py` for the full set of supported `output_path` values); SCONPOS-steered
   generation uses `--decoding_strategy steering --use_vector 1` together with the same
   `target_layer_ratio` / `token_position` / `differnce_method` used to extract the vector,
   plus `--contrastive_alpha` (the suppression intensity λ in the paper) and
   `--token_position_steering` (which tokens the steering vector is applied to during
   prefill/decoding). See `inference.sh` for a full example.

3. **Compute metrics** (Overuse, C-Score, P-Dist, P-F1, BLEU, ROUGE-L, BERTScore, PAS) with
   `--test_task eval_ovs`, which writes an `.xlsx` report under
   `./result/<dataset>/<model_name>/main/metrics/...`.

To compute **only PAS** (no GPU or extra models), use `evaluate_pas.py` — see the next
section.

The GPT-4.1 baseline (used in the Section 2 analysis) is generated separately via the OpenAI
Batch API — see `gpt_inference.py` / `inference_gpt.sh` (requires `OPENAI_API_KEY`).

## PAS (Persona Appropriateness Score)

PAS is implemented in [`utility/pas.py`](utility/pas.py), which depends only on NumPy and
can be copied into other projects as-is.

### Definition

For each sample, build two topic distributions over the speaker's persona topics
(plus an extra slot 0 meaning "no persona topic"):

- **query distribution** — topics labeled on the partner's last utterance, i.e. which
  persona topics the context calls for (slot 0 if none);
- **response distribution** — topics of the persona attributes the generated response
  actually uses (slot 0 if none).

```
PAS = 1 − JSD(p_query, p_response)      # Jensen–Shannon distance, base 2 → PAS ∈ [0, 1]
```

Using persona when the context doesn't call for it (**overuse**) and ignoring a persona
topic the context calls for (**underuse**) both lower PAS. The corpus-level score reported
in the paper is the **harmonic mean** of the mean PAS over the *persona-relevant* samples
(query matches ≥1 persona attribute) and the *persona-irrelevant* samples (query matches
none), so always or never using persona cannot score well.

### Use it in Python

```python
from utility.pas import persona_appropriateness_score, corpus_pas

# Speaker's persona topics: 0, 3, 4. The query asks about topic 3;
# the response uses persona attributes with topics 0 and 3.
persona_appropriateness_score(query_topics=[3], response_topics=[0, 3],
                              persona_topics=[0, 3, 4])            # -> 0.442

# Corpus-level PAS over a list of labeled records (format below)
corpus_pas(records, setting="self", persona_scope="agent")
# -> {'relevant': ..., 'irrelevant': ..., 'pas': <harmonic mean>, 'n_relevant': ..., 'n_irrelevant': ...}
```

### Run it from the command line

```bash
python evaluate_pas.py \
  --input ./result/PersonaChat/llama3/main/response_labeled/self/default/all_persona_input/processed/self/default/all_persona_input.json \
  --setting self
```

Example output:

```
            relevant  irrelevant  PAS (HM)
PAS           0.4407      0.2889    0.3490
PAS(Both)     0.3950      0.2393    0.2980
```

`PAS` uses the speaker's persona topics only; `PAS(Both)` uses speaker + partner persona
topics. These are the same values `main_test.py --test_task eval_ovs` writes to its `.xlsx`
report.

### Input format

A JSON list of records (the output of `processing.py --process_task persona_labeling`):

```json
{
  "agent_persona": ["i enjoy shopping.", "i work in an office.", "..."],
  "self_labels": [[0], [3], "..."],
  "user_persona": ["cooking is my passion.", "..."],
  "partner_labels": [[0], "..."],
  "query_labeled_topic": [3],
  "generated_response_labeled_persona": ["i work in an office."]
}
```

- `self_labels[i]` / `partner_labels[i]`: topic ids of `agent_persona[i]` / `user_persona[i]`
  (PersonaChat uses PANDA's topic taxonomy; MBTI-S2Conv uses its own topic labels).
- `query_labeled_topic`: topic ids of the partner's last utterance.
- `generated_response_labeled_persona`: persona attributes that the generated response uses
  (labeled by GPT in `processing.py`; any labeler that outputs the same fields works).

To apply PAS to your own data, you only need these topic labels — the dialogue model and
labeler can be anything.

## Ablations

The three scripts under `ablations/` are self-contained forks of `main_test.py` for specific
appendix analyses; each supports the same `config.py` CLI flags plus the extras noted below.

- `main_test_multi_layer.py` — pass `--target_layer_ratio all` or `--target_layer_ratio mid3`
  to extract/apply the steering vector across all decoder layers, or a 3-layer window centered
  on the midpoint, respectively (instead of a single `target_layer_ratio` fraction). Reproduces
  Tables 18–19 (single layer vs. mid-3 layers vs. all layers).
- `main_test_small_train_ratio.py` — pass `--train_data_ratio` (e.g. `0.05`/`0.1`/`0.25`/`0.5`)
  to extract the steering vector from a subsample of the contrastive training pairs, to test
  how much data SCONPOS needs. Reproduces Table 17.
- `main_test_certain_persona_matched.py` — extracts a *control* steering vector by contrasting
  "All" against a "Match" prompt whose topic-relevant persona attributes are duplicated to
  match the attribute count (and approximate the prompt length) of the "All" prompt, to rule
  out prompt-length confounds. Run identically to `main_test.py`; output paths are namespaced
  under `.../vector/certain_persona_matched/...`. Reproduces Table 20 ("All, Match+dup.").

## Notes

- `--differnce_method` (sic) is a pre-existing typo (should read `difference_method`) that is
  baked into every generated result path; it's kept as-is here for consistency with how the
  paper's results were produced.
- Korean-language inline comments in a few scripts are the original authors' working notes;
  they don't affect how the scripts run.
- `--train_data` / `--valid_data` in `config.py` are vestigial defaults left over from a
  separate SFT baseline trainer that isn't part of this release; they're unused by every
  script here.

## License

Released under the MIT License (see `LICENSE`).

## Citation

```bibtex
@inproceedings{shin-etal-2026-controlling,
    title = "Controlling and Assessing Appropriate Persona Use in {LLM}-based Dialogue Generation",
    author = "Shin, Jongkyung  and
      Lee, Inkyu  and
      Lim, Chiehyeon",
    editor = "Lee, Hung-Yi  and
      Dev, Sunipa  and
      Li, Junyi Jessy  and
      Elliott, Desmond",
    booktitle = "Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing",
    month = oct,
    year = "2026",
    address = "Budapest, Hungary",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.emnlp-main.2029/",
    doi = "10.18653/2026.emnlp-main.2029",
    pages = "46212--46239",
    ISBN = "979-8-89176-459-0",
    abstract = "In persona-based dialogue generation (PDG), LLMs often overuse persona attributes by incorporating them regardless of dialogue context, resulting in unnatural responses. Despite its practical significance, the underlying causes remain unexplored, with no method to mitigate this problem or metric to assess the appropriateness of persona use. To address these issues, we first conduct a comprehensive analysis of LLM-based PDG, revealing that LLMs exhibit a systematic bias to incorporate all given persona attributes, and that existing metrics fail to capture contextual appropriateness. Building on these findings, we propose Self-CONtrastive Persona Overuse Suppression (SCONPOS) to mitigate overuse by directly intervening in LLMs' internal representations at the prompt encoding stage, without requiring any response generation. We further propose the Persona Appropriateness Score (PAS), a novel metric that penalizes both overuse and underuse. Experimental results demonstrate that SCONPOS systematically reduces the overuse, and PAS captures the contextual appropriateness of persona use."
}
```
