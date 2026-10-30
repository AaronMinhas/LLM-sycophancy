## About This Fork

This repository contains my reimplementation of the behavioural experiments presented in Figure 2 of *When Truth Is Overridden* (Wang et al., AAAI 2026), conducted as part of my thesis at Macquarie University.

The original implementation was adapted for Apple Silicon, with experimental results documented in [RESULTS.md](RESULTS.md). Results for Qwen2.5-7B and Mistral-7B were within one percentage point of the published findings.

This work builds on the original [LLM-sycophancy](https://github.com/kaustpradalab/LLM-sycophancy) repository. Credit for the original research, code and datasets belongs to the authors, whose README is preserved below.

### What changed

My changes are everything after the authors' last commit (`8d8bf4c`):

- `experiments/behavioral_analysis/run_syco.py`:
  - runs on Apple Silicon (MPS) in float16, with CUDA and CPU fallbacks
  - reads `HF_TOKEN` from the environment instead of an untracked `config.py`
  - drops the unused `temperature=0.0` setting (decoding is still greedy via
    `do_sample=False`)
  - adds `--limit N` for quick test runs
- `experiments/data_generation/build_mmlu_reimplementation.py`: rebuilds the
  MMLU prompts from scratch (seed 42). It is not used for the reported
  results, which use the authors' original dataset files (see below).
- `requirements-macos-lock.txt` and `environment-mac-m4pro.txt`: exact
  environment.
- `RESULTS.md` and this section.

### Reproducing

```bash
python3.14 -m venv .venv && source .venv/bin/activate
pip install -r requirements-macos-lock.txt

# The authors' original prompt files, from this repository's history
mkdir -p lib/plain lib/opinion_only/prefix lib/opinion_only/suffix
git show 2d4ebb4:prefix_full_question/mmlu/mmlu_plain.pkl > lib/plain/mmlu_plain.pkl
git show 2d4ebb4:prefix_full_question/mmlu/opinion_only/prefix/mmlu_opinion_only.pkl > lib/opinion_only/prefix/mmlu_opinion_only.pkl
git show 2d4ebb4:prefix_full_question/mmlu/opinion_only/suffix/mmlu_opinion_only.pkl > lib/opinion_only/suffix/mmlu_opinion_only.pkl
shasum -a 256 lib/plain/*.pkl lib/opinion_only/*/*.pkl   # compare with RESULTS.md

cd experiments/behavioral_analysis
export HF_TOKEN=...   # needed for gated models such as Llama
python run_syco.py --model_name Qwen/Qwen2.5-7B-Instruct \
  --question_type plain --input_filename ../../lib/plain/mmlu_plain.pkl
python run_syco.py --model_name Qwen/Qwen2.5-7B-Instruct \
  --question_type opinion_only --input_filename ../../lib/opinion_only/prefix/mmlu_opinion_only.pkl
```

Outputs are written to `experiments/behavioral_analysis/output/mmlu/<question_type>/`.
`git show` needs a full clone; a shallow `--depth 1` clone will not have commit
`2d4ebb4`.

### Not reproduced

- The first- vs third-person and expertise experiments: the `lib/pov/` data
  described below is not in the repository.
- The mechanistic analysis (logit lens, activation patching).
- Five of the seven Figure 2 models. See [RESULTS.md](RESULTS.md#models-that-could-not-be-run).

---


<div align="center">

## When Truth Is Overridden: Uncovering the Internal Origins of Sycophancy in Large Language Models


Keyu Wang*, Jin Li*, Shu Yang, Zhuoran Zhang, Di Wang

(*Contribute equally)


[![AAAI 2026](https://img.shields.io/badge/AAAI-2026-blue)](https://aaai.org/conference/aaai/aaai-26/)
[![Paper](https://img.shields.io/badge/Paper-PDF-red)](https://arxiv.org/abs/2508.02087)

</div>

<p align="center">
    <img src="img/overview.png" alt="" width="70%">
</p>


## Abstract

LLMs often exhibit sycophantic behavior, agreeing with user-stated opinions even when those contradict factual knowledge. While prior work has documented this tendency, the internal mechanisms that enable such behavior remain poorly understood. In this paper, we provide a mechanistic account of how sycophancy arises within LLMs through:

1. **Behavioral Analysis**: Simple opinion statements reliably induce sycophancy, whereas user expertise framing has negligible impact
2. **Mechanistic Analysis**: Two-stage emergence via (1) late-layer output preference shift and (2) deeper representational divergence
3. **Perspective Analysis**: First-person prompts ("I believe...") induce higher sycophancy than third-person framings ("They believe...")

Our findings highlight that sycophancy is not a surface-level artifact but emerges from a structural override of learned knowledge in deeper layers.

---

## Repository Structure

```
.
├── experiments/
│   ├── behavioral_analysis/       # Section: "User Opinion Induces Sycophancy"
│   │   ├── run_syco.py           # Main behavioral experiments      
│   │   └── run_syco.slurm
│   ├── mechanistic_analysis/     # Section: "Mechanistic Analysis"
│   │   ├── run_syco_logit_cot.py # Logit-lens & activation patching
│   │   └── run_syco_logit_cot.slurm
│   └── data_generation/          # Prefix & prompt generation
│       ├── generate_prefixes.py  
│       ├── full_question_builder.py 
│       └── apply_prefixes.py    
├── utils/
│   ├── DataFrameAligner.py       
│   ├── SycophancyAnalysis.py    
│   └── EarlyDecodingAnalysis.py 
├── lib/                           # Dataset library
│   ├── plain/                  # Plain baseline questions
│   ├── opinion_only/             # Opinion-only questions
│   │   ├── prefix/              # With prefix
│   │   └── suffix/              # With suffix
│   └── pov/                     # Perspective (1st/3rd person)
│       ├── prefix/
│       │   ├── first_pov/       # First-person prompts
│       │   └── third_pov/       # Third-person prompts
│       └── suffix/
│           ├── first_pov/
│           └── three_pov/
├── csv/                          
├── requirements.txt
├── README.md
└── DATA_STRUCTURE.md              # Detailed data organization guide
```

---

## Quick Start

> **Note:** For a detailed explanation of the `lib/` data structure and file naming conventions, see [DATA_STRUCTURE.md](DATA_STRUCTURE.md)

### Installation

```bash
# Clone the repository
git clone https://github.com/kaustpradalab/LLM-sycophancy.git
cd LLM-sycophancy

# Install dependencies
pip install -r requirements.txt
```

### Running Experiments

The experiments are organized by paper sections. Each section has standalone scripts that can be run independently.

---

## Experiments by Paper Section

### 1. Behavioral Analysis (Section: "User Opinion Induces Sycophancy")

**Objective**: Measure sycophancy rates across different prompt conditions (Plain, Opinion-only, Opinion + Expertise)

#### Run Basic Inference

```bash
cd experiments/behavioral_analysis

# Plain questions (baseline)
python run_syco.py \
  --model_name meta-llama/Llama-3.1-8B-Instruct \
  --question_type plain \
  --input_filename ../../lib/plain/mmlu_plain.pkl

# Opinion-only questions
python run_syco.py \
  --model_name meta-llama/Llama-3.1-8B-Instruct \
  --question_type opinion_only \
  --input_filename ../../lib/opinion_only/prefix/mmlu_opinion_only.pkl

# Opinion + Expertise (First-person, Advanced)
python run_syco.py \
  --model_name meta-llama/Llama-3.1-8B-Instruct \
  --question_type prefix_and_opinion \
  --prefix_type academic \
  --academic_level advanced \
  --prefix_subtype original \
  --input_filename ../../lib/pov/prefix/first_pov/mmlu_academic_opinion_advanced.pkl
```

**Key Parameters:**
- `--question_type`: `plain`, `opinion_only`, or `prefix_and_opinion`
- `--academic_level`: `beginner`, `intermediate`, or `advanced`
- `--prefix_subtype`: `original` (first-person) or `third_pov` (third-person)

#### SLURM Cluster

```bash
# Submit job to SLURM cluster
sbatch run_syco.slurm "meta-llama/Llama-3.1-8B-Instruct"

# Monitor job
squeue -u $USER
```

**Paper Findings** (Figure 2, Figure 3):
- Opinion-only prompts induce ~63.7% average sycophancy rate
- Expertise level (Beginner/Intermediate/Advanced) has <4.4% impact

---

### 2. Mechanistic Analysis (Section: "Mechanistic Analysis")

**Objective**: Understand *how* and *when* sycophancy emerges through layer-wise analysis

#### Logit-Lens & Activation Patching

```bash
cd experiments/mechanistic_analysis

# Run with logit-lens analysis (all layers)
python run_syco_logit_cot.py \
  --model_name meta-llama/Llama-3.1-8B-Instruct \
  --question_type opinion_only \
  --inference_mode logit_only \
  --inference_layer all \
  --input_filename ../../lib/opinion_only/prefix/mmlu_opinion_only.pkl

# Run with specific layers (e.g., odd layers)
python run_syco_logit_cot.py \
  --model_name Qwen/Qwen2.5-7B-Instruct \
  --question_type plain \
  --inference_layer odd \
  --input_filename ../../lib/plain/mmlu_plain.pkl
```

**Key Parameters:**
- `--inference_mode`: `logit_only` (just logits) or `logit_and_cot` (+ chain-of-thought)
- `--inference_layer`: `all`, `odd`, `even`, or `last`

**Paper Findings** (Figure 4, Figure 5, Figure 6):
- Decision score shift occurs at layers 16-19 (Llama 8B)
- KL divergence peaks at layer 23 (representational shift)
- Activation patching at critical layers reduces sycophancy by 36%

#### Early Decoding Analysis

```bash
# Layer-wise decision tracking
python run_early_decoding.py \
  --model_name meta-llama/Llama-3.1-8B-Instruct \
  --input_file ../../lib/plain/mmlu_plain.pkl
```

---

### 3. Grammatical Perspective Analysis (Section: "Grammatical Person Analysis")

**Objective**: Compare first-person ("I believe") vs. third-person ("They believe") prompts

```bash
cd experiments/behavioral_analysis

# First-person (1st POV)
python run_syco.py \
  --model_name meta-llama/Llama-3.1-8B-Instruct \
  --question_type prefix_and_opinion \
  --prefix_type academic \
  --academic_level advanced \
  --prefix_subtype original \
  --input_filename ../../lib/pov/prefix/first_pov/mmlu_academic_opinion_advanced.pkl

# Third-person (3rd POV)
python run_syco.py \
  --model_name meta-llama/Llama-3.1-8B-Instruct \
  --question_type prefix_and_opinion \
  --prefix_type academic \
  --academic_level advanced \
  --prefix_subtype third_pov \
  --input_filename ../../lib/pov/prefix/third_pov/mmlu_academic_opinion_advanced.pkl
```

**Paper Findings** (Figure 8, Figure 9, Figure 10):
- First-person prompts induce 13.6% higher sycophancy on average
- KL divergence shows first-person creates stronger representational shifts
- Cosine similarity reveals orthogonal encoding of 1st vs. 3rd person perspectives

---

## Tested Models

We tested 7 model families across similar parameter sizes:

| Model | Parameters | Family |
|-------|-----------|--------|
| Llama-3.1-8B-Instruct | 8B | Meta |
| Llama-3.2-1B, 3B | 1B, 3B | Meta |
| Qwen2.5-1.5B, 3B, 7B, 14B-Instruct | 1.5B-14B | Alibaba |
| Mistral-7B-Instruct-v0.3 | 7B | Mistral AI |
| Falcon-7B | 7B | TII |
| OLMoE-1B-7B-Instruct | 1B-7B | AI2 |
| OPT-6.7B | 6.7B | Meta |
| Pythia-6.9B | 6.9B | EleutherAI |

---

## Citation

If you use this code or build upon our work, please cite:

```bibtex
@inproceedings{wang2026sycophancy,
  title={When Truth Is Overridden: Uncovering the Internal Origins of Sycophancy in Large Language Models},
  author={Wang, Keyu and Li, Jin and Yang, Shu and Zhang, Zhuoran and Wang, Di},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2026}
}
```

---

## Key Findings Summary

### Takeaway 1: Opinion-Driven, Not Authority-Driven
> Sycophantic behavior in LLMs is primarily triggered by the presence of a user opinion, regardless of the user's claimed expertise or authority.

### Takeaway 2: Two-Stage Emergence
> Sycophancy emerges in two stages: (1) late-layer output preference shift compared to Plain, then (2) deep representational divergence, confirming opinion framing overrides learned knowledge both behaviorally and internally.

### Takeaway 3: Expertise Has No Internal Encoding
> Expertise-level framing fails to influence behavior because models do not encode it internally: opinion prompts form distinct clusters while level prompts overlap, indicating expertise cues are ignored representationally.

### Takeaway 4: Perspective Matters
> Grammatical person is a key driver of sycophancy in LLMs. Changing prompts from first- to third-person framing substantially reduces sycophantic behavior, with this effect encoded deep within the model's representations.


---

## Acknowledgments

This work was supported by KAUST funding BAS/1/1689-01-01, KAUST Center of Excellence for Generative AI (award 5940), and a gift from Google.

Special thanks to the AAAI 2026 reviewers for their valuable feedback.
