# nba-predict — What to Build and Why

**Repo:** https://github.com/damionrashford/nba-predict
**Purpose of this doc:** Close the gaps between the current project and what Shopify's MLE interview requires.

---

## The Problem With the Current State

Right now nba-predict is a well-built **research project**. The ML fundamentals are solid:
- Temporal splits (no leakage)
- XGBoost with Optuna tuning
- Calibration metrics (Brier, ECE)
- Multiple model types
- A promote.py that mimics a deployment workflow

But when a Shopify interviewer asks *"how does it serve predictions?"* the honest answer is:
> "You run a Python script that loads a .joblib file and prints to the terminal."

That is not production. That gets you rejected at Stage 1.

The two things that turn this from a research project into something you can credibly call production:
1. A serving layer — an API that receives requests and returns predictions
2. A monitoring layer — something that watches the model's behavior over time

---

## Gap 1 — Production Serving

### What to build

A FastAPI app that:
- Loads the trained `.joblib` models at startup
- Exposes a `/predict` endpoint for each model type
- Tracks latency on every request
- Has a `/health` endpoint to confirm the service is alive
- Has a `/metrics` endpoint that returns basic request stats

### Why this matters for the interview

Every system design and modeling interview at Shopify eventually lands on:
- "How does this model run in production?"
- "What are your latency requirements?"
- "How do you serve it — in-process, remote endpoint, cached sidecar?"
- "What happens if the model endpoint is slow or down?"

Right now you cannot answer any of these from personal experience. With a FastAPI wrapper you can say:
> "I serve it as a REST API. At startup the model artifact loads into memory from disk. Each prediction request runs in under 50ms. I track p50/p95 latency per endpoint and expose it at /metrics."

That's a C6 answer. The current state is a C4 answer.

### Specifically what to implement

```
nba_predict/
  serving/
    app.py          ← FastAPI app, loads models at startup
    schemas.py      ← Pydantic request/response models
    middleware.py   ← Latency tracking middleware

scripts/
  serve.py          ← Entry point: uvicorn nba_predict.serving.app:app
```

**Endpoints:**

| Endpoint | Method | What it does |
|---|---|---|
| `/health` | GET | Returns `{"status": "ok", "models_loaded": [...]}` |
| `/predict/game_winner` | POST | Takes two team names + date, returns win probability |
| `/predict/point_spread` | POST | Takes two team names + date, returns predicted spread |
| `/predict/player/{name}` | GET | Returns next-season stat predictions for a player |
| `/metrics` | GET | Returns request count, p50/p95 latency per endpoint |

**Why latency tracking specifically:**
The MLOps interview at Shopify probes GPU utilization, batching, and latency budgets in detail. If you can say "my p95 prediction latency is 12ms because the model loads into memory at startup and runs entirely on CPU with XGBoost's optimized C++ backend" — that's a real answer from real experience.

---

## Gap 2 — Monitoring and Drift Detection

### What to build

A monitoring module that:
- Tracks the distribution of incoming features over time
- Compares current feature distributions against the training distribution
- Flags when a feature has drifted beyond a threshold
- Logs prediction confidence distributions (are probabilities clustering near 0.5? That's a signal something is off)
- Generates a daily/weekly drift report

### Why this matters for the interview

The Shopify MLE interview — across all three shortlist stages — comes back to the same theme:

> "What happens after you deploy? How do you know the model is still working?"

Common probes:
- "How would you detect that your model's performance has degraded in production?"
- "What if the input data distribution shifts? How do you catch that?"
- "Your offline metrics looked great but online performance dropped. How do you debug that?"

Right now you have no answer from experience. With a monitoring module you can say:
> "I track PSI (Population Stability Index) on my key features weekly. If PSI > 0.2 on any feature I flag it for investigation. I also track the distribution of predicted probabilities — if the mean confidence drifts toward 0.5 it usually means a feature pipeline is broken upstream."

That's a direct answer to a direct interview question.

### Specifically what to implement

```
nba_predict/
  monitoring/
    drift.py        ← PSI calculation, feature distribution comparison
    predictions.py  ← Tracks predicted probability distributions over time
    report.py       ← Generates drift report (markdown or JSON)

scripts/
  monitor.py        ← Run drift check against latest predictions
```

**Drift detection — PSI (Population Stability Index):**

PSI is the standard metric for feature drift in production ML. Formula:

```
PSI = sum((actual% - expected%) * ln(actual% / expected%))
```

Thresholds:
- PSI < 0.1 = No significant change
- PSI 0.1–0.2 = Moderate change, monitor closely
- PSI > 0.2 = Significant shift, investigate

For nba-predict, run PSI on:
- Rolling win rate features (home/away)
- Point differential features
- Roster quality scores
- Prior season SRS

**Prediction confidence monitoring:**

Log every prediction the API makes. Weekly, compute:
- Mean predicted probability (should hover around 0.52-0.55 for home teams historically)
- Standard deviation of predictions (if it collapses, model is overconfident)
- % of predictions in extreme buckets (<0.3 or >0.7) — spikes indicate data issues

**Online evaluation (where you have ground truth):**

Since NBA games resolve same-day, you actually can do online evaluation — something most ML projects can't:
- After games play out, compare predicted winner to actual winner
- Track rolling accuracy over last 30 days vs training accuracy
- Alert if rolling accuracy drops more than 3% below training baseline

This is a feature most deployed ML systems don't have (delayed labels). The fact that nba-predict has same-day ground truth makes it a uniquely strong demo.

---

## What This Looks Like in an Interview

### Before (current state)

**Interviewer:** Walk me through an end-to-end model you built and shipped to production.

**You:** I built an NBA prediction system using XGBoost. It predicts game winners, point spreads, and player stats. I used temporal splits, Optuna for hyperparameter tuning, and calibration metrics.

**Interviewer:** How does it serve predictions?

**You:** You run a Python script... *(rejected)*

---

### After (with these improvements)

**Interviewer:** Walk me through an end-to-end model you built and shipped to production.

**You:** I built an NBA prediction system. Four XGBoost models — game winner, point spread, player performance, season outcomes. 26 seasons of Basketball Reference data, 200 features including rolling team stats, prior-season SRS, and roster quality scores built from player advanced stats.

The models train on a temporal split — I use seasons 1999-2018 for training, 2019-2021 for validation, 2022-2026 for test. No leakage. I ran Optuna hyperparameter search and calibrated the game winner model — measured ECE and Brier score to make sure my probabilities were meaningful, not just ranked correctly.

Serving: FastAPI app, models load into memory at startup. `/predict/game_winner` takes two team names and returns a win probability in under 15ms p95. I track latency per endpoint and expose it at `/metrics`.

Monitoring: I compute PSI weekly on my key rolling features to catch data drift. I also track the distribution of predicted probabilities over time — if mean confidence drifts toward 0.5 that usually means a feature pipeline broke upstream. Because NBA games resolve same-day I can do online evaluation too — I track rolling 30-day accuracy and alert if it drops more than 3 points below my test baseline.

**Interviewer:** What would you do differently with more time?

**You:** I'd move from a static model to one that retrains on a rolling window — right now I retrain at the start of each season. I'd also add a feature store so the rolling stats precompute on a schedule instead of at request time, which would get latency under 5ms. And I'd want to A/B test the calibrated vs uncalibrated model in production — right now I made that decision purely offline.

---

That answer clears Stage 1 and gives you real content for the Technical Deep Dive.

---

---

## Gap 3 — LLM Fine-Tuning (GenAI/AI Agent Role Specific)

### The opportunity

You already have 26 seasons of structured game data, feature-engineered matchups, and calibrated predictions. That is a training dataset waiting to happen. Fine-tune a small open-source LLM to be an **NBA prediction analyst** — a model that takes in game context and generates a reasoned prediction with explanation, not just a probability number.

This directly closes the second gap identified for the GenAI/AI Agent role. The interview question most relevant to that role is the **LLM Coding Assistant modeling problem** which probes:
- SFT vs RL — when do you use each?
- Data preparation for LLM training
- Scaling and model parallelization
- Inference optimization
- Evaluation: how do you measure first-pass correctness?

If you have done this on NBA data, every one of those questions becomes answerable from personal experience. The domain is different (NBA vs coding assistant) but the techniques are identical.

---

### What to build

**Base model:** Llama 3.1 8B or Mistral 7B — small enough to fine-tune on a single GPU (via Unsloth or Hugging Face TRL), large enough to reason meaningfully.

**The task:** Given structured game context (team stats, recent form, matchup history, home/away), generate:
1. A predicted winner with confidence
2. A 2-3 sentence explanation of the key factors driving the prediction
3. Key risks to the prediction (what would make this call wrong)

This is a generative task on top of your structured prediction system — the XGBoost model gives you the probability signal, the LLM adds the reasoning layer.

---

### Phase 1 — Build the SFT Dataset

This is where the work is. Garbage data = garbage model. The quality of your instruction dataset is the single most important factor in fine-tuning quality.

**What an instruction sample looks like:**

```json
{
  "instruction": "Predict the winner of Lakers vs Celtics on 2025-01-15. 
                  Lakers: 8-3 last 10, +4.2 avg margin, home.
                  Celtics: 6-4 last 10, +1.8 avg margin, away.
                  Prior season SRS: Lakers +3.1, Celtics +4.8.",
  "output": "Predicted winner: Lakers (62% confidence).
             Key factors: Home court advantage is significant here — 
             Lakers are 7-1 at home this season. Their recent form (+4.2 
             avg margin last 10) outpaces the Celtics (+1.8). However, 
             Celtics' stronger prior-season SRS (+4.8 vs +3.1) suggests 
             they are the better team overall. This game is closer than 
             the home record suggests.
             Key risk: If Tatum is healthy and playing at his season average, 
             the Celtics close this gap significantly."
}
```

**How to generate this dataset at scale:**

You have two options:

1. **GPT-4 bootstrapping (faster):** Feed your structured matchup data into GPT-4 with a prompt that generates analyst-style reasoning. Use this to create ~5,000-10,000 high-quality instruction samples. Then fine-tune your small model to replicate this reasoning style at much lower inference cost.

2. **Human-written + augmented (slower, higher quality):** Write 100-200 high-quality reasoning samples yourself, then use data augmentation (vary the team names, stats, dates) to expand to a larger corpus.

**Why GPT-4 bootstrapping is the right call here:**
This is exactly how many production instruction-following models are built (Alpaca, Vicuna). You use a large expensive model to generate the training data, then distill that capability into a small cheap model. This is a real technique with a real name — **knowledge distillation via instruction following** — and saying you used it in an interview is a strong signal.

**Dataset size target:** 5,000-10,000 examples for SFT. Split 90/5/5 train/val/test.

---

### Phase 2 — SFT (Supervised Fine-Tuning)

Fine-tune the base model on your instruction dataset.

**Tools:**
- **Unsloth** — fastest way to fine-tune Llama/Mistral on a single GPU, 2x faster than standard HuggingFace TRL, uses QLoRA (4-bit quantization during training)
- **HuggingFace TRL** — standard library, more flexible, slightly slower
- **Modal or RunPod** — rent a GPU if you don't have one locally (A100 80GB for ~$1.50/hr, fine-tune 8B model in 2-4 hours)

**What to track during training:**
- Training loss and validation loss (watch for overfitting — val loss increasing while train loss drops)
- Perplexity on held-out game examples
- Sample generations at each checkpoint — does the model's reasoning actually make sense?

**LoRA config (what you'll tune):**
```
r = 16           # LoRA rank — higher = more parameters, more capacity
lora_alpha = 32  # scaling factor
target_modules = ["q_proj", "v_proj"]  # which layers to adapt
lora_dropout = 0.05
```

**Why LoRA/QLoRA specifically:**
Full fine-tuning an 8B model requires ~80GB VRAM. QLoRA reduces this to ~10GB by keeping the base model in 4-bit and only training small adapter weights. This is the standard production technique for fine-tuning LLMs without a fleet of H100s. Being able to explain why you used LoRA vs full fine-tuning — and the tradeoffs — is a C6/C7 signal in the LLM Coding Assistant modeling interview.

---

### Phase 3 — RL with Ground Truth Reward (the unique advantage)

This is where nba-predict has something most fine-tuning projects don't: **real ground truth that resolves same-day.**

Most RL for LLMs uses human preference feedback (RLHF) or an AI reward model because you can't automatically evaluate whether generated text is "correct." NBA games give you an objective outcome to reward against.

**The reward signal:**

```python
def reward(prediction_text: str, actual_outcome: dict) -> float:
    """
    Score the model's prediction after the game resolves.
    
    Components:
    - Correctness: Did the model pick the right winner? (+1.0 / 0.0)
    - Calibration: How close was the stated confidence to actual win rate? 
                   (penalize overconfidence)
    - Reasoning quality: Did the model identify the factors that actually 
                        mattered? (scored by a lightweight classifier)
    """
    ...
```

**RL algorithm:** PPO (Proximal Policy Optimization) via TRL's `PPOTrainer`. This is the same algorithm used in InstructGPT/ChatGPT's RLHF stage.

**Why this is a strong interview answer:**
Most candidates doing RL for LLMs are doing it with synthetic or human-labeled rewards. You have a real, automatically-verifiable reward signal. When asked "how did you know your RL training was working?" you can say: "The model's predictions became better calibrated and the win rate on held-out 2025-2026 games improved from 64% to 67% after RL. More importantly, post-RL the model stopped making overconfident predictions on close matchups."

---

### Phase 4 — Inference Optimization

After fine-tuning, get the model serving predictions fast.

**Target:** Sub-500ms response time for a full prediction with reasoning. This is achievable on a consumer GPU with the right setup.

**What to implement:**

1. **Quantization:** Convert the fine-tuned model to GGUF format (via llama.cpp) or GPTQ. This reduces the model from ~16GB (fp16) to ~4-5GB (4-bit) with minimal quality loss. Enables serving on a single consumer GPU or even CPU.

2. **vLLM serving:** Use vLLM as the inference engine instead of vanilla HuggingFace `generate()`. vLLM uses PagedAttention for efficient KV cache management, dramatically improving throughput when handling multiple concurrent requests.

3. **Streaming:** Stream the response token by token so the user sees output immediately instead of waiting for the full generation. This makes p50 latency feel much better even if p95 is the same.

**Why this matters for the interview:**
The MLOps interview and the LLM Coding Assistant problem both probe: "Your model is deployed but GPU utilization is only 20%. How do you investigate and fix it?" If you've done this hands-on — found that batching wasn't enabled, switched from HuggingFace to vLLM, and watched throughput 3x — that's a real story. You can't fake that answer.

---

### Phase 5 — Evaluation

How do you know the fine-tuned model is actually better than just calling GPT-4 directly?

**Metrics to track:**

| Metric | How to measure |
|---|---|
| Prediction accuracy | Does the model's stated winner match the actual outcome? Compare SFT vs RL vs base model vs GPT-4 baseline |
| Calibration | When the model says "65% confidence," does it win ~65% of those games? |
| Reasoning quality | Human eval on 100 samples: does the explanation identify the real factors? |
| Latency | p50/p95 response time vs GPT-4 API |
| Cost | Per-prediction cost: fine-tuned local model vs GPT-4 API |

The last two are important. A key reason to fine-tune instead of just calling GPT-4 is cost and latency at scale. If your fine-tuned 8B model achieves 90% of GPT-4's reasoning quality at 1/20th the cost and 5x lower latency, that's a real business case — and that's exactly the kind of tradeoff discussion Shopify's LLM Coding Assistant problem is designed to elicit.

---

### How This Translates to the Interview

The LLM Coding Assistant modeling problem asks you to build an LLM with "high first-pass correctness and predictable latency" using SFT and RL. Here's how your NBA fine-tuning maps directly:

| Interview probe | Your answer from experience |
|---|---|
| "How did you build your training dataset?" | GPT-4 bootstrapping — fed structured matchup data, generated analyst-style reasoning, created 8,000 instruction samples |
| "Why SFT before RL?" | SFT teaches the model the output format and reasoning style. RL then optimizes for correctness using the game outcome as reward. Doing RL without SFT first leads to reward hacking — the model finds ways to maximize the reward signal without actually reasoning well |
| "How did you handle the reward signal in RL?" | I had real ground truth — games resolve same-day. I used a composite reward: correctness (did the predicted winner win?), calibration penalty (overconfident predictions get penalized), and a lightweight reasoning quality scorer |
| "How did you optimize inference?" | Quantized to 4-bit GGUF, switched from HuggingFace generate() to vLLM, enabled streaming. Got p95 from 2.1s to 380ms |
| "How did you evaluate whether fine-tuning was worth it vs just calling GPT-4?" | Accuracy within 3% of GPT-4, 20x cheaper per prediction, 5x lower latency. For a high-volume use case that's the right tradeoff |

---

## Full Build Order (All Three Gaps)

Do these in sequence:

1. **FastAPI serving layer** — clears the Stage 1 hard reject
2. **Latency middleware** — adds to serving layer, big interview signal
3. **PSI drift detection** — standalone script, catch feature drift
4. **Online evaluation tracking** — rolling accuracy, ground truth advantage
5. **Drift report** — wraps 3 and 4
6. **SFT dataset generation** — GPT-4 bootstrapping on your matchup data
7. **SFT fine-tuning** — Unsloth + QLoRA on Llama 3.1 8B
8. **RL training** — PPO with game outcome reward signal
9. **Inference optimization** — quantization + vLLM
10. **LLM evaluation** — accuracy, calibration, cost/latency vs GPT-4

Steps 1-5 = closes the "not production" problem.
Steps 6-10 = closes the "never trained an LLM" problem.

---

## Files to Create

**Serving + Monitoring (Steps 1-5):**
```
nba_predict/serving/app.py
nba_predict/serving/schemas.py
nba_predict/serving/middleware.py
nba_predict/monitoring/drift.py
nba_predict/monitoring/predictions.py
nba_predict/monitoring/report.py
scripts/serve.py
scripts/monitor.py
```

**LLM Fine-Tuning (Steps 6-10):**
```
nba_llm/
  data/
    generate_dataset.py     ← GPT-4 bootstrapping, outputs JSONL
    augment.py              ← Data augmentation to expand corpus
    validate.py             ← Quality checks on generated samples
  training/
    sft.py                  ← SFT with Unsloth/TRL
    rl.py                   ← PPO training with game outcome reward
    reward.py               ← Reward function (correctness + calibration)
  inference/
    serve.py                ← vLLM serving wrapper
    quantize.py             ← GGUF conversion script
  evaluation/
    accuracy.py             ← Prediction accuracy vs base model vs GPT-4
    calibration.py          ← Confidence calibration evaluation
    latency.py              ← Latency benchmarking
scripts/
  train_llm.py              ← Entry point: run full SFT + RL pipeline
  evaluate_llm.py           ← Run evaluation suite
```

Update `README.md` with a section on the LLM component — this is the first thing a Shopify MLE interviewer will read when you share the repo link.
