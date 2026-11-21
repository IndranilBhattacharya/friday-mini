# Machine Learning & LLM Theory Notes
*A Comprehensive, Beginner-to-Advanced Reference Guide from First Principles*

This document collects and explains the foundational mathematical and engineering concepts of Small Language Models (SLMs), Transformers, and Deep Learning as applied to the F.R.I.D.A.Y. Mini project.

---

## 1. The Two-Phase Training Paradigm: Pretraining vs SFT

A common misconception is that language models learn facts from Question & Answer (Q&A) pairs. **They do not.**

```
[Raw Text Corpus (111 MB Encyclopedia)] 
                 ⬇  Phase 1: Pretraining (Next-Token Prediction)
        [Base Model (Educated Brain)]
                 ⬇  Phase 2: Supervised Fine-Tuning (SFT)
      [Chat Assistant (Helpful Dialog Persona)]
```

### Phase 1: Pretraining (The College Education)
- **Data Source**: Massive, continuous, unstructured prose (Wikipedia, encyclopedias, books, Wikia/Fandom lore). In our project, this is **`data/pretrain.txt` (111 MB, ~27.14 Million tokens)**.
- **Task**: **Next-Token Prediction** (Causal Language Modeling).
  Given the prompt:
  $$\text{"Peter Parker was bitten by a radioactive "}$$
  The model must predict the single next token: $\text{"spider"}$.
- **What is Learned**: Grammar, sentence structure, world logic, character biographies, actor names, relationships, comic timelines, and semantic concepts. **This is where 98% of the model's factual intelligence lives.**

### Phase 2: Supervised Fine-Tuning / SFT (The Job Training)
- **Data Source**: Formatted Question/Answer pairs (`data/sft.jsonl`).
- **Task**: Teaching the model **conversational manners and turn-taking**.
- **What is Learned**: Without SFT, if you ask a pretrained base model *"Who is Thor?"*, it might just continue writing more questions (*"Who is Loki? Who is Odin?"*) because it only knows how to complete documents. SFT teaches it:
  $$\text{When you see "Question: ... Answer:", write the concise answer and emit the stop token } \langle/s\rangle.$$
- **Critical Rule**: SFT should **never** be used to cram new facts into a model. It is merely a behavioral mask placed over the pretrained brain.

---

## 2. Dropout vs Early Stopping: Why They Are NOT the Same

Both techniques share one common goal: **preventing overfitting**. However, their mechanisms, timings, and internal workings are fundamentally different.

```
+--------------------------------------------------------------------------+
|                              Overfitting                                 |
|             (Memorizing test answers instead of understanding)           |
+--------------------------------------------------------------------------+
        |                                                  |
        ▼                                                  ▼
[DROPOUT]                                          [EARLY STOPPING]
Internal Architectural Damper                      External Stop Button
(Alters the brain during training)                 (Monitors the clock & saves the best)
```

### What is Dropout? (Internal Architectural Randomness)
- **Where it operates**: Inside the neural network's layers during every single forward/backward pass.
- **How it works**: At every training step, the computer randomly zeroes out (deactivates) a small fraction of neuron outputs (e.g., $p = 0.05$ or 5%).
- **The Analogy**: Imagine a football team where the coach randomly benches 1 or 2 players on every practice drill. The team cannot rely solely on one superstar player to carry the ball. Every player must learn to coordinate, creating a robust, versatile team.
- **Mathematical Effect**: It prevents **co-adaptation**—neurons can no longer form brittle, interlocking shortcuts that memorize exact training sentences.

### What is Early Stopping? (External Monitoring Policy)
- **Where it operates**: Outside the network, in the training loop orchestrator.
- **How it works**: It changes nothing inside the math of the layers. Instead, every $N$ steps, it feeds a separate "validation set" (questions the model has never been trained on) through the model and measures validation loss.
  - As long as the validation loss is going down, training continues.
  - When the validation loss begins to rise (meaning the model is starting to overfit and lose its general knowledge), the training stops or saves the best checkpoint.
- **The Mathematical Signal**:
  $$\text{Train Loss} \rightarrow 0 \quad \text{while} \quad \text{Val Loss} \uparrow \text{ (spikes upward)}$$
  The training loop tracks the running minimum of $\text{Val Loss}$. If $\text{Val Loss}$ fails to reach a new low after a grace period (patience), the training loop stops.
- **How Does the Computer Know the Answer is "Correct" Without a Human?**:
  The computer does **not** generate free text and judge it subjectively. For every held-out question, the validation dataset already contains the true, correct answer written by humans. The computer calculates the **log-probability (cross-entropy loss)** of the exact true tokens. If the model is confident in the true tokens, loss drops. If it gets confused or assigns high probability to wrong tokens, loss rises immediately.
- **The Analogy**: A chef baking a cake. Early stopping is inserting a toothpick into the cake; when the inside is baked and moist, you pull it out of the oven immediately before the crust burns.

| Feature | Dropout | Early Stopping |
|---|---|---|
| **Category** | Regularization technique inside the architecture | Training termination policy outside the architecture |
| **When it acts** | Every single millisecond during forward pass | Every few hundred steps during evaluation checkpoints |
| **What it alters** | Activations inside the attention / MLP matrices | Decides when to save weights or stop the training loop |
| **Combined use?** | **Yes!** Modern LLMs use both simultaneously. |

---

## 3. The Hierarchy of Time in Training: Tokens, Steps, and Epochs

To understand how **Dropout**, **Early Stopping**, and **Epochs** relate to one another, it helps to see the hierarchy of training from smallest to largest unit:

```
[Token]  (A single piece of a word: "Thor", " is", " the")
   ▼
[Step / Batch]  (A small bite of data: e.g., 2,048 tokens at once)
   │  └── DROPOUT happens here on every single step!
   ▼
[Epoch]  (One complete reading of the entire library: all 27M tokens)
   │  └── An epoch consists of thousands of steps.
   ▼
[Full Training Run]  (Target: e.g., 10 Epochs total)
      └── EARLY STOPPING can stop the run at Epoch 7 if performance peaks!
```

### 1. Step (A Single Bite):
You cannot feed 111 MB of text into a GPU at the same second—it would blow up the memory. Instead, the model takes a small "bite" of data (e.g. 2,048 tokens), makes guesses, updates its weights, and repeats.
- **Dropout** acts right here: on every single step/bite, 5% of connections are randomized.

### 2. Epoch (Finishing the Whole Book):
One **Epoch** is the time it takes the model to finish reading the **entire 111 MB library** (all 27 Million tokens) through thousands of consecutive steps.
- In our project, 1 Epoch of pretraining takes about **1,640 steps** (~1.5 hours on your GPU).

### 3. Early Stopping (Deciding How Many Epochs to Actually Keep):
You might set a target of **10 Epochs** (meaning: *"read the whole encyclopedia 10 times"*).
- But what if by **Epoch 6 or 7**, the model has already learned all the facts and starts memorizing exact punctuation?
- **Early Stopping** steps in: it observes that the score peaked at Epoch 7, and stops the run right there, saving you 3 hours of wasted compute and preventing the model from overcooking!

---

## 4. Loss, Perplexity, and Uncertainty

### What is Cross-Entropy Loss?
Loss measures how "surprised" the neural network is by the true next word.
$$\text{Loss} = - \log P(\text{target token})$$

- If the true next token is assigned probability $1.0$ (100% certainty), $\text{Loss} = -\log(1.0) = \mathbf{0}$.
- If the true next token is assigned probability $0.0001$ (clueless), $\text{Loss} = -\log(0.0001) = \mathbf{9.21}$.

### What is Perplexity?
Perplexity ($PPL$) is the exponential of the loss:
$$\text{PPL} = e^{\text{Loss}}$$
**Intuitive Meaning**: Perplexity represents the **effective number of words the model is hesitating between** when choosing the next token.

| Stage | Loss | Perplexity ($e^{\text{Loss}}$) | Model State |
|---|---|---|---|
| **Untrained (Step 0)** | ~9.0 | $e^{9.0} \approx 8,103$ | Completely random guessing over the 8,000 token vocabulary. |
| **Early Pretrain** | 5.0 | $e^{5.0} \approx 148$ | Knows basic syllable structures; outputs broken English words. |
| **Our Model Pretrain End** | **2.71** | $e^{2.71} \approx \mathbf{15.0}$ | Knows English grammar well, but still hesitates between ~15 words at each step. Factual bindings are loose. |
| **Ideal Pretraining Target** | **< 1.8** | $e^{1.8} \approx \mathbf{6.0}$ | Highly confident about facts, grammar, and entity relationships. |
| **Our Overfit SFT (The Flaw)** | **0.0160** | $e^{0.016} \approx \mathbf{1.016}$ | **Catastrophic Overfitting.** The model is 99% certain of only one exact word sequence. Any rephrased input shatters its expectations. |

---

## 4. The Token-to-Parameter Ratio (Scaling Laws)

### The Chinchilla Scaling Law
In 2022, Jordan Hoffmann et al. (DeepMind) published *"Training Compute-Optimal Large Language Models"*, demonstrating that for an LLM to be compute-optimal:
$$\text{Number of Training Tokens} \approx 20 \times \text{Number of Parameters}$$

### Why Frontier Models (like LLaMA 3.2 3B) Use Trillions of Tokens:
- LLaMA 3.2 3B was trained on **9,000 Billion (9 Trillion) tokens**.
- Ratio: $\frac{9,000\text{B}}{3\text{B}} = 3,000\text{ tokens per parameter}$.
- Why? Meta wanted an extremely small model that performs like an 8B model. To do this, they "over-trained" it on the entire internet so that running inference on phones and laptops is lightning fast.

### Why Small Domain Models Can Train on Millions of Tokens:
1. **Narrow Domain**: Our model does not need to learn Russian history, quantum chromodynamics, or Java programming. It only covers Marvel comics and the MCU.
2. **The "Epochs" Multiplier**:
   When training data is limited (e.g. 27M tokens of high-quality Marvel text), the model can make multiple passes (**epochs**) through the text.
   $$\text{Total Tokens Processed} = \text{Unique Tokens} \times \text{Epochs}$$
   $$\text{Total Tokens} = 27.14\text{M} \times 12\text{ Epochs} \approx \mathbf{325.6\text{ Million Tokens}}$$
   For a ~70M parameter model, processing 325M tokens brings the ratio to:
   $$\frac{325\text{M}}{70\text{M}} \approx 4.65\text{ tokens per parameter}$$
   This is well within the effective learning regime for specialized domain SLMs.

---

## 5. Summary of Why the Current Model Hallucinated and Refused

1. **Pretraining was stopped prematurely (Epoch 3, Loss 2.71)**:
   The model had only seen 81M tokens total. It was like reading an encyclopedia once very quickly. It knew Thor existed, but mixed up who played him because the associations were weakly consolidated.
2. **SFT was pushed to an extreme (Epoch 8, Loss 0.0160)**:
   The model was forced to memorize 72,000 exact strings. It lost all flexible language understanding and became a brittle lookup dictionary.
3. **The Refusal Imbalance**:
   2,600 non-Marvel refusal examples trained the model that whenever it encounters a phrase it doesn't recognize verbatim, it should output: *"That one isn't in my Marvel universe, boss."*

---

## 6. What Can a 70M SLM Actually Do? (Capabilities vs Boundaries)

A realistic breakdown of what an optimized ~70M domain-specialized model can achieve versus a 3B–70B general-purpose model:

| Task Type | Example Prompt | 70M SLM Capability | Why / How It Works |
|---|---|---|---|
| **Direct Facts** | *"Who plays Iron Man in the MCU?"* | **Excellent** | Direct entity-binding stored in the feed-forward weights. |
| **Category Listing** | *"List the main Avengers"* or *"Who are Marvel's anti-heroes?"* | **Strong (Top 5–10)** | Associates categories (heroes, villains, anti-heroes) with high-frequency entities. Won't be an exhaustive 500-item dictionary, but will name key figures accurately. |
| **Rephrased Intent** | *"Tell me about Asgard"*, *"asgard"*, *"What is Asgard?"* | **Strong** | Regularized SFT (loss ~0.9 + dropout) clusters these prompts to the same semantic meaning instead of refusing. |
| **Thematic Speculation** | *"What do you think happens in Avengers: Doomsday?"* | **Moderate (Lore-Grounded)** | Can weave together known MCU premises (Doctor Doom, collapsing multiverse, Fantastic Four) into a compelling, lore-accurate paragraph. |
| **Abstract Logic / Math** | *"If Thor is taller than Tony and Hulk is taller than Thor, who is shortest?"* | **Limited** | Multi-step symbolic reasoning requires billions of parameters and vast web reasoning datasets. |

---

## 7. How to Teach an SLM "Critical Thinking" & Speculation

Small Language Models do not possess conscious thought, but they can produce impressive **thematic reasoning and speculation** (e.g., *"What do you think is going to happen in Avengers: Doomsday?"*) if four specific engineering changes are implemented:

### 1. Data Diversification (The Analytical SFT Mix)
- **The Problem**: 99% of our current SFT pairs were rigid trivia flashcards (*"What is Thor's hammer called?"* ➔ *"Mjolnir"*). If a model only trains on flashcards, it only knows how to recite flashcards.
- **The Fix**: Inject **500–1,000 analytical & speculative examples** into the fine-tuning set:
  - *Speculative*: *"What role might Doctor Doom play in the multiverse collapse?"*
  - *Comparative*: *"Compare the leadership styles of Captain America and Iron Man."*
  - *Analytical*: *"Why is the concept of Incursions central to the Multiverse Saga?"*
  - *Format*: Teach the model the **3-part reasoning template**:
    1. Acknowledge uncertainty/lore context (*"While Marvel keeps official plots sealed..."*)
    2. Synthesize 2–3 narrative premises (*"Drawing from the 2015 Secret Wars comic run and Doom's mastery of sorcery..."*)
    3. Conclude with a logical climax (*"...the heroes will likely face a fractured Battleworld reality."*)

### 2. Deeper Pretraining Representation (Dense Concept Clustering)
- For the model to connect *"Doctor Doom"*, *"Incursions"*, *"Avengers"*, and *"Secret Wars"*, those concepts must sit close together in its high-dimensional latent space.
- Extending pretraining to 10–12 epochs (loss < 2.0) binds these concepts tightly via self-attention so that mentioning one naturally primes the model to recall the others.

### 3. FFN Dimension Expansion (More "Scratchpad" Capacity)
- In a transformer, the **Feed-Forward Network (FFN/SwiGLU)** acts as the model's factual memory store.
- Upgrading from FFN 1408 (36M) to **FFN 1792 (70M)** provides ~40% more associative storage, allowing the intermediate layers to evaluate multiple storylines simultaneously without collapsing into gibberish.

### 4. Sampling Hyperparameters for Speculation
- Factual questions (*"Who is Thor?"*) work best at low temperature (**0.2–0.3**).
- Open-ended / speculative questions (*"What do you think...?"*) require slightly higher temperature (**0.5–0.7**) and a moderate top-p (0.9) to allow creative, narrative phrasing rather than looping greedy tokens.

---

## 8. Hyperparameter Mathematics: How Dimensions Are Chosen

Engineers do not guess numbers like `1,792` or `640` randomly. They are derived from **two strict mathematical and hardware constraints**:

### 1. The SwiGLU 2/3 Rule (Meta LLaMA Scaling)
In classic GPT-2 transformers, the Feed-Forward Network uses two linear layers and expands the dimension by $4\times$:
$$\text{Classic FFN} = 4 \times \text{dim}$$

However, modern LLaMA architecture uses **SwiGLU**, which introduces a gating mechanism with three matrices instead of two (`gate_proj`, `up_proj`, `down_proj`). To keep the total parameter count and compute equivalent to a standard transformer, Meta researchers determined the optimal scaling ratio:
$$\text{SwiGLU FFN} \approx \frac{8}{3} \times \text{dim} \approx 2.667 \times \text{dim}$$

For our chosen width ($\text{dim} = 640$):
$$\frac{8}{3} \times 640 = 1,706.67$$

### 2. GPU Hardware Alignment (Tensor Core Multiples)
NVIDIA GPUs compute matrix multiplications using **warps of 32 threads** and memory cache lines of **128 or 256 bytes**. 
- If a dimension is not divisible by 64 or 128, the GPU wastes memory cycles on padding and suffers memory bank conflicts.
- We round $1,706.67$ up to the nearest clean multiple of **256**:
$$7 \times 256 = \mathbf{1,792}$$

### 3. Attention Head Divisibility
To ensure standard, highly optimized matrix multiplication in the attention heads:
- We set head dimension: $\text{head\_dim} = 64$.
- Number of heads: $\text{num\_heads} = \frac{\text{dim}}{\text{head\_dim}} = \frac{640}{64} = \mathbf{10\text{ heads}}$ (clean integer division).

---

## 9. The LIMA Principle: Why 500 Examples Are Enough

A crucial question: *Won't adding 500 speculative examples just be another key-value map? How can showing 500 examples teach a model to answer any creative question?*

This was answered definitively in 2023 by Meta AI in a landmark paper titled **"LIMA: Less Is More for Alignment"** (*Chuan Zhou et al.*):

### The Superficial Alignment Hypothesis
1. **Knowledge lives in Pretraining, NOT Fine-Tuning**:
   All facts, character personalities, comic plots, and relationships are learned while reading the **111 MB raw text encyclopedia**. The model's brain already contains the raw information.
2. **Fine-Tuning is a "Format Lens", NOT a Knowledge Base**:
   SFT does not teach the model *what* to think; it teaches the model *how to format its output*.
   - If you show an educated human 500 examples of a business proposal, you don't need to teach them proposals for every industry on earth. They grasp the **general structure** (Problem $\rightarrow$ Solution $\rightarrow$ Budget) and apply it to their existing knowledge.
   - When a transformer is shown 500 examples of analytical reasoning (Premise $\rightarrow$ Comic Context $\rightarrow$ Predicted Outcome), the attention layers learn the **generative template of analysis**.
   - It then applies that generative template to *any* character or event stored in its pretraining memory!



