# Pretrained assets used

| item | value |
|---|---|
| checkpoint | `microsoft/deberta-v3-large` |
| revision (HF commit) | `64a8c8eab3e352a784c658aef62be1662607476f` |
| architecture | DeBERTa-v2/v3, 24 layers, hidden 1024, 16 heads, vocab 128 100 |
| parameters | 434.6 M total in our model (≈304 M backbone + 131 M embeddings + 1.5 M heads) |
| license | MIT |
| adaptation | **fully fine-tuned** (no freezing, no LoRA/PEFT) |
| precision | fp32 master weights, bf16 autocast compute |
| tokenizer | SentencePiece (`spm.model`), `is_split_into_words=True` over the benchmark's whitespace tokens, first-sub-token pooling |

Note on sequence length: the config reports `max_position_embeddings: 512`, but
DeBERTa-v2/v3 uses relative position attention, and in any case the longest
window is 240 sub-tokens on dev and 240 on test (one training window reaches 523).
Sequence length is not a constraint on this benchmark.

Downloaded but **not** used in any reported result: `answerdotai/ModernBERT-large`
(fetched for the H2 backbone sweep, which was deprioritised — see HYPOTHESES.md H2).

No external training data, pseudo-labels, synthetic examples, or LLM supervision
of any kind were used. Training data is exactly the 1278 official training windows.
