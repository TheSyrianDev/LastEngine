# Measured performance (RTX 3050 Laptop 4 GB, 24 GB RAM, Sage-A1.5 35B-A3B APEX I-MiniPlus, via HTTP)

| Test | LastEngine | Legacy LastEngine | llama.cpp (b11538, --n-cpu-moe 38) |
|---|---|---|---|
| Prefill, 3K prompt | 787 tok/s | 637 | 309 (2K) |
| Decode, short context | 28-32 tok/s | 25.6-32 | 25.9 |
| Decode, 12K context (f16 KV) | 31.7 tok/s | n/m | n/m |
| 24K context, production flags (--ctx 32768 --kv tcq3 --chunk 2048) | prefill 616 tok/s, decode 25 tok/s | n/m | n/m |
| Multi-turn on 24K context (assistant turn re-sent) | 24012/24032 tokens cached, TTFT 0.43 s | n/m | n/m |

n/m = not measured. Decode varies with the expert-cache hit rate (about 85% on short prompts).
Not implemented: MTP / speculative decoding (verification of 2 tokens needs a dedicated batch-2 decode path and recurrent-state rollback;
PCIe-bound expert streaming limits the expected gain on this card).
