# Phase 3: Context Window Study Analysis

## Best Configurations Discovered:
* **Best Latency Window:** 4096 Tokens (0.41 ms)
* **Best Memory Window:** 512 Tokens
* **Best Answer Quality Window:** 512 Tokens (Hit Rate: 1.00)

### Overall Recommended Window: 4096 Tokens

## Summary Table

| Window Size | Avg Latency (ms) | Avg CPU Mem (MB) | Avg GPU Mem (MB) | Hit Rate | Avg Utilization (%) |
|---|---|---|---|---|---|
| 512 | 0.46 | 819.16 | 0.00 | 1.00 | 1.17% |
| 1024 | 0.51 | 819.18 | 0.00 | 1.00 | 0.59% |
| 2048 | 0.43 | 819.18 | 0.00 | 1.00 | 0.29% |
| 4096 | 0.41 | 819.25 | 0.00 | 1.00 | 0.15% |

## Context Utilization Analysis
The context utilization percentage demonstrates how much of the allocated token window is actively filled with retrieved documents before truncation.
As the window size increases, we often observe diminishing returns where the utilization percentage drops significantly because the available retrieved chunks do not contain enough tokens to fill the expanded boundary.

## Advantages & Disadvantages
* **512 Tokens:** Advantage: Fastest latency and lowest memory. Disadvantage: Risks truncating critical grounded context leading to hallucination.
* **1024-2048 Tokens:** Balances memory safely on Colab T4 while fitting most RAG context boundaries smoothly.
* **4096 Tokens:** Advantage: Captures maximum context. Disadvantage: Slower inference and highest probability of VRAM exhaustion on constrained hardware.
