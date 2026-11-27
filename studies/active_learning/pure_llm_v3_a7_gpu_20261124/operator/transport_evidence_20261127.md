# PureLLM A7 transport evidence

Observation cutoff **2026-11-27 05:34:00 CST (UTC+08:00)**. The audit matched Mac bridge jobs, immutable result hashes, and delivery ACKs for unique physical POSTs. It filtered to the five formal seeds, Token4Research, gpt-6.2-astra/xhigh, OpenAI SDK 2.28.0 and direct IPv4 Responses SSE. It kept records with terminal progress timestamps at or before the cutoff, and it excluded preflight-only requests, other protocols, and duplicate copies of an attempt.

Coverage included **222** matched physical POSTs overall, 48 with a verified completed terminal and 84 qualifying pre-text failures across the four exact blocked payloads. Each failure had HTTP 200, a `response.failed` ID that matched the HTTP request trace, explicit `upstream_error`, and a `response.created` timestamp. The archived stream contained no assistant-text milestone or captured complete body. The per-seed counts include all three Key slots.

## Same-response timing

The fixed-input failures are grouped by seed. Timing uses the client's monotonic event clock. Quantiles use linear interpolation.

| Seed | Failures | From dispatch, median s | From `response.created`, p05 / p50 / p95 s |
|---|---:|---:|---:|
| 157 | 18 | 207.459 | 179.064 / 180.062 / 180.628 |
| 2357 | 25 | 211.528 | 179.875 / 180.031 / 184.217 |
| 6101 | 19 | 203.151 | 178.677 / 180.018 / 195.060 |
| 12203 | 22 | 207.623 | 179.667 / 180.063 / 180.222 |
| All four | 84 | 208.295 | 178.917 / 180.033 / 182.090 |

**74/84** failures fell within 1 s of 180 s after `response.created`, and **77/84** fell within 3 s. The median was **180.033 s**. The full quantiles below retain the outliers.

| Metric | Min | p05 | p25 | p50 | p75 | p95 | Max |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dispatch → failure (s) | 196.980 | 198.198 | 200.295 | 208.295 | 213.187 | 242.127 | 414.961 |
| `response.created` → failure (s) | 7.758 | 178.917 | 179.965 | 180.033 | 180.120 | 182.090 | 332.920 |

Across 48 verified complete posts with both timestamps, **9** first recorded a text delta at least 180 s after `response.created`. The longest interval was 648.52 s. The first recorded delta can belong to commentary before the final answer.

## Exact archived input sizes and known completions

The token estimate counts only the model-visible message text with the locally SHA-verified `o200k_base` vocabulary. The exact Astra tokenizer and Token4Research's accounting were unavailable for verification. JSON bytes use the project's compact canonical serialization of the exact archived payload. The failed requests have no completed provider usage report.

| Blocked seed | Round / labels | Proxy tokens | Message UTF-8 bytes | Canonical payload bytes | Matched failure terminal CST |
|---|---|---:|---:|---:|---|
| 157 | R7 / L557 | 210,240 | 390,510 | 417,422 | 05:23:06.597 |
| 2357 | R5 / L493 | 211,434 | 395,110 | 422,282 | 05:33:15.599 |
| 6101 | R8 / L589 | 221,759 | 411,755 | 438,737 | 05:20:57.474 |
| 12203 | R5 / L493 | 222,423 | 409,603 | 436,711 | 05:26:22.058 |

| Seed 887 | Labels before | Proxy tokens | Canonical payload bytes | Actual input / output / reasoning tokens | From dispatch / from `response.created` (s) | Terminal CST |
|---|---:|---:|---:|---:|---:|---|
| R23 | L1069 | 223,101 | 466,069 | 228,470 / 11,442 / 4,137 | 243.216 / 216.329 | 05:14:21.520 |
| R24 | L1101 | 224,027 | 470,398 | 224,038 / 12,600 / 5,291 | 298.522 / 252.496 | 05:21:06.625 |
| R25 | L1133 | 224,951 | 472,988 | 224,962 / 11,329 / 4,616 | 241.670 / 227.007 | 05:27:40.897 |
| R26 | L1165 | 225,793 | 476,650 | 369,859 / 14,607 / 5,455 | 198.504 / 189.053 | 05:32:53.417 |
| R26 correction | L1165 | 228,940 | 490,013 | 374,901 / 966 / 162 | 27.051 / 19.165 | 05:33:38.772 |

The R26 correction was a separate logical call and reported 358,605 cached input tokens. The table includes provider usage as reported. Its reasoning count is part of output usage.

## What the evidence supports

The clustering follows the provider's response lifecycle timestamp. The same requests reached normal HTTP headers and returned explicit terminal failures, well before the local 900 s SSE inactivity and 3,600 s total deadlines. The four successful initial R23–R26 requests were larger by both proxy tokens and canonical bytes, and each completed more than 180 s after `response.created`.

A relay-side or upstream pre-text deadline is a plausible explanation for the repeated approximately 180 s event timing. The exact timeout policy, any context-size ceiling, the responsible component, and the upstream route remain unknown. These counts describe the four repeated payloads in the focused failure group.

## Shared Key allocation and response outcomes

Window **2026-11-26 11:49:49 through 2026-11-27 11:49:49 CST (UTC+08:00)**. The read-only audit joined admitted SQLite lease tickets to Mac traces and result records for the five formal seeds. Dispatch counts follow the persisted flag. A write that stopped partway does not establish upstream processing or billing. Complete receipts include repair calls and responses that subsequently failed scientific validation. Imported fits do not count as new API responses.

| Seed | Admitted leases | Marked dispatched | Complete transport receipts | Explicit failed terminals | Mean admitted queue wait |
|---|---:|---:|---:|---:|---:|
| 157 | 80 | 70 | 4 | 55 | 43.3 s |
| 887 | 81 | 72 | 32 | 26 | 69.1 s |
| 2357 | 87 | 74 | 3 | 59 | 78.5 s |
| 6101 | 73 | 67 | 14 | 39 | 52.0 s |
| 12203 | 91 | 83 | 4 | 65 | 111.8 s |

The 412 admitted tickets had no observed FIFO inversions or same-Key lease overlaps. Key occupancy was **64.91%, 64.54%, and 72.56%** for Key 01, 02, and 03. Occupancy includes anonymous connection checks and local result sealing. Wait timing begins at admission to the shared queue and excludes coordinator recovery and periods before enqueue. The traces in this interval had no observed HTTP 429 or typed rate-limit error.

The larger number of complete receipts for 887 coincided with comparable or fewer submissions than the other seeds and a longer mean Key wait than 157. The relay supplied many more explicit failure terminals to 157, 2357 and 12203. These observations cannot identify the relay's internal routing or distinguish any undisclosed upstream throttling.
