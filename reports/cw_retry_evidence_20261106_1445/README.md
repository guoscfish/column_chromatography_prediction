# CW Free32 retry evidence snapshot

Captured: 2026-11-06T14:42:13.721547+08:00

Each gzip JSONL row contains the original UTF-8 receipt and its SHA256. Source files remain in place for the running controller. Full request/input payloads and training arrays remain local; this export does not read labels or predictions.

HTTP 429 does not identify whether the provider is enforcing quota, capacity, or another rate limit; response bodies were not preserved in those receipts. A successful transport answer is not an accepted batch. See logical_rejections separately.

| Seed | Started | Terminal attempts | Rejected logical answers |
|---|---:|---:|---:|
| 157 | 28047 | 28046 | 97 |
| 887 | 27162 | 27161 | 22 |
| 2357 | 27620 | 27619 | 17 |
| 6101 | 26794 | 26793 | 24 |
| 12203 | 27708 | 27707 | 59 |
