# LARP Index Demo

- Base docs: 9900
- Inserted docs: 100
- Docs after insert: 10000
- Anchors: 256
- Build seconds: 0.5519
- Insert ms/doc: 0.2356
- Save seconds: 0.0557
- Load seconds: 0.1728
- Full search + raw rerank at pool 500: 2.2815 ms/query

| Pool | Candidate Recall@Top-10 | All Top-10 Contained | Candidate ms/query |
|---:|---:|---:|---:|
| 50 | 0.9053 | 0.5260 | 1.2862 |
| 100 | 0.9554 | 0.7200 | 1.3992 |
| 250 | 0.9850 | 0.8830 | 3.0121 |
| 500 | 0.9947 | 0.9520 | 2.4265 |
