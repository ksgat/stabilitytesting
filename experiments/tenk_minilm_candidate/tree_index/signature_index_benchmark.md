# Signature Index Benchmark

- Docs: 10000
- Relative dimensions / anchors: 256
- Query sample: 1000
- Raw target: top-10
- RP forest: 24 trees, leaf size 96, beam leaves 2
- RP forest build seconds: 1.0650

| Method | Pool | Mean Recall | All Top-k | Any Hit | ms/query | Avg Scored |
|---|---:|---:|---:|---:|---:|---:|
| exact_relative | 50 | 0.8568 | 0.3880 | 1.0000 | 0.0842 | 50.0 |
| exact_relative | 100 | 0.9173 | 0.5630 | 1.0000 | 0.0842 | 100.0 |
| exact_relative | 250 | 0.9660 | 0.7890 | 1.0000 | 0.0842 | 250.0 |
| exact_relative | 500 | 0.9835 | 0.8890 | 1.0000 | 0.0842 | 500.0 |
| exact_relative | 1000 | 0.9925 | 0.9450 | 1.0000 | 0.0842 | 1000.0 |
| rp_forest | 50 | 0.8383 | 0.3470 | 1.0000 | 4.9224 | 2713.6 |
| rp_forest | 100 | 0.8893 | 0.4710 | 1.0000 | 4.9224 | 2713.6 |
| rp_forest | 250 | 0.9257 | 0.5870 | 1.0000 | 4.9224 | 2713.6 |
| rp_forest | 500 | 0.9365 | 0.6240 | 1.0000 | 4.9224 | 2713.6 |
| rp_forest | 1000 | 0.9419 | 0.6380 | 1.0000 | 4.9224 | 2713.6 |
