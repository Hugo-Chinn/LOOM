# AutoAna

`AutoAna` is UniRank's automated data analysis module for blocked Parquet datasets. It is read-only: it does not rewrite the source dataset.

It currently provides:

- automatic discovery of `train/valid/test` blocked datasets;
- YAML-aware feature and label validation;
- exact Parquet file, byte, and row counts from metadata;
- bounded streaming samples for null rates, cardinality, common values, numeric ranges, and list lengths;
- label validity, class balance, and split-drift checks;
- block pairing and sampled dense-index range checks;
- machine-readable JSON and human-readable Markdown reports;
- CI-friendly exit codes.
- one directed feedback conflict graph per dataset, with edge value `P(target=0 | source=1)`;

## Quick start

Analyze a dataset directory directly:

```bash
python -m AutoAna analyze \
  --dataset-dir data/synthetic/TencentGR_10M_Action \
  --dataset-id TencentGR_10M_Action_Synthetic \
  --fail-on error
```

Use a UniRank dataset configuration so expected features and labels are checked:

```bash
python -m AutoAna analyze \
  --config config_local/dataset_config.yaml \
  --dataset-id TencentGR_10M_Action_Synthetic
```

Override configured paths while retaining its schema definition:

```bash
python -m AutoAna analyze \
  --config config/dataset_config.yaml \
  --dataset-id KuaiRand_Video_Action \
  --dataset-dir data/preprocessed/KuaiRand_Video_Action
```

Find analyzable datasets:

```bash
python -m AutoAna discover data/preprocessed
```

Reports are written to `AutoAna/reports/<dataset-id>/report.json` and `report.md` by default.
The same directory also contains `feedback_conflict_graph.json` and `feedback_conflict_graph.md`.

Generate a graph from an existing report without rescanning the dataset:

```bash
python -m AutoAna graph \
  --report AutoAna/reports/TencentGR_10M_Action/report.json \
  --output-dir AutoAna/reports/TencentGR_10M_Action
```

The graph is directed because `conflict_rate(A -> B)` and `conflict_rate(B -> A)` are generally different. An edge labeled `0.0000` means every source-positive row is also target-positive; a high value means the proposed implication conflicts with many rows.

## Large dataset behavior

AutoAna always reads every Parquet footer, so file sizes and row counts are exact. Column-level statistics are sampled independently from every block; the default is at most 2,000 rows per file. This avoids loading industrial datasets into memory while preserving coverage across blocks.

Useful controls:

```text
--sample-rows-per-file N
--max-files-per-table N
--top-k N
--distinct-cap N
--drift-warning-threshold FLOAT
--imbalance-warning-threshold FLOAT
--fail-on never|error|warning
```

The JSON report records all sampling options, and the Markdown report explicitly distinguishes exact metadata from sampled statistics.
