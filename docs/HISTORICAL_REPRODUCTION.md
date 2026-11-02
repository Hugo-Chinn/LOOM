# Historical Native LOOM Reproduction

This guide is for original RankMixer tokenization and recurrence, **not** the
portable numeric demo. Full four-seed industrial AUC reproduction in a clean
environment remains unverified. Follow [validation status](VALIDATION.md).

## Reference Configuration Index

These are width-2 Native DAG controls used in behavior-structure experiments,
not the latest tuned width-4 recipes. Seeds are 20262027 through 20262030.

| Dataset | Config Family Under `configs/research/` | Full Experiment Prefix |
|---|---|---|
| Taobao | `config_rankmixer_structure_ablation_dag` | `DAGTrue` |
| MerRec | `config_rankmixer_structure_merrec` | `Struct_MerRec_True` |
| KuaiRand | `config_rankmixer_structure_kuai` | `Struct_Kuai_True` |
| QK-Video | `config_rankmixer_structure_qk` | `Struct_QK_True` |

The exact model hyperparameters are preserved in the YAML. Do not substitute
Branch for DAGTrue, Head-LoopMT for Native recurrence, or an optimized Full for
the width-2 structural reference.

## Taobao Data Preparation

Obtain Ali_Display_Ad_Click from its official Alibaba/Tianchi distribution,
subject to its current license/access requirements. No raw data is bundled or
redistributed. Required files (CSV, gzip or tar supported):

```text
/absolute/raw/Taobao/
  raw_sample.csv
  ad_feature.csv
  user_profile.csv
  behavior_log.csv
```

This is the display-ad dataset, **not an interchangeable generic Taobao
UserBehavior.csv dataset**. Click comes from the impression sample. Cart/Fav/Buy
are category/brand-matched future events within 86400 seconds after exposure;
they are not exact advertising-ID conversions.

Create a separate environment and install the historical research stack:

```bash
python -m venv .venv-research
source .venv-research/bin/activate
python -m pip install -r research/requirements.txt
python research/data/Taobao/preprocess_Taobao_seq_action.py \
  --data_dir /absolute/raw/Taobao \
  --output_dir /absolute/preprocessed/Taobao_Action \
  --min_user_interactions 10 --min_feat_count 2 \
  --train_ratio 0.8 --valid_ratio 0.1 --test_ratio 0.1 \
  --n_user_parts 32 --train_blocks 32 --valid_blocks 8 --test_blocks 8 \
  --behavior_label_window_seconds 86400
```

These arguments match the recorded dataset metadata; raw-to-processed
equivalence has **not** been rerun as part of release validation. Metadata alone
does not guarantee identical vocabulary assignments, rows or history order.
Do not reduce samples, change label window, or change filtering when comparing
against historical AUC.

Recorded data fingerprint (not a cryptographic checksum):

| Property | Recorded Value |
|---|---|
| Train / validation / test rows | 17822296 / 2871767 / 2907238 |
| Train dates | 2017-05-06 through 2017-05-11 |
| Validation / test dates | 2017-05-12 / 2017-05-13 |
| Timezone | Asia/Shanghai |
| Task order | is_click, cart, fav, buy |
| user_id / item_id vocab size | 470571 / 846812 |
| Blocks | 32 / 8 / 8 |

## Prepare Paths Without Editing Historical YAML

```bash
python research/prepare_native_config.py --dataset taobao \
  --data-root /absolute/preprocessed \
  --output configs/local/taobao-full \
  --model-root /absolute/outputs/taobao-full
```

The helper validates task order, historical vocabulary sizes and block
directories, copies only Full's four seeds, and changes only data/checkpoint
paths. It emits a provenance file. It rejects mismatched vocabularies instead
of silently changing the model. For a genuinely new dataset use the portable
integration guide or explicitly adapt the historical feature schema; that is
not historical reproduction.

## Train Sequentially Before Increasing Concurrency

```bash
for seed in 20262027 20262028 20262029 20262030; do
  CUDA_VISIBLE_DEVICES=0 python research/run_expid.py \
    --config configs/local/taobao-full \
    --expid DAGTrue_S${seed} --gpu 0
done
```

First verify a single run before parallel scheduling. The historical entry
may rewrite feature metadata even with `rebuild_dataset=false`; concurrent
initialization against one dataset directory can race. GPU selection is set
before Python imports CUDA. No multi-GPU scheduler is promised by this guide.

Native historical training uses its original optimizers and BF16/compile flags;
it is not the portable CLI's AdamW/linear-encoder training protocol.

## Compare Every Task And Seed

Training logs under `/absolute/outputs/taobao-full/Taobao_Action/` contain
validation and test blocks. Extract **only the final Test evaluation block**,
not best validation AUC or an aggregate macro score. Reference test values:

| Seed | Click | Cart | Fav | Buy |
|---|---:|---:|---:|---:|
| 20262027 | 0.646600 | 0.834625 | 0.861279 | 0.867601 |
| 20262028 | 0.641978 | 0.838124 | 0.863429 | 0.863455 |
| 20262029 | 0.648042 | 0.842577 | 0.865016 | 0.866985 |
| 20262030 | 0.646474 | 0.836861 | 0.860972 | 0.854027 |

These are measured historical `DAGTrue` results, extracted from the original
four run logs; they are not newly reproduced by the GitHub release.
Compare seed-matched per-task differences and report deviations. Hardware,
PyTorch/CUDA versions, initialization and nondeterministic kernels can affect
results; there is no guaranteed universal AUC tolerance.

For other datasets use `--dataset merrec`, `kuairand`, or `qk` with prepared
data in the corresponding dataset directory. Their raw preprocessing and
full per-task reference tables are not yet packaged into this guide.
