# Configuration Index

All shipped model configurations live here.

| Directory | Purpose |
|---|---|
| `portable/default.yaml` | Supported small LOOM example; start here |
| `research/config/` | Original UniRank defaults |
| `research/config_rankmixer_*` | Historical Native RankMixer and component studies |
| `research/config_onetrans_*` | Historical OneTrans studies |
| `research/config_mtl_*` | Multi-task baseline comparisons |
| `research/config_*` | Other exploratory research configurations |

The portable schema is documented in [Configuration](../docs/CONFIGURATION.md).
Historical YAML uses UniRank's different schema. Do not pass historical YAML to
`loom train`, or portable YAML to `research/run_expid.py`.

Historical family names are retained to preserve experiment identities.
Configurations are not all recommended presets, and their presence does not
mean a completed or validated experiment. Some retain machine-local paths.
