"""Export a source-only release without touching the experiment repo's Git state."""
import argparse
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUFFIXES = {".py", ".sh", ".yaml", ".yml", ".md", ".txt", ".png", ".jpg"}
EXCLUDED_PARTS = {
    "__pycache__", ".pytest_cache", ".git", "hf_cache", "logs", "checkpoints",
    "runs_local", "node_modules",
}


def selected_files():
    roots = [ROOT / name for name in ("model_zoo", "unirank", "scripts", "tests",
                                     "assets", "config", "AutoAna")]
    roots += sorted(ROOT.glob("config_*"))
    roots = [p for p in roots if p.name != "config_local"]
    data_dirs = ("KuaiRand", "MerRec", "QK_Video", "TAAC2025", "Taobao")
    files = {ROOT / name for name in (
        "LICENSE", "README.md", "requirements.txt", "autotuner.py",
        "run_expid.py", "run_param_tuner.py", "run_all.sh",
        "data/dataset_stats_utils.py",
    )}
    for directory in data_dirs:
        files.update(p for p in (ROOT / "data" / directory).glob("*")
                     if p.is_file() and p.suffix in {".py", ".sh"})
    for root in roots:
        for p in root.rglob("*"):
            if (p.is_file() and not p.is_symlink()
                    and p.suffix in SUFFIXES
                    and not EXCLUDED_PARTS.intersection(p.relative_to(ROOT).parts)):
                files.add(p)
    return sorted(p for p in files if p.is_file())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    destination = args.destination.resolve()
    if destination.exists():
        raise SystemExit("Destination must be a new directory")
    files = selected_files()
    oversized = [str(p.relative_to(ROOT)) for p in files
                 if p.stat().st_size > 10 * 1024 * 1024]
    if oversized:
        raise SystemExit(f"Review large files before export: {oversized}")
    destination.mkdir(parents=True)
    for p in files:
        target = destination / p.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, target)
    # Keep upstream documentation intact as an explicitly attributed reference.
    (destination / "README.md").rename(destination / "README_UNIRANK.md")
    (destination / "README.md").write_text(
        "# LOOM / LoopMT\n\n"
        "Research implementation of behavior-conditioned recurrent multi-task "
        "ranking, built on [UniRank](https://github.com/salmon1802/UniRank).\n\n"
        "## Implementation\n\n"
        "- Native RankMixer recurrence: `model_zoo/RankMixer.py`.\n"
        "- OneTrans experiments: `model_zoo/OneTrans.py`.\n"
        "- Head-level LoopMT: `model_zoo/LoopMT.py` "
        "(distinct from Native backbone recurrence).\n"
        "- Multi-task baselines: `model_zoo/MTLBaseline.py` and related models.\n"
        "- Experiment configurations: `config_rankmixer_*`, `config_onetrans_*`.\n"
        "- Training entry point: `run_expid.py`.\n\n"
        "## Setup And Training\n\n"
        "```bash\n"
        "python -m venv .venv\n"
        "source .venv/bin/activate\n"
        "pip install -r requirements.txt\n"
        "```\n\n"
        "Prepare the datasets using scripts under `data/` and set dataset paths "
        "in the chosen `dataset_config.yaml`. Some research configurations contain "
        "machine-specific paths that must be adjusted. Datasets are not bundled.\n\n"
        "```bash\n"
        "python run_expid.py --config config_rankmixer_structure_merrec "
        "--expid Struct_MerRec_True_S20262027 --gpu 0\n"
        "```\n\n"
        "## Scope And Reproducibility\n\n"
        "This is a source snapshot of the research workspace, not a validated "
        "minimal release. It includes exploratory configurations and scripts. "
        "Model weights, raw datasets, experiment logs, reports, virtual "
        "environments, and synthetic heatmap values are excluded. "
        "No experimental performance claims are established by this release. "
        "Some analysis scripts require local experiment artifacts that are not "
        "included. Native component ablations use four seeds: "
        "20262027, 20262028, 20262029, 20262030.\n\n"
        "## Attribution And License\n\n"
        "The underlying framework is UniRank; its original documentation is "
        "preserved in `README_UNIRANK.md`. Original copyright notices and "
        "the Apache-2.0 `LICENSE` are retained.\n"
    )
    (destination / ".gitignore").write_text(
        ".DS_Store\n__pycache__/\n*.py[cod]\n.pytest_cache/\n.venv*/\n"
        ".env\n.env.*\n*.pem\n*.key\n*.model\n*.pt\n*.pth\n*.ckpt\n"
        "*.parquet\n*.pkl\n*.log\ncheckpoints/\nruns_local/\nlogs/\nreports/\n"
        "data/preprocessed/\ndata/raw*/\ndata/hf_cache*/\nconfig_local/\n"
    )
    print(f"Exported {len(files)} source files to {destination}")
    print(f"Source size: {sum(p.stat().st_size for p in files) / 1048576:.2f} MiB")


if __name__ == "__main__":
    main()
