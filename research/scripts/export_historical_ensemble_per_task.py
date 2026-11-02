#!/usr/bin/env python3
"""Export every available task metric without replacing it by macro averages."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/historical_ensemble_distillation_20261026"
DISPLAY = {"taobao": "Taobao", "kuai": "KuaiRand", "merrec": "MerRec", "qk": "QK"}


def main():
    lines = [
        "# 历史模型集成与一步学生蒸馏：完整逐任务结果",
        "",
        "生成日期：2026-10-26。仅汇总已完成的历史checkpoint集成实验；不包含短程改图适配教师。",
        "每个任务均报告AUC和Logloss，Logloss越低越好。本文不以Macro替代逐任务结果。",
        "全部单模型及等权组合都列出，没有按测试挑选组合或省略下降任务。",
        "",
        "## 实验口径",
        "",
        "- Taobao seed20262027；KuaiRand、MerRec、QK seed20262028。",
        "- 每数据集验证1,048,576样本，测试2,097,152样本；此前探索过这些前缀，不是新独立确认。",
        "- 单seed，无跨seed均值/标准差或显著性结论。显示值四舍五入；CSV保留原始精度。",
        "- 组合是概率等权平均；逐任务加权融合另表报告，只按验证任务AUC选权重。",
        "- 最佳单模型的摘要按验证Macro选；学生的单教师按目标任务验证AUC选，两者不必相同。",
        "- 单步学生只预测一个指定目标，其他任务没有输出，不能虚构指标。",
        "- 原生Loop与Parallel/旁路/trust成员混合，不声称仅改变拓扑。",
        "- 本轮与之前短程结构适配实验不同；学生最多512 batches，相同初始化和选模机会。",
        "",
    ]
    records = []
    for key, title in DISPLAY.items():
        result_file = OUT / key / "results.json"
        r = json.loads(result_file.read_text())
        prior_file = (ROOT / "reports/topology_ensemble_taobao_20261026/results.json"
                      if key == "taobao" else
                      ROOT / "reports/ensemble_extension_20261026" / key / "results.json")
        prior = json.loads(prior_file.read_text())
        labels = prior["labels"]
        assert r["names"] == prior["weights_model_order"]
        assert r["samples"] == prior["samples"]
        base = max((x for x in r["teacher_subsets"] if len(x["members"]) == 1),
                   key=lambda x: sum(x["valid"]["auc"])/len(labels))
        chosen = max(r["teacher_subsets"],
                     key=lambda x: sum(x["valid"]["auc"])/len(labels))
        lines += [
            f"## {title}",
            "",
            f"目标学生任务：`{r['target']}`。验证摘要最佳单模型：`{'+'.join(base['members'])}`；"
            f"验证选出的等权组合：`{'+'.join(chosen['members'])}`。",
            f"学生单教师：`{r['validation_best_target_single']}`。成员顺序："
            + " / ".join(f"`{n}`" for n in r["names"]) + "。",
            "",
            "### 全部组合：测试逐任务指标",
            "",
            "| 组合 | 任务 | AUC | Logloss |",
            "|---|---|---:|---:|",
        ]
        for row in r["teacher_subsets"]:
            name = "+".join(row["members"])
            for split in ("valid", "test"):
                assert len(row[split]["auc"]) == len(labels)
                assert len(row[split]["logloss"]) == len(labels)
                for i, task in enumerate(labels):
                    auc, ll = row[split]["auc"][i], row[split]["logloss"][i]
                    records.append([title, r["seed"], split, "equal_subset", name, task, auc, ll])
                    if split == "test":
                        lines.append(f"| {name} | {task} | {auc:.6f} | {ll:.9f} |")
        lines += [
            "",
            "### 选中组合对照：测试差值",
            "",
            "差值为集成减单模型：AUC正值更好，Logloss负值更好。",
            "",
            "| 任务 | 单模型 AUC | 集成 AUC | Δ AUC | 单模型 Logloss | 集成 Logloss | Δ Logloss |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for i, task in enumerate(labels):
            a, b = base["test"], chosen["test"]
            lines.append(
                f"| {task} | {a['auc'][i]:.6f} | {b['auc'][i]:.6f} | "
                f"{b['auc'][i]-a['auc'][i]:+.6f} | {a['logloss'][i]:.9f} | "
                f"{b['logloss'][i]:.9f} | {b['logloss'][i]-a['logloss'][i]:+.9f} |")
        lines += [
            "",
            "### 逐任务验证加权融合",
            "",
            "本表复用前轮历史集成结果，成员hash与本轮逐个一致。"
            "这是逐任务权重，不是上面的等权子集；权重仅由验证集选出。",
            "",
            "| 任务 | 权重（按成员顺序） | 验证 AUC | 验证 Logloss | 测试 AUC | 测试 Logloss |",
            "|---|---|---:|---:|---:|---:|",
        ]
        assert [m["sha256"] for m in prior["models"]] == [
            m["sha256"] for m in r["historical_members"]]
        for i, task in enumerate(labels):
            v = prior["metrics"]["valid"]["validation_weighted"]
            t = prior["metrics"]["test"]["validation_weighted"]
            weights = " / ".join(f"{w:.1f}" for w in prior["selected_task_weights"][i])
            lines.append(f"| {task} | {weights} | {v['auc'][i]:.6f} | "
                         f"{v['logloss'][i]:.9f} | {t['auc'][i]:.6f} | {t['logloss'][i]:.9f} |")
            for split, values in (("valid", v), ("test", t)):
                records.append([title, r["seed"], split, "task_weighted", "validation_weighted",
                                task, values["auc"][i], values["logloss"][i]])
        target_idx = labels.index(r["target"])
        single = next(x for x in r["teacher_subsets"]
                      if x["members"] == [r["validation_best_target_single"]])["test"]
        lines += [
            "",
            "### 目标任务：教师与学生",
            "",
            f"仅任务`{r['target']}`；学生其他任务未建模，无AUC/Logloss可报告。",
            f"学生前向计数GFLOPs/样本：{r['student_forward_gflops']:.9f}。"
            "单/多教师学生推理结构相同；多教师增加训练成本，不增加学生部署调用次数。",
            "",
            "| 模型 | 任务 | 测试 AUC | 测试 Logloss |",
            "|---|---|---:|---:|",
        ]
        entries = {
            "单教师模型": {"auc": single["auc"][target_idx],
                          "logloss": single["logloss"][target_idx]},
            "集成教师模型": r["ensemble_target_test"],
            "无蒸馏学生": r["student_test"]["label"],
            "单教师学生": r["student_test"]["single_teacher"],
            "多教师学生": r["student_test"]["multi_teacher"],
        }
        for name, values in entries.items():
            lines.append(f"| {name} | {r['target']} | {values['auc']:.6f} | "
                         f"{values['logloss']:.9f} |")
            records.append([title, r["seed"], "test", "target_distillation", name,
                            r["target"], values["auc"], values["logloss"]])
        a, b = r["student_test"]["single_teacher"], r["student_test"]["multi_teacher"]
        lines += [
            "",
            f"多教师学生减单教师学生：AUC {b['auc']-a['auc']:+.6f}，"
            f"Logloss {b['logloss']-a['logloss']:+.9f}。",
            "",
            "### 全部组合：验证逐任务指标",
            "",
            "| 组合 | 任务 | AUC | Logloss |",
            "|---|---|---:|---:|",
        ]
        for row in r["teacher_subsets"]:
            for i, task in enumerate(labels):
                lines.append(f"| {'+'.join(row['members'])} | {task} | "
                             f"{row['valid']['auc'][i]:.6f} | {row['valid']['logloss'][i]:.9f} |")
        lines += [
            "",
            f"原始来源：[本轮JSON]({key}/results.json)、"
            f"[预测文件]({key}/teacher_predictions.npz)、"
            f"[前轮加权JSON]({prior_file})。",
            "",
        ]
    lines += [
        "## 结论与边界",
        "",
        "- 等权选中组合不是全任务全指标都改善：KuaiRand Follow AUC下降，QK Click AUC及Logloss下降。",
        "- Taobao Cart、MerRec Checkout/Purchase的等权集成Logloss变差，即使AUC提高。",
        "- 历史集成教师学生在Taobao Buy和QK Share提高AUC并降低Logloss；"
        "在KuaiRand Forward与MerRec Purchase未超过单教师学生。",
        "- 等权组合、逐任务加权融合、目标蒸馏教师是不同配置，禁止混用指标。",
        "- 所有结果仅来自一个seed及已探索前缀，没有多seed标准差、显著性或独立确认。",
        "- GFLOPs为PyTorch注册算子计数、MAC=2，lookup/部分elementwise未完全计入；"
        "不是独立推理延迟。",
        "",
        "机器可读：[完整逐任务CSV](PER_TASK_RESULTS.csv)。包含所有等权组合的验证/测试指标、"
        "逐任务验证加权指标和目标教师/学生指标，保留原始数值精度。",
        "",
    ]
    (OUT / "PER_TASK_RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    with (OUT / "PER_TASK_RESULTS.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "seed", "split", "family", "model", "task", "auc", "logloss"])
        writer.writerows(records)
    print(f"Exported {len(records)} task rows to PER_TASK_RESULTS.md and .csv")


if __name__ == "__main__":
    main()
