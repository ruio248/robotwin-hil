"""Summarize a frozen-cohort encoder ablation without changing score direction."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "outputs/disc_s42_encoder_ablation_20261028"
spec = importlib.util.spec_from_file_location("alignment", REPO / "outputs/d_progress_alignment_20261028/analyze.py")
alignment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(alignment)


def load_rows(path):
    raw = json.loads(path.read_text())
    assert raw["complete"] and raw["action_horizon"] == 50 and raw["candidate_count"] == 4
    assert len(raw["per_episode"]) == 50
    rows = [dict(id=str(r["scene_seed"]), success=bool(r["success"]),
                 scores=np.asarray(r["per_query_mean"], dtype=np.float64)) for r in raw["per_episode"]]
    assert len({r["id"] for r in rows}) == 50 and sum(r["success"] for r in rows) == 31
    assert all(np.isfinite(r["scores"]).all() for r in rows)
    rows.sort(key=lambda r: r["id"])
    return rows, raw


def main():
    assert not (OUT / "comparison.json").exists()
    before, b = load_rows(OUT / "baseline_r3m_outcome_scores.json")
    after, a = load_rows(OUT / "outcome_scores.json")
    assert b["candidate_manifest_sha256"] == a["candidate_manifest_sha256"]
    assert [(r["id"], r["success"], len(r["scores"])) for r in before] == [
        (r["id"], r["success"], len(r["scores"])) for r in after]
    protocol = json.loads((OUT / "protocol.json").read_text())
    assert protocol["complete"] and protocol["d_checkpoint_sha256"] == a["d_checkpoint_sha256"]
    handover, h_source = alignment.load_handover()
    tasks = {"historical_disc_s42": alignment.analyze(handover, h_source)}
    for key, rows, raw in (("stack_r3m", before, b), ("stack_imagenet", after, a)):
        metrics = alignment.analyze(rows, {"checkpoint_sha256": raw["d_checkpoint_sha256"],
                                          "candidate_manifest_sha256": raw["candidate_manifest_sha256"]})
        metrics["fixed_prefix"] = {}
        for length in (1, 3, 5, 6):
            assert all(len(r["scores"]) >= length for r in rows)
            p = np.asarray([r["scores"][:length].mean() for r in rows if r["success"]])
            n = np.asarray([r["scores"][:length].mean() for r in rows if not r["success"]])
            metrics["fixed_prefix"][str(length)] = alignment.summary(p, n)
        tasks[key] = metrics
    # Bootstrap the same episodes across both models, not independent resampling.
    positive = np.asarray([r["success"] for r in before])
    arrays = [np.asarray([np.interp(alignment.GRID, np.linspace(0, 1, len(r["scores"])),
                                   r["scores"]).mean() for r in rows]) for rows in (before, after)]
    rng = np.random.default_rng(42)
    deltas = []
    for _ in range(10000):
        p = rng.integers(positive.sum(), size=positive.sum())
        n = rng.integers((~positive).sum(), size=(~positive).sum())
        aucs = [alignment.auc(v[positive][p], v[~positive][n]) for v in arrays]
        deltas.append(aucs[1] - aucs[0])
    change = dict(auc_difference=alignment.auc(arrays[1][positive], arrays[1][~positive]) -
                 alignment.auc(arrays[0][positive], arrays[0][~positive]),
                 paired_bootstrap_ci95=np.quantile(deltas, [.025, .975]).tolist())
    result = dict(complete=True, tasks=tasks, aligned_auc_change=change, training=protocol["training"],
                  alignment="Exact historical disc_s42 101-point linear within-episode progress interpolation; all episodes contribute at all points.",
                  scoring="Mean raw logit over same frozen K4 full H50 chunks, then episode-equal aggregation; never reverse sign.",
                  limits=["Encoder-only ablation, not full historical H10/K4/uniform-observation training replication.",
                          "50 outcome episodes have been inspected repeatedly; this is diagnostic, not untouched model-selection validation.",
                          "Progress alignment does not align semantic stage and uses retrospective episode termination.",
                          "Raw score magnitudes from separately trained models are not calibrated onto a common scale."])
    (OUT / "comparison.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    plt.rcParams.update({"font.family": "Arial Unicode MS", "axes.unicode_minus": False, "font.size": 11})
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharex=True, layout="constrained")
    titles = [("historical_disc_s42", "历史 handover_to_tray\n原版 disc_s42 · H10 · 100 条"),
              ("stack_r3m", "stack_blocks_two\n现有 R3M 编码器 · H50 · 50 条"),
              ("stack_imagenet", "stack_blocks_two\n换回原版 ImageNet 编码器 · H50 · 同 50 条")]
    for ax, (key, title) in zip(axes, titles):
        item = tasks[key]
        for group, label, color, style in (("success", "成功", "#2166ac", "-"),
                                            ("failure", "失败", "#d95f0e", "--")):
            g = item["groups"][group]
            ax.plot(alignment.GRID * 100, g["mean"], color=color, linestyle=style, linewidth=2.2,
                    label=f"{label} n={g['episodes']}")
            ax.fill_between(alignment.GRID * 100, g["q25"], g["q75"], color=color, alpha=.14, linewidth=0)
        m = item["aligned_101_point_episode_mean"]
        ax.set_title(title + f"\n成功高分 AUC = {m['success_higher_auc']:.3f}", fontsize=12)
        ax.set_xlabel("每条完整轨迹的归一化进度（%）")
        ax.set_ylabel("支撑函数原始 D 分数（K4 平均）")
        ax.set_xlim(0, 100)
        ax.grid(axis="y", alpha=.25)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False)
    fig.suptitle("恢复 disc_s42 视觉编码器的单变量对照：评分方向与轨迹时间对齐均保持不变", fontsize=15)
    fig.supxlabel("阴影为轨迹间 25–75% 区间；各模型纵轴独立，绝对分值不可跨模型比较。\n全轨分离是回顾性诊断，不等于提前识别接管需求。", fontsize=10)
    fig.savefig(OUT / "encoder_ablation_comparison.png", dpi=180)
    fig.savefig(OUT / "encoder_ablation_comparison.pdf")
    for key, item in tasks.items():
        print(key, json.dumps(item["aligned_101_point_episode_mean"]))
    print("PAIRED_CHANGE", json.dumps(change))


if __name__ == "__main__":
    main()
