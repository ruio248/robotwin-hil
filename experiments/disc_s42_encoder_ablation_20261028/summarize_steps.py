"""Compare frozen H50 discriminator outcomes at 1,000 and 10,000 updates."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "outputs/disc_s42_step_ablation_20261028"
OLD = REPO / "outputs/disc_s42_encoder_ablation_20261028"
spec = importlib.util.spec_from_file_location("alignment", REPO / "outputs/d_progress_alignment_20261028/analyze.py")
alignment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(alignment)


def load(path):
    record = json.loads(path.read_text())
    assert record["complete"] and record["candidate_count"] == 4 and record["action_horizon"] == 50
    rows = [dict(id=str(r["scene_seed"]), success=bool(r["success"]),
                 scores=np.asarray(r["per_query_mean"], dtype=np.float64))
            for r in record["per_episode"]]
    rows.sort(key=lambda r: r["id"])
    assert len(rows) == len({r["id"] for r in rows}) == 50
    assert sum(r["success"] for r in rows) == 31
    assert all(np.isfinite(r["scores"]).all() for r in rows)
    return rows, record


def main():
    assert not (OUT / "comparison.json").exists()
    sources = {
        "r3m_1000": OUT / "r3m_1000_scores.json",
        "r3m_10000": OLD / "baseline_r3m_outcome_scores.json",
        "imagenet_1000": OUT / "imagenet_1000_scores.json",
        "imagenet_10000": OLD / "outcome_scores.json",
    }
    inputs = {name: load(path) for name, path in sources.items()}
    ref = inputs["r3m_10000"]
    identities = [(r["id"], r["success"], len(r["scores"])) for r in ref[0]]
    assert all([(r["id"], r["success"], len(r["scores"])) for r in rows] == identities
               for rows, _ in inputs.values())
    assert len({record["candidate_manifest_sha256"] for _, record in inputs.values()}) == 1
    protocols = {
        "r3m_1000": json.loads((OUT / "r3m_1000_protocol.json").read_text()),
        "r3m_10000": json.loads((OUT / "r3m_10000_protocol.json").read_text()),
        "imagenet_1000": json.loads((OUT / "imagenet_1000_protocol.json").read_text()),
        "imagenet_10000": json.loads((OLD / "protocol.json").read_text()),
    }
    assert all(record["d_checkpoint_sha256"] == protocols[name]["d_checkpoint_sha256" if name.startswith("imagenet") else "checkpoint_sha256"]
               for name, (_, record) in inputs.items())
    results = {}
    for name, (rows, record) in inputs.items():
        item = alignment.analyze(rows, dict(source=str(sources[name]),
                                            checkpoint_sha256=record["d_checkpoint_sha256"]))
        for n in (1, 3, 5, 6):
            pos = np.asarray([r["scores"][:n].mean() for r in rows if r["success"]])
            neg = np.asarray([r["scores"][:n].mean() for r in rows if not r["success"]])
            assert len(pos) == 31 and len(neg) == 19
            item.setdefault("fixed_prefix", {})[str(n)] = alignment.summary(pos, neg)
        item["updates"] = 1000 if "1000" in name and not name.endswith("10000") else 10000
        item["final_train_loss"] = (protocols[name].get("training") or {}).get("final_minibatch_loss",
                               protocols[name].get("final_loss"))
        results[name] = item
    output = dict(complete=True, fixed_cohort="50 scene_seed=300000..300049, success31 failure19",
                  action_horizon=50, train_expert_episodes=50,
                  method="K4 raw D-logit mean at each policy query; each episode normalized to 101 progress points; equal episode weights",
                  models=results,
                  limits=["Only checkpoint step and frozen encoder vary in the four model comparison.",
                          "Every policy outcome trace and K4 action set is reused unchanged.",
                          "Full-trajectory separation includes post-success-time information: failures have 16 queries, successes only 6-8."])
    (OUT / "comparison.json").write_text(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    plt.rcParams.update({"font.family": "Arial Unicode MS", "axes.unicode_minus": False, "font.size": 11})
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey="row", layout="constrained")
    for i, encoder in enumerate(("r3m", "imagenet")):
        for j, step in enumerate((1000, 10000)):
            key = f"{encoder}_{step}"
            item = results[key]
            ax = axes[i, j]
            for group, label, color, linestyle in (("success", "成功", "#2166ac", "-"),
                                                    ("failure", "失败", "#d95f0e", "--")):
                data = item["groups"][group]
                ax.plot(alignment.GRID * 100, data["mean"], color=color, linestyle=linestyle,
                        linewidth=2, label=f"{label} n={data['episodes']}")
                ax.fill_between(alignment.GRID * 100, data["q25"], data["q75"], color=color, alpha=.12)
            auc = item["aligned_101_point_episode_mean"]["success_higher_auc"]
            ax.set_title(f"{'R3M' if encoder=='r3m' else '原版 ImageNet'} · {step} 步；成功高分 AUC {auc:.3f}")
            ax.set_xlim(0, 100)
            ax.grid(axis="y", alpha=.25)
            ax.spines[["top", "right"]].set_visible(False)
            ax.legend(frameon=False)
            ax.set_ylabel("原始 D logit，K4 平均")
            ax.set_xlabel("每条轨迹的归一化进度（%）")
    fig.suptitle("stack_blocks_two：同 50 条轨迹、同 K4 候选，比较训练 1,000 与 10,000 步", fontsize=15)
    fig.savefig(OUT / "step_comparison.png", dpi=180)
    fig.savefig(OUT / "step_comparison.pdf")
    for key, item in results.items():
        print(key, json.dumps({"train_loss": item["final_train_loss"],
                               "raw": item["raw_saved_observation_episode_mean"],
                               "aligned": item["aligned_101_point_episode_mean"],
                               "first3": item["fixed_prefix"]["3"]}))


if __name__ == "__main__":
    main()
