"""Offline reports: support suggestions are not observed success labels."""
from __future__ import annotations

import numpy as np
import torch

from .model import POLICY_BOOTSTRAP, action_distances, finite_guard


def distribution(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return None
    return {"count": int(values.size), "mean": float(values.mean()), "std": float(values.std()),
            "quantiles": np.quantile(values, [0, .05, .5, .95, 1]).tolist()}


def correlation(x, y):
    x, y = np.asarray(x), np.asarray(y)
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 2 or x[valid].std() < 1e-10 or y[valid].std() < 1e-10:
        return None
    return float(np.corrcoef(x[valid], y[valid])[0, 1])


@torch.no_grad()
def score_episode(model, target, arrays, kind, gamma, batch_size, device, score_limit,
                  bootstrap=POLICY_BOOTSTRAP):
    # Old checkpoints remain readable, but their TD metric must retain its meaning.
    if bootstrap not in (POLICY_BOOTSTRAP, "expert_next_action"):
        raise ValueError(f"Unsupported checkpoint bootstrap: {bootstrap}")
    model.eval()
    target.eval()
    fields = {key: torch.from_numpy(arrays[key].astype(np.float32)) for key in ("features", "state", "a_pi")}
    demo, policy, target_values, distances = [], [], [], []
    n = len(arrays["state"])
    for start in range(0, n, batch_size):
        sl = slice(start, start + batch_size)
        z, state, actions = (fields[key][sl].to(device) for key in ("features", "state", "a_pi"))
        q = model.score_many(z, state, actions)
        finite_guard({"q_pi": q}, score_limit)
        policy.append(q.cpu().numpy())
        if kind == "demo":
            expert = torch.from_numpy(arrays["a_demo"][sl]).to(device)
            q_e = model(z, state, expert)
            if bootstrap == POLICY_BOOTSTRAP:
                q_target_candidates = target.score_many(z, state, actions)
                finite_guard({"q_target_candidates": q_target_candidates}, score_limit)
                q_target = q_target_candidates.mean(1)
            else:
                q_target = target(z, state, expert)
            finite_guard({"q_demo": q_e, "q_target": q_target}, score_limit)
            demo.append(q_e.cpu().numpy())
            target_values.append(q_target.cpu().numpy())
            distances.append(action_distances(actions, expert, model.action_scale).cpu().numpy())
    q_pi = np.concatenate(policy)
    result = {"frame_index": arrays["frame_index"], "q_pi": q_pi,
              "q_pi_mean": q_pi.mean(1), "q_pi_p05": np.quantile(q_pi, .05, axis=1),
              "q_pi_p95": np.quantile(q_pi, .95, axis=1), "q_pi_min": q_pi.min(1), "q_pi_max": q_pi.max(1),
              "candidate_std_mean": arrays["a_pi"].std(1).mean(1)}
    if kind == "demo":
        q_demo, distance, q_target = np.concatenate(demo), np.concatenate(distances), np.concatenate(target_values)
        farthest = distance.argmax(1)
        td_target = np.full(n, np.nan)
        valid = np.flatnonzero(arrays["valid_transition"])
        td_target[valid] = 1 + gamma * q_target[valid + 1]
        result.update({"q_demo": q_demo, "distance": distance, "farthest_index": farthest,
                       "farthest_distance": distance[np.arange(n), farthest],
                       "q_farthest": q_pi[np.arange(n), farthest], "td_target": td_target,
                       "td_squared_error": (q_demo - td_target) ** 2, "gap": q_pi.mean(1) - q_demo})
    else:
        result["control_source"] = arrays["control_source"]
    return result


def summarize(episode_results):
    def joined(key):
        return np.concatenate([r[key].reshape(-1) for r in episode_results if key in r]) if any(key in r for r in episode_results) else np.empty(0)
    metrics = {key: distribution(joined(key)) for key in ("q_demo", "q_pi", "q_farthest", "gap", "td_squared_error", "candidate_std_mean", "farthest_distance")}
    metrics["td_mse"] = metrics["td_squared_error"]["mean"] if metrics["td_squared_error"] else None
    distances = joined("distance")
    corresponding_scores = np.concatenate([r["q_pi"].reshape(-1) for r in episode_results if "distance" in r]) if len(distances) else np.empty(0)
    metrics["distance_score_pearson"] = correlation(distances, corresponding_scores)
    metrics["episodes"] = len(episode_results)
    metrics["frames"] = sum(len(r["frame_index"]) for r in episode_results)
    return metrics


def report_rows(result, entry, suite, checkpoint_label):
    n = len(result["frame_index"])
    rows = []
    scalar_keys = [key for key, value in result.items() if value.ndim == 1]
    for i in range(n):
        row = {"checkpoint": checkpoint_label, "suite": suite, "episode": entry["id"], "kind": entry["kind"]}
        for key in scalar_keys:
            value = result[key][i].item()
            row[key] = value if np.isfinite(value) else None
        row["candidate_scores"] = result["q_pi"][i].tolist()
        row["candidate_distances"] = result["distance"][i].tolist() if "distance" in result else None
        row["supervisor_label"] = entry.get("metadata", {}).get("supervisor_label")
        rows.append(row)
    return rows


def plot_episode(result, entry, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x = result["frame_index"]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(x, result["q_pi_mean"], label="policy candidate mean")
    ax.fill_between(x, result["q_pi_p05"], result["q_pi_p95"], alpha=.2, label="candidate 5–95%")
    if "q_demo" in result:
        ax.plot(x, result["q_demo"], label="recorded demo target")
        ax.plot(x, result["q_farthest"], alpha=.7, label="farthest candidate")
    else:
        mask = result["control_source"] == 1
        boundaries = np.flatnonzero(np.diff(np.r_[False, mask, False]))
        for j, (start, stop) in enumerate(zip(boundaries[::2], boundaries[1::2], strict=True)):
            ax.axvspan(x[start] - .5, x[stop - 1] + .5, alpha=.15, color="orange", label="HIL active" if j == 0 else None)
        for index in np.flatnonzero(np.diff(mask)) + 1:
            ax.axvline(x[index], linestyle=":", color="grey", alpha=.6)
    outcome = entry.get("metadata", {}).get("supervisor_label", "demonstration")
    ax.set(title=f"{entry['id']} | recorded outcome: {outcome}", xlabel="Saved frame index (not uniform elapsed time)", ylabel="Support C (not success probability)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_summary(results, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    for key, label in (("q_pi", "policy candidates"), ("q_demo", "recorded demo"), ("q_farthest", "farthest")):
        values = [r[key].reshape(-1) for r in results if key in r]
        if values:
            ax.hist(np.concatenate(values), bins=50, density=True, alpha=.4, label=label)
    ax.set(xlabel="Support C", ylabel="Density")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "score_distribution.png", dpi=140)
    plt.close(fig)
    demo_results = [r for r in results if "distance" in r]
    if demo_results:
        x = np.concatenate([r["distance"].reshape(-1) for r in demo_results])
        y = np.concatenate([r["q_pi"].reshape(-1) for r in demo_results])
        indices = np.linspace(0, len(x) - 1, min(10000, len(x)), dtype=int)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.scatter(x[indices], y[indices], s=3, alpha=.2)
        ax.set(xlabel="Normalized action distance to recorded demo", ylabel="Support C")
        fig.tight_layout()
        fig.savefig(output / "distance_vs_score.png", dpi=140)
        plt.close(fig)
