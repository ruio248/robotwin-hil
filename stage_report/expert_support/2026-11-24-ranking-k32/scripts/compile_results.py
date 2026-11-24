#!/usr/bin/env python3
"""Validate three matched-SFT experiments and write a compact result record."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from run_online20 import OUT, REPO, SPECS
from train import digest


def main():
    results = {}
    for task, spec in SPECS.items():
        model_root = OUT / 'matched_sft_models' / task
        report = json.loads((model_root / 'report.json').read_text())
        outcome = json.loads((model_root / 'sft_outcomes.json').read_text())
        online_root = OUT / task / 'online_k32_20'
        summary = json.loads((online_root / 'summary.json').read_text())
        trace = json.loads((online_root / 'trace/manifest.json').read_text())
        scenes = json.loads((online_root / 'scene_manifest.json').read_text())['episodes']
        baseline = json.loads(spec['baseline'].read_text())
        baseline_rows = [e for e in baseline['episodes'] if e.get('counted_in_official_result')][:20]
        online_rows = [e for e in trace['episodes'] if e.get('counted_in_official_result')]
        expected = [(e['scene_seed'], e['instruction']) for e in scenes]
        if (report['task'] != task or len(report['test20']) != 20 or outcome['task'] != task
                or summary['task'] != task or trace['status'] != 'complete' or len(online_rows) != 20
                or trace['infra_error_episodes'] != 0 or trace['query_covered_official_episodes'] != 20
                or [(e['scene_seed'], e['instruction']) for e in online_rows] != expected
                or [(e['scene_seed'], e['instruction']) for e in baseline_rows] != expected):
            raise ValueError(f'Incomplete or mismatched experiment: {task}')
        before = np.asarray([bool(e['success']) for e in baseline_rows], dtype=np.int8)
        after = np.asarray([bool(e['success']) for e in online_rows], dtype=np.int8)
        if (before.sum() != summary['baseline_successes']
                or after.sum() != summary['k32_min_support_successes']):
            raise ValueError(f'Summary and trace counts differ: {task}')
        decisions = [json.loads(line) for line in (online_root / 'decisions.jsonl').read_text().splitlines()]
        if len(decisions) != summary['decisions'] or any(len(row['scores']) != 32 for row in decisions):
            raise ValueError(f'Incomplete K32 decision audit: {task}')
        policy_name = 'Pi_05' if task == 'place_empty_cup' else 'Pi_05_RobotTwin'
        eval_root = (REPO / 'outputs/place_empty_cup_3epoch_20261122/eval_runtime'
                     if task == 'place_empty_cup' else
                     Path('/hdd/h10_dagger_lora_5050_68_ubuntu_20261115/runtime/eval_runtime'))
        run_id = f'support_{task}_k32_20_20261123'
        official_files = list((eval_root / 'eval_result' / task / policy_name / 'demo_clean' / run_id).glob('*/_result.txt'))
        if len(official_files) != 1:
            raise ValueError(f'Official result missing or ambiguous: {task}')
        official_rate = float(official_files[0].read_text().strip().splitlines()[-1])
        if abs(official_rate - after.mean()) > 1e-9:
            raise ValueError(f'Official result differs from trace: {task}')
        rng = np.random.default_rng(42)
        delta = after - before
        ci = np.quantile(delta[rng.integers(20, size=(10000, 20))].mean(axis=1), [.025, .975])
        test = report['test20']
        mean = lambda key: float(np.mean([row[key] for row in test]))
        dmean = lambda key: float(np.mean([row[key] for row in decisions]))
        results[task] = {
            'horizon': spec['horizon'], 'training_report_sha256': digest(model_root / 'report.json'),
            'checkpoint_sha256': report['checkpoint_sha256'], 'test20_candidate_count': 4,
            'test20_expert_rank': mean('expert_rank'),
            'test20_expert_logit': mean('expert_logit_mean'),
            'test20_policy_logit': mean('policy_logit_mean'),
            'test20_candidate_pair_gap_mean': mean('candidate_pair_gap_mean'),
            'test20_beta1_ess_over_k': mean('ess_over_k_beta1'),
            'test20_beta1_weighted_joint_shift_mrad': mean('weighted_joint_rms_mrad_beta1'),
            'sft_successes_50': outcome['successes'], 'sft_failures_50': outcome['failures'],
            'sft_success_score_mean': outcome['success_score_mean'],
            'sft_failure_score_mean': outcome['failure_score_mean'],
            'sft_success_auc': outcome['success_auc'], 'sft_success_auc_ci95': outcome['auc_ci95'],
            'online_count': 20, 'online_baseline_successes': int(before.sum()),
            'online_min_support_successes': int(after.sum()),
            'online_paired_change': float(delta.mean()), 'online_paired_change_ci95': ci.tolist(),
            'online_success_to_failure': int(np.sum((before == 1) & (after == 0))),
            'online_failure_to_success': int(np.sum((before == 0) & (after == 1))),
            'online_decisions': len(decisions), 'online_k32_beta1_ess_over_k': dmean('ess_over_k_beta1'),
            'online_k32_score_range': dmean('score_range'),
            'online_k32_min_minus_mean_score': float(np.mean([
                row['min_support_score'] - row['candidate_score_mean'] for row in decisions])),
            'online_k32_selected_joint_shift_mrad': dmean('joint_rms_mrad'),
            'official_result': str(official_files[0]),
            'trace_sha256': digest(online_root / 'trace/manifest.json'),
            'decisions_sha256': digest(online_root / 'decisions.jsonl'),
        }
    target = OUT / 'final_summary.json'
    target.write_text(json.dumps(results, indent=2, ensure_ascii=False) + '\n')
    lines = [
        '# 三任务专家支撑函数实验结果（2026-11-23）', '',
        '所有在线结果使用对应任务的本地 SFT checkpoint、冻结的前 20 条 SFT scene seed 与指令；每次决策从同一观察采样 32 个完整 chunk，直接执行原始支撑 logit 最低者。', '',
        '| 任务 | test20 专家排序（K4） | SFT 成功/失败平均分 | 成功 AUC（50） | SFT→最低支撑（20） | 配对变化 | 成功→失败 / 失败→成功 | K32 ESS/K（β=1） | K32 动作偏移 mrad |',
        '| --- | ---: | --- | ---: | --- | ---: | --- | ---: | ---: |',
    ]
    for task, row in results.items():
        lines.append(
            f'| {task} | {row["test20_expert_rank"]:.3f} | '
            f'{row["sft_success_score_mean"]:.3f} / {row["sft_failure_score_mean"]:.3f} | '
            f'{row["sft_success_auc"]:.3f} | '
            f'{row["online_baseline_successes"]}/20 → {row["online_min_support_successes"]}/20 | '
            f'{row["online_paired_change"]:+.1%} | '
            f'{row["online_success_to_failure"]} / {row["online_failure_to_success"]} | '
            f'{row["online_k32_beta1_ess_over_k"]:.4f} | '
            f'{row["online_k32_selected_joint_shift_mrad"]:.3f} |')
    lines.extend(['', '## 逐任务诊断', ''])
    for task, row in results.items():
        ci = row['online_paired_change_ci95']
        lines.extend([
            f'### {task}', '',
            f'- test20：专家/策略平均 logit {row["test20_expert_logit"]:.3f} / {row["test20_policy_logit"]:.3f}；'
            f'K4 同观察候选两两平均分差 {row["test20_candidate_pair_gap_mean"]:.4f}；'
            f'β=1 ESS/K {row["test20_beta1_ess_over_k"]:.4f}；加权关节 RMS 偏移 '
            f'{row["test20_beta1_weighted_joint_shift_mrad"]:.3f} mrad。',
            f'- 50 条 SFT rollout：成功 {row["sft_successes_50"]}、失败 {row["sft_failures_50"]}；'
            f'成功分 AUC {row["sft_success_auc"]:.3f}（bootstrap 95% 区间 '
            f'[{row["sft_success_auc_ci95"][0]:.3f}, {row["sft_success_auc_ci95"][1]:.3f}]）。',
            f'- 在线 K32：{row["online_decisions"]} 次决策，候选平均分数范围 '
            f'{row["online_k32_score_range"]:.4f}，β=1 ESS/K '
            f'{row["online_k32_beta1_ess_over_k"]:.4f}；最低分候选相对候选均值的关节 RMS 偏移 '
            f'{row["online_k32_selected_joint_shift_mrad"]:.3f} mrad。配对成功率变化的 bootstrap 95% 区间 '
            f'[{ci[0]:+.0%}, {ci[1]:+.0%}]。', '',
        ])
    lines.extend(['## 解释边界', '',
                  '成功/失败平均分是每条完整 SFT rollout 的查询均分；它检验跨状态结局分离，不能代替同状态候选的因果排序。最低支撑使 cup 成功率从 16/20 降至 14/20，但 click 与 shoes 分别升至 15/20、9/20，因此三任务没有一致的下降证据。在线 20 场景只估计这组冻结场景的变化，置信区间应与点估计一起阅读。', '',
                  '训练报告、checkpoint、场景清单、逐决策日志及官方结果的路径和哈希保存在 `outputs/support_three_tasks_20261123/final_summary.json`。', ''])
    document = REPO / 'docs/support_three_tasks_20261123_results.md'
    document.write_text('\n'.join(lines))
    print(target, document)


if __name__ == '__main__':
    main()
