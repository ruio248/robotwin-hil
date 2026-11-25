"""Report frozen-D reward migration, held-out action variation and paired rollout outcomes."""
import json
import numpy as np
from run import ROOT,write

def read(name):
    return json.loads((ROOT/name).read_text())

def main():
    report=read('training.json')
    online=read('online.json')
    modes=['native','random','max','min']
    results={}
    for mode in modes:
        rows=online['runs'][mode]['episodes']
        assert min(r['queries'] for r in rows)>=5
        results[mode]={}
        for key in ['success','steps','discounted_reward','lifted','reached_target_geometry']:
            results[mode][key]=[float(r[key]) for r in rows]
        results[mode]['first5']=[float(np.mean(r['reward_trace'][:5])) for r in rows]
        results[mode]['raw_first5']=[float(np.mean(r['raw_reward_trace'][:5])) for r in rows]
    rng=np.random.default_rng(42)
    draws=rng.integers(20,size=(20000,20))
    diffs={}
    for a,b in [('max','min'),('max','random'),('min','random')]:
        diffs[a+'-'+b]={}
        for key in results[a]:
            d=np.array(results[a][key])-results[b][key]
            diffs[a+'-'+b][key]=dict(mean=d.mean(),ci95=np.quantile(d[draws].mean(1),[.025,.975]).tolist())
    means={m:{k:float(np.mean(v)) for k,v in values.items()} for m,values in results.items()}
    write(ROOT/'fixed_window.json',dict(window=5,means=means,paired_differences=diffs))
    audit=read('input_audit.json')
    profiles=read('query_rewards.json')
    raw=np.concatenate([r['raw'] for r in profiles])
    offline=read('offline_candidates/summary.json')['means']
    lines=['# place_empty_cup 配对占据奖励与 Native TD','',
           f"冻结 C_paired。{len(audit)} 对查询首帧的 RGB、状态和 logit 逐对一致。H10 SFT 查询级转移共 {len(raw)} 条，其中训练 {report['q_train_transitions']} 条、验证 {report['q_validation_transitions']} 条。下一状态原始奖励均值 {raw.mean():.3f}，达到上限 5 的比例 {(raw>=5).mean():.1%}。折扣 0.9，Q 和 V 各训练 10k 步。",'']
    for k in ('q','v'):
        best=min(report[k+'_curve'],key=lambda x:x['validation']['mc_rmse'])
        lines.append(f"- {k.upper()} 的最优步数 {best['step']}，训练 / 验证 MC RMSE 为 {best['train']['mc_rmse']:.3f} / {best['validation']['mc_rmse']:.3f}。")
    lines+=['',f"四条验证轨迹各取三个观察。K32 平均极差 {offline['range']:.4f}，两两平均绝对分差 {offline['pair_gap']:.4f}，β=1 的 ESS/K 为 {offline['ess_over_k']:.5f}。",'',
            '20 个闭环场景与训练及选模场景隔离，沿用 V1 的评测种子。Native 复用 V1 实际轨迹并重新打分。其余三组使用新 Q 全程运行，每次查询采样 32 个候选。固定窗口取每个回合前五个真实查询到达状态，每场景等权。','',
            '| 模式 | 成功数 / 20 | 平均动作数 | 前五次平均奖励 | 前五次平均原始奖励 | 折扣回报 | 抬杯数 | 对准数 |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for m,x in means.items():
        lines.append(f"| {m} | {20*x['success']:.0f} | {x['steps']:.1f} | {x['first5']:.3f} | {x['raw_first5']:.3f} | {x['discounted_reward']:.2f} | {20*x['lifted']:.0f} | {20*x['reached_target_geometry']:.0f} |")
    lines+=['','按场景配对 bootstrap，以下为前五次平均奖励差及 95% 区间。','']
    for pair,x in diffs.items():
        v=x['first5']
        lines.append(f"- {pair} 为 {v['mean']:+.3f}，区间 [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]。")
    lines+=['','四个开发场景同时用于 checkpoint 选择。每种方法每个评测场景各运行一次。逐回合几何量、原始奖励、截断奖励及成功标记见 [online.json](online.json)，配对统计见 [fixed_window.json](fixed_window.json)。']
    (ROOT/'RESULTS.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':
    main()
