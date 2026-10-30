# SFT 训练

## 基线配置

本项目的 `handover_to_tray_v2_promptfix` Pi0.5 配置见 `robotwin_hil/compat/XPolicyLab/policy/Pi_05_RobotTwin/openpi/src/openpi/training/config.py`，服务端配置移植见 `robotwin_hil/local_serving/port_openpi_config.py`。训练使用三路相机、14 维状态/动作、动作 horizon `H=10`；十二个手臂关节在训练适配器中转成 delta，模型输出在执行前还原为绝对关节目标。`prompt_from_task=True`，因此任务指令及其修正必须随数据和服务端保持一致。

原先用于 HG-DAgger 方法验证的 SFT 基线按实验记录为 `discrete_state_input=False`、Pi0.5 预训练权重继续**全参数**微调；配置中 `batch_size=128`、`num_train_steps=10000`、`save_interval=1000`、`keep_period=5000`。但这个配置块**未设置 `weight_loader`**，其类默认是 `NoOpWeightLoader`；因此仅执行配置名并不能证明加载了预训练权重。复现时必须从实际训练命令、权重加载日志和 checkpoint 元数据核实初始化来源，并显式提供对应加载器。服务时常用的 `..._9999` 是该轮训练保存的索引，不要从名称直接推断另一套训练协议。后续其他任务曾采用 `discrete_state_input=True`、不同数据/学习率计划及 5k 训练；它们是独立实验，不能写成原基线的继续训练。

## 运行检查

1. 在记录中明确：训练数据集及 split、预训练权重加载器与参数路径、训练配置名、`discrete_state_input`、H、batch、梯度累积/FSDP、seed、学习率计划、优化器、EMA/LoRA 开关、目标步数与保存规则。默认配置不等于某次实际命令行覆盖值。
2. 运行 `openpi/scripts/train.py` 前检查 LeRobot schema、三路图像键、14 维关节、动作单位和提示词；先做短步数 smoke，检查首批前向/反向与保存/加载。
3. 正式训练时保留完整命令、源码提交、日志、checkpoint、训练态、assets 与 `norm_stats`。每次评测必须加载该 checkpoint 对应的配置和统计量；尤其不能用 `state=False` 的服务配置去评估 `state=True` 权重。
4. 检查末步日志有无 NaN/OOM，checkpoint 中 `params`、所需训练态、assets 与 `norm_stats` 是否完整。训练 loss 仅说明拟合过程，不替代仿真成功率。

复现实例以 [全流程说明](../tasks/HANDOVER_TO_TRAY_FULL_PIPELINE.md) 和具体实验目录的启动命令为准。此处不绑定机器绝对路径，也不把某台机器的 GPU 数写成模型本身的要求。
