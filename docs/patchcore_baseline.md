# PatchCore 基线

## 运行

```powershell
uv run scripts/train.py
```

配置：`configs/patchcore.yaml`；数据位置和类别：`configs/dataset.yaml`。原始图像保持不动，训练、验证和测试按 `configs/splits/metal_nut.json` 加载。

## 数据隔离

`validate_manifest` 重新计算当前原始数据的文件、像素及掩膜指纹，并重新建立相同划分，与固定清单逐项比较；数据变化、路径变化、清单被修改或集合间重复会阻止运行。

只用 176 张正常训练图建立特征库，44 张正常验证图用于阈值校准，115 张测试图只在阈值冻结后进行评估。加载器不会通过遍历整个 `train/good` 来构建训练批次。

## 算法和预处理

使用 Anomalib 2.7.1 的底层 `PatchcoreModel`，显式组织提取特征、coreset 采样和预测流程，以避免框架默认验证划分和后处理改变本项目的数据协议。

- torchvision ResNet-18，`IMAGENET1K_V1` 预训练权重。
- 提取 `layer2` 和 `layer3`，正常特征采样比例 1%，9 个邻居。
- 原图整体缩放至 224×224，不裁剪，OpenCV `INTER_AREA`。
- RGB、像素除以 255、ImageNet 均值与标准差归一化。
- 掩膜用最近邻缩放至 224×224，像素大于 0 作为缺陷。
- 固定随机种子 42，FP32，batch size 4，CUDA。

这是针对 6 GB 显存的初始基线，不是论文默认 WideResNet 配置的复现。

## 阈值与指标

保留原始异常分数，不将其解释为概率。图像阈值来自 44 个正常验证分数的 95% 分位数（`method=higher`）；像素阈值来自每张正常验证图异常图最大值的相同分位数。判定均使用严格大于阈值。

正常验证集不能估计异常召回率或确定最佳 F1。44 张图片得到的经验误报率也不保证生产环境误报率。测试集没有用于阈值扫描或优化。

报告图像 AUROC、固定阈值下的 Precision/Recall/F1、误报率、漏检率和混淆矩阵。像素 AUROC 基于缩放后的 224×224 标注，不能直接等同于原始 700×700 分辨率的官方评估结果。

测试耗时包括图片读取、预处理、推理和传回 CPU；没有单独预热，不作为正式延迟 benchmark。

## 输出

默认保存到 `outputs/patchcore/metal_nut_resnet18/`：

- `model.pt`：模型参数、正常特征记忆库、配置、数据指纹与阈值。
- `thresholds.json`：只用正常验证图校准的阈值。
- `validation_scores.npz`：44 张验证图的原始分数与路径。
- `test_predictions.csv`：每张测试图的分数、固定阈值和判定。
- `test_predictions.npz`：测试分数、异常图、缩放掩膜、标签和图片路径。
- `metrics.json`：指标、配置、版本、数据指纹、运行环境及局限。
- `<类别>_<编号>.png`：每个测试类别一张原图、预测热力图和叠加图。

首次运行从 torchvision 官方下载 ResNet-18 权重到 `outputs/cache/torch/`。所有权重与运行结果均留在被 Git 忽略的 `outputs/`。

已有完整结果时，重新训练需要在配置中设置新的输出目录，避免覆盖。加载可信的本地 checkpoint 复核推理可执行：

```powershell
uv run scripts/train.py --checkpoint outputs/patchcore/metal_nut_resnet18/model.pt
```

重载时检查配置和数据指纹；重复评估写入输出目录的 `reload_check/` 子目录，保留初次训练结果；不重新建立记忆库，不根据测试集重选阈值。

## 首次基线结果

2026-11-21：176 张训练、44 张正常验证、115 张测试。图像 AUROC 0.994624，224×224 像素 AUROC 0.982538；正常验证阈值固定后 Precision/Recall/F1 均为 0.978495。混淆矩阵 TN=20、FP=2、FN=2、TP=91。测试正常样本仅 22 张，误报率为 2/22，结果不能直接推广至真实产线。记忆库包含 1379 条 384 维特征。完整数值和环境记录在本地 metrics.json。
