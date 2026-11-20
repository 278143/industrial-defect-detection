# 本地数据集

本项目使用 MVTec Anomaly Detection（MVTec AD）数据集。

官方下载与许可说明：https://www.mvtec.com/company/research/datasets/mvtec-ad

数据集不纳入 Git，也不随项目重新分发。下载后将类别目录放在本目录下，例如：

```text
 data/
 └── metal_nut/
     ├── train/good/
     ├── test/
     │   ├── good/
     │   └── <缺陷类型>/
     └── ground_truth/
         └── <缺陷类型>/
```

- `train/good/`：正常训练图片。
- `test/`：正常和异常测试图片。
- `ground_truth/`：异常测试图片对应的像素级掩膜，白色表示缺陷。
- 例如 `test/scratch/000.png` 对应 `ground_truth/scratch/000_mask.png`。

首版使用 `metal_nut`（金属螺母）；`screw` 是螺丝，属于另一个类别。

使用 `uv run scripts/split_data.py` 从正常训练图片中留出验证样本，固定划分保存在 `configs/splits/<类别>.json`。默认比例为 20%，随机种子为 42；原始目录不改变，后续模型按清单读取。不得使用最终测试集选择阈值或调节模型参数。

保留数据包附带的原始 `readme.txt` 和 `license.txt`，使用前阅读官方许可。
