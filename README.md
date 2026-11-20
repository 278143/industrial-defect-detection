# Industrial Defect Detection

工业视觉缺陷检测系统，基于 MVTec AD 数据集开展正常/异常判定与缺陷定位。

当前处于 V0.1：已完成工程与算法环境初始化，提供数据完整性检查和标注预览脚本。模型训练、推理和界面尚未实现。

首版按项目规划使用 `metal_nut`（金属螺母），计划以 PatchCore 建立基线，后续增加评估、异常热力图、批量推理和 Streamlit 界面。

## 开发环境

- Python 3.11
- uv
- Git

在项目根目录执行：

```powershell
uv sync --locked
uv run python -c "import defect_detection; print(defect_detection.__file__)"
uv run ruff check .
```

`uv.lock` 记录开发依赖版本，`.venv/` 是本地虚拟环境。已配置 CUDA 12.6 版 PyTorch、Anomalib 和图像处理依赖，GPU 运行已通过验证。

为兼容 Windows 上 Python 3.11 对中文路径的处理，开发安装使用 Hatch 的精确导入模式。IDE 如无法定位包，可将 src/ 配置为源码搜索目录。

## 项目目录

| 目录 | 用途 |
| --- | --- |
| `configs/` | 数据和模型配置 |
| `src/defect_detection/` | 数据处理、模型、推理、评估和可视化 |
| `scripts/` | 数据准备、训练、评估和预测入口 |
| `app/` | Streamlit 应用 |
| `tests/` | 自动化测试 |
| `docs/` | 项目规划与技术文档 |
| `assets/` | 演示素材 |
| `data/` | 本地数据集，参见其中的 README |
| `outputs/` | 运行时生成的结果、报告和日志 |

Git 不跟踪本地数据集、虚拟环境、模型权重和运行结果。部分待实现目录使用 `.gitkeep` 保留结构。

## 数据检查与可视化

在项目根目录执行：

```powershell
uv run scripts/prepare_data.py
uv run pytest
```

默认检查 `data/metal_nut/`，配置见 `configs/dataset.yaml`。脚本检查 PNG 图片解码、尺寸统计、异常测试图与掩膜的一一对应、掩膜尺寸及二值像素；发现错误时返回非零退出码。

输出保存在 `outputs/reports/`：

- `metal_nut_inspection.json`：数量、图片尺寸和错误报告。
- `metal_nut_preview.png`：每个测试类别各一张原图、真实标注掩膜和标注叠加图；这些是数据标注预览，不是模型预测。

可用 `--category screw` 检查螺丝类别。脚本不移动、修改或重新划分原始数据；检查通过也不代表模型已训练或数据泄漏检查已完成。

## 正常验证集划分与重复检查

```powershell
uv run scripts/split_data.py
```

设置见 `configs/dataset.yaml`：验证比例为 `0.2`、随机种子为 `42`。只从原始 `train/good/` 留出正常验证图，测试图片和标注全部保留；不移动、不复制或修改原始数据。

- `configs/splits/metal_nut.json`：固定划分清单，可提交 Git。图片和掩膜路径均相对于配置中的数据根目录；包含数据内容指纹。
- `outputs/reports/metal_nut_split_audit.json`：原始数量、实际划分数量、重复分组和泄漏检查报告。

重复检查同时比较文件 SHA-256 和解码后像素 SHA-256，因此重新压缩但像素相同的图片也可识别。训练来源内部的重复图片作为一组划分，数量可能因此略偏离目标比例。训练来源与测试集出现像素重复，或相同图片对应冲突类别时，程序返回失败，不生成新的可用清单。同一测试集内部的同类重复会记录在报告中，不自动删除。

重复运行且数据、参数相同时，保留相同清单；已有清单与新结果不同则默认拒绝覆盖。确认数据或参数确实要变更后，才使用 `--force`。发生失败后应查看最新审计报告，不能继续使用旧清单。该检查只覆盖完全相同的文件或像素，不覆盖近似图片、同一实物的相邻拍摄帧等关联样本。

后续模型必须按清单读取训练、验证和测试图片。正常验证集用于正常分数分布与误报率分析，缺少异常样本，不能用于可靠估计异常召回率或选择最佳 F1。测试集保留用于最终评估。

## 下一步

1. 核对数据检查报告和标注预览。
2. 核对固定划分清单和重复检查报告。
3. 跑通 PatchCore 基线，保存异常分数和异常热力图。
