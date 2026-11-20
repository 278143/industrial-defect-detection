# Industrial Defect Detection

工业视觉缺陷检测系统，基于 MVTec AD 数据集开展正常/异常判定与缺陷定位。

当前处于 V0.1：已整理项目目录和数据集，正在准备开发环境与数据读取流程。模型、推理和界面尚未实现。

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

`uv.lock` 记录开发依赖版本，`.venv/` 是本地虚拟环境。PyTorch、Anomalib 和其他算法依赖将在确认兼容版本后加入配置。

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

## 下一步

1. 完成 GitHub 仓库关联和首次提交。
2. 配置 PyTorch、OpenCV 和 Anomalib。
3. 检查 `metal_nut` 图片、掩膜及文件对应关系。
4. 固定随机种子，从正常训练样本中留出验证集，保持测试集独立。
5. 跑通 PatchCore 基线，保存异常分数和异常热力图。
