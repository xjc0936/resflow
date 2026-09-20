# ResFlow：论文级非官方复现

这是 CVPR 2025 论文 [Reversing Flow for Image Restoration](https://openaccess.thecvf.com/content/CVPR2025/html/Qin_Reversing_Flow_for_Image_Restoration_CVPR_2025_paper.html) 的 PyTorch 非官方复现。实现覆盖论文正文和[补充材料](https://openaccess.thecvf.com/content/CVPR2025/supplemental/Qin_Reversing_Flow_for_CVPR_2025_supplemental.pdf)中的训练、四步 Euler 推理、全分辨率测试和指标计算。

与只实现公式的演示代码不同，这个仓库提供：

- DDPM 256 U-Net 主干、辅助变量 Adapter、零初始化和自适应调制；
- 完整的联合速度目标 `(v_x,v_y)`、熵保持辅助路径及时间加权损失；
- 单卡或 `torchrun` 八卡训练、断点续训、余弦学习率和原子化 checkpoint；
- Snow100K-L、RealSnow、Outdoor-Rain/LHP、Dense-Haze、NH-HAZE、SIDD、DPDD、JPEG QF=10 配置；
- 全分辨率四步恢复以及 PSNR、SSIM、MAE、LPIPS 评测；
- 对论文未披露和相互矛盾设置的逐项审计：[REPRODUCIBILITY.md](REPRODUCIBILITY.md)。

## 安装

建议使用 Python 3.10–3.12 与支持 CUDA 的 PyTorch：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[metrics,test]'
```

## 数据目录

默认配置使用统一的成对目录，LQ 和 HQ 文件按相对路径及文件名匹配：

```text
data/Snow100K-L/
├── train/
│   ├── LQ/
│   └── HQ/
└── test/
    ├── LQ/
    └── HQ/
```

其他数据集采用相同的 `train/{LQ,HQ}`、`test/{LQ,HQ}` 结构。原始文件名不一致时，可在 YAML 中设置 `strip_lq_suffixes` / `strip_hq_suffixes`，或者提供含 `lq,hq` 两列的 CSV `manifest`。数据集需按各自许可从官方来源取得，本仓库不自动下载或重新切分。

论文使用的数据与规模为：Snow100K-L 50,000/50,000、RealSnow 61,500 crops/240、Outdoor-Rain 8,100/900、LHP 300 test、Dense-Haze 49/6、NH-HAZE 49/6、SIDD 288/32、DPDD 350/76。JPEG 训练集合并 DIV2K 900 与 Flickr2K 2,650，测试为 LIVE1 29 和 BSD500 500；将训练图像放入 `data/JPEG/train/HQ`，测试配置可复制 `jpeg_q10.yaml` 并切换 HQ 路径。

## 训练

论文硬件配置（八张 A100，每个数据集独立训练 400K iterations）：

```bash
bash scripts/train_8gpu.sh configs/snow100k_l.yaml runs/snow100k_l
```

单卡调试：

```bash
python -m resflow.train \
  --config configs/snow100k_l.yaml \
  --output runs/snow100k_l_debug \
  --set train.iterations=20 \
  --set train.global_batch_size=1 \
  --set model.base_channels=32 \
  --set 'model.channel_multipliers=[1,2,2]'
```

续训：

```bash
bash scripts/train_8gpu.sh configs/snow100k_l.yaml runs/snow100k_l \
  --resume runs/snow100k_l/latest.pt
```

## 推理与评测

```bash
python -m resflow.infer \
  --config configs/snow100k_l.yaml \
  --checkpoint runs/snow100k_l/latest.pt \
  --output outputs/snow100k_l

python -m resflow.evaluate \
  --config configs/snow100k_l.yaml \
  --restored outputs/snow100k_l \
  --lpips \
  --output outputs/snow100k_l/metrics.json
```

默认严格使用论文的 4 个均匀时间步和 Euler 积分。`--steps` 可用于复现实验中的 1/2/4 步曲线，但主表应保持 4。

## 论文目标值

主要合成数据结果用于完整训练后的 sanity check：

| 数据集 | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| Snow100K-L | 31.86 | 0.917 | 0.030 |
| Outdoor-Rain | 32.82 | 0.936 | 0.0514 |
| Dense-Haze | 17.12 | 0.59 | — |

这些数字是论文报告值，不是本仓库实测值。由于原文没有公开所有结构、优化器和评测细节，不能诚实地承诺逐位复现；所有差异风险都列在复现审计中。

默认网络共有 129,930,246 个参数，其中不含辅助 Adapter/调制层的 DDPM 主干约 113.68M，与 DDPM 官方 256×256 模型标注的 114M 一致。各 YAML 也会检查论文给出的数据集样本数，避免误用 split；有意做小规模调试时可用 `--set data.train.expected_count=null` 关闭检查。

## 验证

```bash
python -m pytest -q
```

测试覆盖论文辅助路径及导数、联合速度目标、损失权重端点、Euler 反推方向、U-Net/Adapter 输出形状和零初始化。正式训练前建议先使用上面的单卡调试命令跑通本机数据路径。
