# V1 C++20 推理库与 DAT 接入

**状态：已完成并通过验收。** 2026-11-22，在 RTX 5080（Intel Core Ultra 7 265KF）机器上，
C++20 静态库和命令行工具直接读取 Himawari HSD/DAT，输出 **降压 dP（hPa）**；模型为
[V1](../V1/README.md)（全体折外 RMSE 8.901 hPa），不是中心气压预测，也不是新模型。

## 结论

| 验收项 | 门槛 | 实测 | 结果 |
|---|---|---|---|
| 546 条缓存特征 OOF RMSE 差（C++−Python） | ≤ 0.35 hPa | 3.7×10⁻⁸ hPa，逐条最大差 6.2×10⁻⁶ hPa | 通过 |
| 329 条从本地 DAT 端到端 OOF RMSE 差 | ≤ 0.35 hPa | 1.6×10⁻⁸ hPa，C++ RMSE 8.9348783 | 通过 |
| 总耗时比（C++ 读 DAT 到 dP / Python-offline 预处理到描述子） | ≤ 1.5× | **1.125×**（41.48 s / 36.86 s） | 通过 |
| 单记录 p95 / 最坏耗时比 | — | 1.238× / 1.289× | 记录 |

`passed: true`，逐条数据在 `/mnt/data/H9dP/runs/v1/rotation_seed1/cpp_inference/acceptance_final.json`
与 `acceptance_final.csv`，日志为同目录 `acceptance_final.log`。

## 输入与调用

输入是 **每帧四个波段的 DAT**（B08、B09、B10、B16，`.DAT`、`.DAT.bz2`、`.DAT.gz` 均可，
顺序任意），1 至 6 帧，另需气旋中心经纬度——DAT 本身不含气旋中心，V1 以此定位 500×500
原生窗口。时间默认取四个波段观测中点（MJD）的均值，也可显式给 `minutes`。

```cpp
#include "inference_best.hpp"

h9dp::InferenceBest model({"/path/to/fold0.v1m", "/path/to/fold1.v1m"}); // 多折等权平均
std::vector<h9dp::HimawariScan> scans = {
    {{"B08.DAT", "B09.DAT", "B10.DAT", "B16.DAT"}, std::nullopt},
    {{"B08b.DAT", "B09b.DAT", "B10b.DAT", "B16b.DAT"}, std::nullopt},
};
double dp = model.predict(scans, {25.3, 126.9}); // hPa
```

- `predict` 为 `const`，加载后无需 GPU，多个线程可并发调用；错误以 `std::exception` 抛出。
- 帧少于 6 时按可用帧计算，与 Python 缺失帧路径一致（此时要求各帧时间不同）。
- `fold{k}.v1m` 由 `scripts/export_v1_inference.py export` 生成，输入必须与 Python 同源
  （源元数据、特征、MOTIS 目录、种子 1 折号都会复核，否则拒绝导出）。
- `--features REQUEST.v1q` 模式用于只提供缓存描述子时的回归核对，不需要 DAT。

```bash
cmake -S inference -B /mnt/data/H9dP/runs/v1/rotation_seed1/cpp_inference/build \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_PREFIX_PATH="$CONDA_PREFIX"     # 提供 zlib；BZip2 可用时自动启用
cmake --build /mnt/data/H9dP/runs/v1/rotation_seed1/cpp_inference/build -j
inference_best_cli LAT LON MODEL.v1m [MODEL.v1m ...] -- DAT DAT DAT DAT [DAT ...]
```

本机 `/usr/include` 没有 `bzlib.h`，所以这次构建只启用 gzip；仓库内的 `.DAT.bz2`
会报 “This build has no BZip2 support”。需要读 `.bz2` 时用
`-DCMAKE_PREFIX_PATH="$CONDA_PREFIX"` 并确保环境内 `libbz2.so` 可被链接（本次未验证该链接路径）。

## 计时口径与含义

- C++ 时间：`InferenceBest::predict` 全程，含 24 个 DAT 的解压/定标/裁切、V1 描述子与逆风计算。
- Python 对照：单进程、仅重算同一批 DAT 到 V1 描述子，**不含**后续银行矩阵与 GP 拟合；
  这是对 C++ 更严格（更难达到 1.5×）的对照，因为完整 Python 服务还要多一步。
- 模型文件事前加载，不计入任一方的单次计时。磁盘页缓存未逐帧预热。
- 绝对耗时（约 125 ms/条）尚未优化；如需更低延迟，解码和极坐标采样是主要剩余成本。

## 代码与验证

- 库：`inference/inference_best.hpp`（公共接口）、`inference_best.cpp`（后验回归）、
  `hsd.cpp`（DAT 解码/几何/定标）、`v1_features.cpp`（V1 描述子）、`main.cpp`（CLI）、
  `CMakeLists.txt`（C++20）。
- Python 侧：`scripts/export_v1_inference.py`（导出/打包）、`scripts/check_v1_inference.py`
  （验收）；`src/h9dp/hsd.py` 增加 `decode_counts`，`_normalize` 拆出
  `_normalization_state`/`_apply_normalization`，`scripts/v1.py` 增加 `device` 参数（默认仍为 CUDA）。
- 测试：`tests/test_v1_inference.py`（合成 DAT 端到端、截断模型拒绝、模型/描述子对照）、
  `tests/test_hsd.py` 参数化；全套 `246 passed`。

验收命令（模型与验证数据均在外部目录）：

```bash
python -m scripts.check_v1_inference \
  --exe /mnt/data/H9dP/runs/v1/rotation_seed1/cpp_inference/build/inference_best_cli \
  --models /mnt/data/H9dP/runs/v1/rotation_seed1/cpp_inference \
  --features /mnt/data/H9dP/runs/v1/rotation_seed1/features.npz \
  --formal /mnt/data/H9dP/runs/v1/rotation_seed1/formal \
  --dataset HIMAWARI_data \
  --output /mnt/data/H9dP/runs/v1/rotation_seed1/cpp_inference/acceptance_final2.json
```

## 限制

- 329 条 DAT 端到端检查只覆盖本地已有的 2020–2025 年记录，其中 293 条为 R3、36 条含
  FLDK 帧；2016–2019 全盘记录未纳入本次 DAT 核对。
- 端口一致性说明的是“实现与 V1 一致”，不代表新的泛化证据；V1 的现场独立观测检验仍待进行。
- Python `scripts/v1.py` 的完整流程（含 GPU 银行/GP）未被替换，C++ 版只承担已训练模型的推理。

[V1 报告](../V1/README.md) · [评估口径](../EVALUATION.md) · [文档索引](../README.md)
