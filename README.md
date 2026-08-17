# DPVO 实时单目 SLAM：摄像头实时定位与重建

基于 **DPVO**（Deep Patch Visual Odometry，Princeton VL，NeurIPS 2023）的实时单目 SLAM 项目。读取笔记本/USB 摄像头视频流，**实时估计相机位姿（定位）+ 构建稀疏 3D 点云（重建）+ 实时绘制相机轨迹**。

> DPVO 论文：*Deep Patch Visual Odometry*, Teed, Lipson, Deng, NeurIPS 2023
> 官方仓库：https://github.com/princeton-vl/DPVO

## 效果

- 移动摄像头（或移动电脑），终端实时打印相机位置 `(x, y, z)`
- OpenCV 窗口左侧显示摄像头画面，右侧实时绘制相机运动轨迹（2D 俯视投影）
- 退出时自动保存轨迹图 `trajectory_plots/live.pdf` 和轨迹文件

## 目录结构

```
dpvo_real_time/
├── README.md               # 本说明
├── environment.yml         # conda 环境定义（python3.10 + torch2.3.1 + cuda12.1）
├── dpvo.pth                # DPVO 预训练权重（14MB）
├── cam_server.py           # 【Windows 侧】摄像头推流脚本
├── dpvo_live.py            # 【WSL 侧】实时 SLAM 脚本（定位 + 轨迹显示）
├── dpvo/                   # DPVO 核心 Python 包（已编译好 CUDA 扩展则直接可用）
├── config/                 # DPVO 配置（default.yaml）
├── calib/                  # 相机内参文件（fx fy cx cy）
├── thirdparty/             # eigen（编译 CUDA 扩展时需要）
├── DPViewer/               # 3D 可视化模块（可选，见"3D 可视化"节）
└── demo.py                 # 离线 demo（跑 TUM 数据集）
```

## 环境要求

| 组件 | 要求 | 说明 |
|---|---|---|
| Windows | 10/11 + WSL2 | 摄像头在 Windows 侧读取 |
| WSL2 发行版 | Ubuntu 22.04 | 跑 SLAM 主程序（GPU 加速） |
| GPU | NVIDIA，≥6GB 显存 | 需要 CUDA 驱动 |
| WSL 内 Python | conda 环境 `dpvo` | 见下方搭建步骤 |
| Windows 侧 Python | 3.10 + opencv-python | 只需读摄像头 + 推流 |

## 环境搭建（一次性）

### 1. WSL 内创建 conda 环境（约 2-3GB 下载）

```bash
# 进入项目目录（WSL 里 Windows 桌面路径）
cd /mnt/c/Users/luyicheng/Desktop/暑期研0培训/7.28-8.13/cv_learning/slam_learning/dpvo_real_time

# 创建环境（python3.10 + torch2.3.1 + cuda12.1 + 全部依赖）
conda env create -f environment.yml
conda activate dpvo
```

### 2. 安装编译工具（编译 CUDA 扩展需要 nvcc）

```bash
conda install -c nvidia cuda-nvcc=12.1 cuda-cudart-dev=12.1
```

> **⚠️ 重要坑**：装完 nvcc 后，`cuda-cccl` 可能被解析到最新版（13.x），会导致编译报错 `nv/target: No such file`。必须固定到 12.1：
> ```bash
> conda install -c nvidia "cuda-cccl=12.1.109"
> conda remove -n dpvo cuda-cccl_linux-64 -y
> ```

### 3. 编译安装 DPVO 包（3 个 CUDA 扩展）

```bash
export CUDA_HOME=$CONDA_PREFIX
export TORCH_CUDA_ARCH_LIST="8.6"   # 3060 = sm_86；40 系改 8.9，50 系改 12.0
pip install --no-build-isolation .  # 必须加 --no-build-isolation（torch 是 conda 装的）
```

> 编译需要 `thirdparty/eigen-3.4.0`（本仓库已含）。若缺失：`wget https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip && unzip -d thirdparty`。

### 4. 下载权重（若 dpvo.pth 缺失）

```bash
pip install gdown
gdown 1dRqftpImtHbbIPNBIseCv9EvrlHEnjhX -O models.zip && unzip models.zip   # 得到 dpvo.pth
```

### 5. Windows 侧 Python（只需 opencv）

```powershell
# 已有 Python 3.10 的话：
pip install opencv-python
```

## 运行：摄像头实时定位（核心功能）

需要**两个终端**同时运行（Windows 一个，WSL 一个）。

### 终端 1（Windows PowerShell）—— 摄像头推流

```powershell
python "C:\Users\luyicheng\Desktop\暑期研0培训\7.28-8.13\cv_learning\slam_learning\dpvo_real_time\cam_server.py"
```

看到 `[OK] 摄像头已打开 (640x480)，等待 WSL 端连接 127.0.0.1:8765 ...` 即就绪。

> 如果报"无法打开摄像头"：检查笔记本摄像头的 **Fn 快捷键/物理开关**（联想笔记本常见 Fn+F8/F10，被关闭时设备管理器里摄像头是 Code 45）。

### 终端 2（WSL）—— 实时 SLAM

```bash
cd /mnt/c/Users/luyicheng/Desktop/暑期研0培训/7.28-8.13/cv_learning/slam_learning/dpvo_real_time
conda activate dpvo
python -u dpvo_live.py
```

看到 `[OK] 已连接，开始实时 SLAM` 后：

1. 弹出一个窗口：**左 = 摄像头画面，右 = 相机轨迹**
2. **移动摄像头 / 拿笔记本转一圈**——终端实时打印位置，右侧轨迹实时生长
3. 按 `q` 退出，轨迹自动保存到 `trajectory_plots/live.pdf`

### 关键注意

- **位姿一直 (0,0,0) 是正常的**：单目 SLAM 必须靠相机运动产生视差，摄像头静止时轨迹就是原点。**动起来才有轨迹**。
- 移动时要**平缓**，避免剧烈晃动导致初始化失败；初始化约需前几秒。
- 摄像头内参用的是近似值（fx=fy=360 @ 384x288），轨迹**形状正确但尺度不精确**。追求精度可做相机标定后更新 `dpvo_live.py` 里的 `FX, FY, CX, CY`。

## 运行：离线 demo（TUM 数据集，不需要摄像头）

```bash
# 下载 TUM fr1_desk（329MB）
curl -sL -A "Mozilla/5.0" -o fr1_desk.tgz \
  "https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_desk.tgz"
tar -xzf fr1_desk.tgz

# 跑离线 demo（输出轨迹图 + 点云 + 轨迹文件）
python demo.py --imagedir=rgbd_dataset_freiburg1_desk/rgb \
  --calib=calib/fr1.txt --stride=2 --plot --save_ply --save_trajectory --name=fr1_desk
```

输出：`trajectory_plots/fr1_desk.pdf`（轨迹图）、`fr1_desk.ply`（点云）、`saved_trajectories/fr1_desk.txt`（TUM 格式轨迹）。

## 常见问题

| 问题 | 解决 |
|---|---|
| 编译报 `nv/target` / `thrust/complex.h` 找不到 | cccl 版本问题，按环境搭建第 2 步降级 |
| `pip install .` 报 `No module named 'torch'` | 加 `--no-build-isolation` |
| 摄像头打开失败（Code 45） | 打开笔记本摄像头开关（Fn 快捷键/物理挡板/BIOS） |
| 跑一会儿 CUDA OOM | 6GB 显存限制，已默认降到 384×288 处理分辨率；显存更大的机器可调高 |
| 位姿全是 0 | 摄像头静止，移动摄像头 |
| 3D 可视化（viz）报错/崩溃 | 见下方说明 |

## 3D 可视化（可选，默认关闭）

本项目默认用 OpenCV 窗口显示 **2D 轨迹**（稳定可靠）。仓库含 `DPViewer/`（Pangolin 3D 可视化模块），如需 3D 点云 + 轨迹窗口，把 `dpvo_live.py` 里的 `viz=False` 改为 `viz=True`，并满足：

- 在**本机直接使用**（不要通过远程桌面/Todesk，WSLg 图形转发会异常）
- 先编译旧 ABI 版 Pangolin（`~/DPVO/Pangolin`，DPVO 官方 fork，`-D_GLIBCXX_USE_CXX11_ABI=0`）
- 运行时设置 `export LD_LIBRARY_PATH=<pangolin_install>/lib:$LD_LIBRARY_PATH`

> 注意：Pangolin 窗口在 WSLg 远程会话下无法工作（连最小示例都会段错误），这是环境限制而非代码问题。

## 参考

- DPVO 官方仓库：https://github.com/princeton-vl/DPVO
- DPVO 论文：https://arxiv.org/abs/2208.04726
- TUM RGB-D 数据集：https://cvg.cit.tum.de/data/datasets/rgbd-dataset
