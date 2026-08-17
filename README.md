# DPVO 实时单目 SLAM：手机/摄像头实时定位与重建

基于 **DPVO**（Deep Patch Visual Odometry，Princeton VL，NeurIPS 2023）的实时单目 SLAM 项目。**手机（或笔记本摄像头）实时推流 → 电脑实时估计相机位姿（定位）+ 构建稀疏 3D 点云（重建）+ 双窗口实时显示**。

> DPVO 论文：*Deep Patch Visual Odometry*, Teed, Lipson, Deng, NeurIPS 2023
> 官方仓库：https://github.com/princeton-vl/DPVO

## 效果

运行后弹出**两个窗口**：

| 窗口 | 内容 |
|---|---|
| **Live Camera** | 手机/摄像头实时视频画面 |
| **3D Map & Trajectory** | 实时 3D 点云重建（按深度着色：近红远蓝）+ 相机轨迹（黄色线 + 当前位置红点） |

终端实时打印相机位置 `(x, y, z)`；退出时自动保存轨迹图 `trajectory_plots/live.pdf`。

**算力要求**：RTX 3060 6GB 实测流畅运行（显存占用稳定 ~0.2GB，CPU 解码手机流约 10% 单核）。手机端负责编码，电脑只解码 + SLAM，负载很低，**不会卡死**。唯一感知到的"延迟"是 WiFi 网络延迟（50-200ms），属正常现象。

## 目录结构

```
dpvo_real_time/
├── README.md               # 本说明
├── environment.yml         # conda 环境定义（python3.10 + torch2.3.1 + cuda12.1）
├── dpvo.pth                # DPVO 预训练权重（14MB）
├── cam_server.py           # 【Windows 侧】笔记本摄像头推流脚本（TCP 模式用）
├── dpvo_live.py            # 【WSL 侧】实时 SLAM 脚本（定位 + 双窗口显示）
├── dpvo/                   # DPVO 核心 Python 包
├── config/                 # DPVO 配置（default.yaml）
├── calib/                  # 相机内参文件（fx fy cx cy）
├── thirdparty/             # eigen（编译 CUDA 扩展时需要）
├── DPViewer/               # Pangolin 3D 可视化（可选，见"3D 可视化"节）
└── demo.py                 # 离线 demo（跑 TUM 数据集）
```

## 环境要求

| 组件 | 要求 | 说明 |
|---|---|---|
| Windows | 10/11 + WSL2 | 手机推流时无需 Windows 侧运行任何程序 |
| WSL2 发行版 | Ubuntu 22.04 | 跑 SLAM 主程序（GPU 加速） |
| GPU | NVIDIA，≥6GB 显存 | 实测 6GB 稳定运行（显存仅用 ~0.2GB） |
| WSL 内 Python | conda 环境 `dpvo` | 见下方搭建步骤 |
| 手机 | Android / iPhone + 推流 App | 见"手机推流"节 |

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

> 编译需要 `thirdparty/eigen-3.4.0`。若缺失（GitHub 仓库未包含），手动下载：
> ```bash
> cd dpvo_real_time && mkdir -p thirdparty && cd thirdparty
> wget https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip && unzip eigen-3.4.0.zip
> ```

### 4. 下载权重（若 dpvo.pth 缺失）

```bash
pip install gdown
gdown 1dRqftpImtHbbIPNBIseCv9EvrlHEnjhX -O models.zip && unzip models.zip   # 得到 dpvo.pth
```

## 运行一：手机推流（推荐，无需 Windows 侧程序）

**手机装推流 App**（都支持 HTTP MJPEG 流，免费）：

| 手机 | App | 操作 |
|---|---|---|
| Android | **IP Webcam**（Google Play 免费） | 打开后底部显示 `http://192.168.x.x:8080`，点 `Start server` |
| iPhone | **DroidCam** 或 **Camo**（免费版） | 打开后记下界面上的 `http://192.168.x.x:4747/video` |

**前提**：手机和电脑连**同一个 WiFi**。

**WSL 终端**（只需要这一个终端）：

```bash
cd /mnt/c/Users/luyicheng/Desktop/暑期研0培训/7.28-8.13/cv_learning/slam_learning/dpvo_real_time
conda activate dpvo
python -u dpvo_live.py --source http://<手机IP>:8080/video
# 例：python -u dpvo_live.py --source http://192.168.1.5:8080/video
```

看到 `[OK] 已连接，开始实时 SLAM` 后：
1. 弹出两个窗口：**Live Camera**（手机画面）+ **3D Map & Trajectory**（实时重建）
2. **拿着手机慢慢平移/转动**——终端实时打印位置，3D 点云和轨迹实时生长
3. 按 `q` 退出，轨迹自动保存到 `trajectory_plots/live.pdf`

> 无需手动改代码：WSL2 镜像网络下 WSL 可直接访问手机 IP（局域网直连，不走代理）。

## 运行二：笔记本摄像头（Windows + WSL 双终端）

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
python -u dpvo_live.py        # 默认 tcp://127.0.0.1:8765 连 Windows 推流
```

## 运行三：离线测试（图像序列循环，不需要手机/摄像头）

```bash
# 用 TUM fr1_desk 数据（329MB，一次性下载）
curl -sL -A "Mozilla/5.0" -o fr1_desk.tgz \
  "https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_desk.tgz"
tar -xzf fr1_desk.tgz

# 循环播放图像序列，模拟实时流（双窗口效果完全一致）
python -u dpvo_live.py --source imagefolder:///home/<user>/DPVO/datasets/rgbd_dataset_freiburg1_desk/rgb
```

## 关键注意

- **位姿一直 (0,0,0) 是正常的**：单目 SLAM 必须靠相机运动产生视差，摄像头静止时轨迹就是原点。**动起来才有轨迹**。
- 移动时要**平缓**，避免剧烈晃动导致初始化失败；初始化约需前几秒。
- 手机画面建议 640×480 或 720p（App 里可调），推太高分辨率浪费流量且无增益（处理端统一降为 384×288）。
- 内参用的是近似值（fx=fy=360 @ 384x288），轨迹**形状正确但尺度不精确**。追求精度可做相机标定后更新 `dpvo_live.py` 里的 `FX, FY, CX, CY`。

## 常见问题

| 问题 | 解决 |
|---|---|
| 编译报 `nv/target` / `thrust/complex.h` 找不到 | cccl 版本问题，按环境搭建第 2 步降级 |
| `pip install .` 报 `No module named 'torch'` | 加 `--no-build-isolation` |
| 摄像头打开失败（Code 45） | 打开笔记本摄像头开关（Fn 快捷键/物理挡板/BIOS） |
| 手机流连不上 | 确认同一 WiFi；WSL 里 `curl -s http://<手机IP>:8080/video | head -c 100` 测试；Windows 防火墙需允许入站 8080（IP Webcam 首次会提示） |
| 跑一会儿 CUDA OOM | **已根治**（推理包在 `torch.no_grad()` 内，显存稳定 ~0.2GB）。若仍出现，检查是否有其他 GPU 程序占用 |
| 位姿全是 0 | 摄像头静止，移动手机/摄像头 |
| 3D 可视化（DPViewer/viz）报错 | 见下方说明；内置 OpenCV 3D 视图不依赖它 |

## 内置 3D 视图 vs DPViewer

- **内置 3D 视图**（默认）：`dpvo_live.py` 自带的 OpenCV 3D 点云投影窗口，零额外依赖，WSLg 远程会话下也稳定。深度按颜色区分（近红远蓝），叠加黄色轨迹线。
- **DPViewer**（可选）：仓库含 Pangolin 3D 可视化模块（`DPViewer/`），交互更强（可旋转视角），但：
  - 需编译旧 ABI 版 Pangolin（DPVO 官方 fork，`-D_GLIBCXX_USE_CXX11_ABI=0`）到独立目录
  - **Pangolin 窗口在 WSLg 远程会话（Todesk 等）下无法工作**（连最小示例都段错误），本机直接使用才行
  - 运行时设置 `export LD_LIBRARY_PATH=<pangolin_install>/lib:$LD_LIBRARY_PATH`

## 参考

- DPVO 官方仓库：https://github.com/princeton-vl/DPVO
- DPVO 论文：https://arxiv.org/abs/2208.04726
- TUM RGB-D 数据集：https://cvg.cit.tum.de/data/datasets/rgbd-dataset
- IP Webcam（Android）：https://play.google.com/store/apps/details?id=com.pas.webcam
- DroidCam（iPhone/Android）：https://dev47apps.com/
