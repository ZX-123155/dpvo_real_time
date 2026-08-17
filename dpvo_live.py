# -*- coding: utf-8 -*-
"""DPVO 实时 SLAM：手机/摄像头视频流 → 实时定位 + 3D 重建显示。

双窗口：
  窗口 1 "Live Camera"      : 手机实时视频（原始画面）
  窗口 2 "3D Map & Trajectory" : 实时 3D 点云重建（按深度着色）+ 相机轨迹

视频源模式（--source）：
  tcp://127.0.0.1:8765                     默认：连 Windows 侧 cam_server.py
  http://<手机IP>:8080/video               手机 App 推流（IP Webcam / DroidCam 等，
                                           WSL 镜像网络直连，无需 Windows 中转）
  video://<路径>.mp4                       本地视频文件（离线测试）
  imagefolder://<目录>                     图像序列（循环播放，离线测试）

用法（WSL 内 dpvo 环境）：
  python dpvo_live.py                                        # TCP 模式
  python dpvo_live.py --source http://192.168.1.5:8080/video # 手机直连
  python dpvo_live.py --source video://a.mp4 --calib=calib/phone.txt  # 视频文件

按 q 或 Ctrl+C 退出，退出时保存轨迹图到 trajectory_plots/live.pdf
"""
import argparse
import socket
import struct
import threading
import time

import numpy as np
import cv2
import torch

from dpvo.config import cfg
from dpvo.dpvo import DPVO

TARGET_LONG = 640             # 处理长边（保持宽高比缩放，DPVO 对低分辨率/变形敏感）
VIEW_W, VIEW_H = 640, 480      # 3D 视图画布尺寸
VF = 500.0                     # 3D 视图虚拟焦距（控制视角缩放）
VRAM_LIMIT = 4.2e9             # 显存保护阈值（4.2GB，6GB 卡留余量）
MAX_POINTS = 5000              # 3D 视图单帧最多绘制的点数（控制开销）


# ---------------------------------------------------------------- 视频源
class TCPSource:
    """连 Windows 侧 cam_server.py（旧模式，备选）"""

    def __init__(self, host, port):
        self.conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.conn.connect((host, port))
        print("[OK] 已连接摄像头服务器 {}:{}".format(host, port))

    def get_frame(self):
        hdr = b""
        while len(hdr) < 4:
            chunk = self.conn.recv(4 - len(hdr))
            if not chunk:
                return None
            hdr += chunk
        n = struct.unpack(">I", hdr)[0]
        data = b""
        while len(data) < n:
            chunk = self.conn.recv(n - len(data))
            if not chunk:
                return None
            data += chunk
        return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)

    def close(self):
        self.conn.close()


class HTTPSource(threading.Thread):
    """手机 http/mjpeg 流：后台线程持续收帧，只保留最新一帧。

    好处：DPVO 处理再慢，显示的永远是最近帧，不会延迟累积。
    """

    def __init__(self, url):
        super().__init__(daemon=True)
        self.url = url
        self.cap = cv2.VideoCapture(url)
        if not self.cap.isOpened():
            raise RuntimeError("无法打开视频流: {}".format(url))
        self.lock = threading.Lock()
        self.latest = None
        self.running = True
        self.frames = 0
        self.start()

    def run(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                print("[WARN] 视频流读取失败/中断")
                time.sleep(0.5)
                continue
            with self.lock:
                self.latest = frame
            self.frames += 1

    def get_frame(self):
        with self.lock:
            f = self.latest
            self.latest = None
        return f

    def close(self):
        self.running = False
        self.cap.release()


class VideoFileSource:
    """本地视频文件（离线测试用）"""

    def __init__(self, path):
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise RuntimeError("无法打开视频: {}".format(path))
        print("[OK] 视频文件 {} 已打开".format(path))

    def get_frame(self):
        ret, frame = self.cap.read()
        return frame if ret else None

    def close(self):
        self.cap.release()


class FolderSource:
    """图像序列目录（离线测试用，循环播放模拟实时流）"""

    def __init__(self, path):
        import glob as _glob
        self.imgs = sorted(_glob.glob(path + "/*.png") + _glob.glob(path + "/*.jpg"))
        if not self.imgs:
            raise RuntimeError("目录中没有图像: {}".format(path))
        self.i = 0
        print("[OK] 图像序列共 {} 帧（循环播放）".format(len(self.imgs)))

    def get_frame(self):
        if self.i >= len(self.imgs):
            self.i = 0
        f = cv2.imread(self.imgs[self.i])
        self.i += 1
        return f

    def close(self):
        pass


# ---------------------------------------------------------------- 几何工具
def quat_to_rot(wxyz):
    """wxyz 四元数 → 旋转矩阵 R_wc（world→camera，行向量右乘约定）"""
    w, x, y, z = wxyz
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def project_to_view(points_w, pose):
    """世界坐标点 → 3D 视图画布像素坐标（以当前相机位姿为观察视角）"""
    t_wc = pose[:3]
    R_wc = quat_to_rot(pose[[6, 3, 4, 5]])          # wxyz
    p_c = (points_w - t_wc) @ R_wc                  # 行向量右乘 = R_wc.T @ (p - t)
    z = p_c[:, 2]
    valid = (z > 0.05) & (z < 30.0) & np.isfinite(z)
    # 先过滤再投影，避免 0/0 与 inf 参与除法产生 NaN 警告
    p_c, z = p_c[valid], z[valid]
    u = (VF * p_c[:, 0] / z + VIEW_W / 2).astype(np.int32)
    v = (VF * p_c[:, 1] / z + VIEW_H / 2).astype(np.int32)
    m = (u >= 0) & (u < VIEW_W) & (v >= 0) & (v < VIEW_H)
    return u[m], v[m], z[m]


def draw_3d_view(points, colors, pose, traj_w):
    """3D 视图：点云按深度着色（近=红 远=蓝），叠加相机轨迹（黄线）"""
    canvas = np.full((VIEW_H, VIEW_W, 3), 18, np.uint8)
    if points is None or len(points) == 0:
        return canvas

    # DPVO 的 points_ 是预分配数组（N*M 行），只更新了前 m*4 行有效数据，
    # 其余是零填充 —— 过滤掉全零行（真实点恰好为 (0,0,0) 的概率可忽略，
    # 且原点附近的点投影后也会被 z 过滤剔除）
    mask = (points != 0).any(axis=1)
    points = points[mask]
    if len(points) == 0:
        return canvas

    # 点云随机采样，控制绘制开销
    if len(points) > MAX_POINTS:
        idx = np.random.choice(len(points), MAX_POINTS, replace=False)
        points, colors = points[idx], colors[idx]

    u, v, z = project_to_view(points, pose)
    if len(u) > 0:
        # 深度 → jet 伪彩色（近红远蓝），叠加白点底避免太暗
        d = np.clip((z - 0.05) / (15.0 - 0.05), 0.0, 1.0)
        bgr = cv2.applyColorMap((d * 255).astype(np.uint8), cv2.COLORMAP_JET)
        canvas[v, u] = bgr.reshape(-1, 3)

    # 相机轨迹（世界系 → 当前视角投影）
    if len(traj_w) >= 2:
        tr = np.array(traj_w)
        u2, v2, _ = project_to_view(tr, pose)
        if len(u2) >= 2:
            pts = np.stack([u2, v2], axis=1).astype(np.int32)
            cv2.polylines(canvas, [pts], False, (0, 255, 255), 2)
            cv2.circle(canvas, tuple(pts[-1]), 5, (0, 0, 255), -1)   # 当前位置

    cv2.putText(canvas, "3D points (jet: near=red far=blue)  [yellow: trajectory]",
                (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    return canvas


# ---------------------------------------------------------------- 主流程
def parse_args():
    p = argparse.ArgumentParser(description="DPVO 实时 SLAM：手机/摄像头流 → 实时定位 + 3D 重建")
    p.add_argument("--source", default="tcp://127.0.0.1:8765",
                   help="视频源：tcp://host:port、http://手机IP:端口/video、video://路径 或 imagefolder://目录")
    p.add_argument("--calib", default=None,
                   help="内参文件（fx fy cx cy，对应原始分辨率）。默认按 640x480 横屏估 fx=fy=600 cx=320 cy=240")
    p.add_argument("--no-gui", action="store_true",
                   help="不弹窗口（远程/无显示环境下用），只打印位姿，退出时仍保存轨迹")
    p.add_argument("--screenshot", type=int, default=0, metavar="N",
                   help="调试用：在第 N 帧保存双窗口截图到当前目录（live_camera.png / live_3d.png）")
    return p.parse_args()


def load_calib(path):
    """读内参文件（fx fy cx cy），返回对应原始分辨率的 K"""
    c = np.loadtxt(path, delimiter=" ")[:4]
    return float(c[0]), float(c[1]), float(c[2]), float(c[3])


def main():
    args = parse_args()
    cfg.merge_from_file("config/default.yaml")
    no_gui = args.no_gui
    if no_gui:
        print("[INFO] --no-gui 模式：不显示窗口，仅打印位姿")

    # 原始内参（对应输入流原始分辨率）；可通过 --calib 覆盖
    if args.calib:
        FX0, FY0, CX0, CY0 = load_calib(args.calib)
    else:
        FX0, FY0, CX0, CY0 = 600.0, 600.0, 320.0, 240.0   # 640x480 横屏近似
    print("[INFO] 原始内参: fx={:.0f} fy={:.0f} cx={:.0f} cy={:.0f}".format(FX0, FY0, CX0, CY0))

    if args.source.startswith("tcp://"):
        host, port = args.source[6:].rsplit(":", 1)
        src = TCPSource(host, int(port))
    elif args.source.startswith("http://"):
        print("[INFO] 连接手机视频流 {} ...".format(args.source))
        src = HTTPSource(args.source)
        print("[OK] 已连接，开始实时 SLAM（按 q 退出）")
    elif args.source.startswith("video://"):
        src = VideoFileSource(args.source[len("video://"):])
    elif args.source.startswith("imagefolder://"):
        src = FolderSource(args.source[len("imagefolder://"):])
    else:
        raise SystemExit("不支持的 --source: {}（用 tcp://、http://、video:// 或 imagefolder://）".format(args.source))

    slam = None
    traj = []          # 世界系轨迹（保留全程，用于 3D 视图）
    t = 0

    try:
        # 关键：推理必须包在 no_grad 里！否则 autograd 计算图每帧累积，
        # 显存以 ~0.5GB/帧 暴涨，几十帧就 OOM（demo.py 正是这么做的）
        with torch.no_grad():
            while True:
                frame = src.get_frame()
                if frame is None:
                    time.sleep(0.005)
                    continue

                # 保持宽高比缩放到长边 TARGET_LONG，16 对齐。
                # 不能强制压成固定尺寸：DPVO 对变形/过低分辨率非常敏感，
                # 384x288 下位姿完全出不来（实测全零）
                h, w = frame.shape[:2]
                scale = TARGET_LONG / max(h, w)
                nw, nh = int(w * scale) // 16 * 16, int(h * scale) // 16 * 16
                frame = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
                H, W = frame.shape[:2]
                # 内参按缩放比例同步：fx_new = fx_orig * W / w（缩放后/原始）
                fx, fy, cx, cy = FX0 * W / w, FY0 * H / h, CX0 * W / w, CY0 * H / h

                image = torch.from_numpy(frame).permute(2, 0, 1).cuda()
                intrinsics = torch.as_tensor([fx, fy, cx, cy]).cuda()

                if slam is None:
                    slam = DPVO(cfg, "dpvo.pth", ht=H, wd=W, viz=False)

                slam(t, image, intrinsics)

                # 注意：poses_ 是预分配缓冲区，poses_[-1] 是"最后一个槽位"（未使用，恒为 0）！
                # 当前最新帧是 poses_[slam.n - 1]（上一帧是 slam.n - 2）
                pose = slam.pg.poses_[slam.n - 1].detach().cpu().numpy()
                traj.append(pose[:3])

                # ---- 显存保护（兜底保险，no_grad 后一般不会触发） ----
                if t > 30 and torch.cuda.memory_allocated() > VRAM_LIMIT:
                    print("[WARN] 显存 {:.1f}GB 接近上限，自动重置 SLAM 状态（轨迹保留）"
                          .format(torch.cuda.memory_allocated() / 1e9))
                    del slam
                    torch.cuda.empty_cache()
                    slam = None
                    t = 0

                if t % 5 == 0:
                    print("[t={:4d}] pos=({:6.3f}, {:6.3f}, {:6.3f})".format(
                        t, pose[0], pose[1], pose[2]))

                # ---- 双窗口显示（--no-gui 或远程环境下自动降级为不显示） ----
                if not no_gui:
                    # 每 5 帧取一次点云（GPU→CPU 拷贝有开销，控制频率）
                    if t % 5 == 0 and slam is not None:
                        pts = slam.pg.points_.detach().cpu().numpy()
                        cols = slam.pg.colors_.detach().cpu().numpy()
                    else:
                        pts = cols = None
                    view3d = draw_3d_view(pts, cols, pose, traj)
                    try:
                        cv2.imshow("Live Camera", frame)
                        cv2.imshow("3D Map & Trajectory", view3d)
                    except cv2.error:
                        print("[WARN] 窗口显示失败（远程/无显示环境？），自动切换到 --no-gui 模式")
                        no_gui = True

                    # 调试截图
                    if args.screenshot > 0 and t == args.screenshot:
                        cv2.imwrite("live_camera.png", frame)
                        cv2.imwrite("live_3d.png", view3d)
                        print("[INFO] 截图已保存: live_camera.png / live_3d.png")

                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        print("[INFO] 用户按 q 退出")
                        break
                t += 1
    except KeyboardInterrupt:
        print("[INFO] 中断")
    finally:
        src.close()
        cv2.destroyAllWindows()
        if slam is not None and t > 0:
            poses, tstamps = slam.terminate()
            from dpvo.plot_utils import plot_trajectory
            from evo.core.trajectory import PoseTrajectory3D
            trajectory = PoseTrajectory3D(
                positions_xyz=poses[:, :3],
                orientations_quat_wxyz=poses[:, [6, 3, 4, 5]],
                timestamps=tstamps)
            plot_trajectory(trajectory, title="DPVO live trajectory",
                            filename="trajectory_plots/live.pdf")
            print("[OK] 处理 {} 帧完成，轨迹已保存到 trajectory_plots/live.pdf".format(t))


if __name__ == "__main__":
    main()
