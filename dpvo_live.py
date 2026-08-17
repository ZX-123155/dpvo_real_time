# -*- coding: utf-8 -*-
"""WSL 侧 DPVO 实时 SLAM：从 Windows 摄像头服务器收帧，实时定位 + 显示轨迹。

用法（WSL 内 dpvo 环境）：
    python dpvo_live.py
按 q 或 Ctrl+C 退出，退出时保存轨迹图到 trajectory_plots/live.pdf
"""
import socket
import struct
import numpy as np
import cv2
import torch

from dpvo.config import cfg
from dpvo.dpvo import DPVO

HOST = "127.0.0.1"
PORT = 8765
FX, FY, CX, CY = 360.0, 360.0, 192.0, 144.0   # 笔记本摄像头近似内参（640x480）


def recv_frame(conn):
    """接收一帧 JPEG 并解码为 BGR 图像"""
    hdr = b""
    while len(hdr) < 4:
        chunk = conn.recv(4 - len(hdr))
        if not chunk:
            return None
        hdr += chunk
    n = struct.unpack(">I", hdr)[0]
    data = b""
    while len(data) < n:
        chunk = conn.recv(n - len(data))
        if not chunk:
            return None
        data += chunk
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)


def draw_trajectory(traj, size=(288, 384)):
    """把 2D 轨迹（x-y 俯视投影）画到画布上"""
    canvas = np.full((size[0], size[1], 3), 30, np.uint8)
    if len(traj) < 2:
        return canvas
    tr = np.array(traj)[:, :2]
    mn = tr.min(axis=0)
    mx = tr.max(axis=0)
    span = (mx - mn).max() or 1.0
    pad = 20
    pts = ((tr - mn) / span * (min(size) - 2 * pad) + pad).astype(np.int32)
    pts[:, 1] = size[0] - pts[:, 1]
    for i in range(1, len(pts)):
        cv2.line(canvas, tuple(pts[i - 1]), tuple(pts[i]), (0, 165, 255), 2)
    if len(pts):
        cv2.circle(canvas, tuple(pts[-1]), 4, (0, 0, 255), -1)
    return canvas


def main():
    cfg.merge_from_file("config/default.yaml")

    print("[INFO] 连接摄像头服务器 {}:{} ...".format(HOST, PORT))
    conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    conn.connect((HOST, PORT))
    print("[OK] 已连接，开始实时 SLAM（按 q 退出）")

    slam = None
    traj = []
    t = 0

    try:
        while True:
            frame = recv_frame(conn)
            if frame is None:
                print("[INFO] 视频流结束")
                break

            frame = cv2.resize(frame, (384, 288))
            H, W = frame.shape[:2]
            image = torch.from_numpy(frame).permute(2, 0, 1).cuda()
            intrinsics = torch.as_tensor([FX, FY, CX, CY]).cuda()

            if slam is None:
                slam = DPVO(cfg, "dpvo.pth", ht=H, wd=W, viz=False)

            slam(t, image, intrinsics)

            pose = slam.pg.poses_[-1].detach().cpu().numpy()
            traj.append(pose[:3])

            if t % 5 == 0:
                print("[t={:4d}] pos=({:6.3f}, {:6.3f}, {:6.3f})".format(
                    t, pose[0], pose[1], pose[2]))

            disp = np.hstack([frame, draw_trajectory(traj)])
            cv2.imshow("DPVO live (left: camera, right: trajectory)", disp)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("[INFO] 用户按 q 退出")
                break

            t += 1
    except KeyboardInterrupt:
        print("[INFO] 中断")
    finally:
        conn.close()
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
