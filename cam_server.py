# -*- coding: utf-8 -*-
"""Windows 侧摄像头推流服务器：读取笔记本内置摄像头，JPEG 编码后通过 TCP 发给 WSL 端的 DPVO。

用法（Windows，Python 3.10 + opencv-python）：
    python cam_server.py

监听 127.0.0.1:8765，等 WSL 端 DPVO 连上来后开始推流。
"""
import socket
import struct
import cv2

HOST = "127.0.0.1"
PORT = 8765
WIDTH, HEIGHT = 640, 480


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] 无法打开摄像头（可能被 Fn 开关 / 物理挡板 / BIOS 关闭，或隐私设置禁用）")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 30)

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((HOST, PORT))
    srv.listen(1)
    print("[OK] 摄像头已打开 ({}x{})，等待 WSL 端连接 {}:{} ...".format(WIDTH, HEIGHT, HOST, PORT))
    conn, addr = srv.accept()
    print("[OK] 客户端已连接: {}".format(addr))

    n = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            if not ok:
                continue
            data = buf.tobytes()
            conn.sendall(struct.pack(">I", len(data)) + data)
            n += 1
            if n % 30 == 0:
                print("[INFO] 已推流 {} 帧".format(n))
    except (BrokenPipeError, ConnectionResetError):
        print("[INFO] 客户端断开，退出")
    finally:
        cap.release()
        conn.close()


if __name__ == "__main__":
    main()
