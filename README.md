# ESP32-S3-EYE 手语识别边缘协作系统

## 作品简介

本作品使用 ESP32-S3-EYE 采集摄像头画面，通过 Wi-Fi 将 JPEG 视频帧发送到 Windows 边缘端；Windows 端运行手语模型，识别结果通过短文本回传到开发板 LCD。摄像头画面只在网页显示，LCD 用于显示识别文字和交互状态。

## 目录

- `board_source/`：ESP32-S3-EYE 摄像头、LCD、文字 Agent 相关源码和测试脚本。
- `edge_server/`：Windows/Python 边缘服务、模型推理、网页预览和板端文字回传。
- `edge_server/models/`：25 类主模型以及“有/要”“好/谢谢”二分类模型。
- `logs/`：AI Coding 日志。

## Windows 端运行

```powershell
cd edge_server
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python run_server.py --host 0.0.0.0 --port 8000 --gui `
  --model models/sign_rich_frames_25_final.pth `
  --board-text-url http://<ESP32-IP>:81/text
```

浏览器打开 `http://127.0.0.1:8000/` 查看摄像头画面和识别状态。没有桌面显示环境时去掉 `--gui`。

## 测试

```powershell
cd edge_server
python -m unittest discover -s tests -v
```

## 板端说明

板端使用 OpenVela/NuttX 的 ESP32-S3-EYE 摄像头和 LCD 适配。`board_source/` 中的源码对应当前已验证版本；完整 OpenVela 工程不放入本仓库，按参赛 manifest 在仓库外同步。

## 注意事项

不要把 Wi-Fi 密码、SSH 密码、个人绝对路径、`.venv/`、`__pycache__/`、录制视频和临时备份提交到仓库。
