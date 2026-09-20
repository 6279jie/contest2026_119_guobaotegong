# ESP32 edge server

## Install and start on the cloud notebook

```powershell
cd <extracted-folder>
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python run_server.py --host 0.0.0.0 --port 8000 --gui `
  --model models/sign_rich_frames_25_final.pth `
  --board-text-url http://<ESP32-IP>:81/text
```

If the notebook has no desktop/OpenCV display, omit `--gui`.

The three `.pth` files in `models/` are the trained 25-class model and the
two binary disambiguation models. The server loads the pair models from the
same directory as the selected main model.

Keep this PowerShell window open. Stop with `Ctrl+C`.

## Firewall (Administrator PowerShell, once)

```powershell
Set-NetFirewallRule -DisplayName "Allow camera 8000" -Profile Any
```

If the rule does not exist:

```powershell
New-NetFirewallRule -DisplayName "Allow camera 8000" -Direction Inbound -Protocol TCP -LocalPort 8000 -Action Allow -Profile Any
```

## Check

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
Invoke-RestMethod http://192.168.1.109:8000/health
```

The response must contain `status: ok`. After replacing the old source, restart the Python service so the stream parser fix is loaded.
