# RLPlayground

Windows quick-start guide for running the 1v1 Rocket-League-style RL trainer and 3D viewer.

## Project Layout

- backend: Python training loop, physics, websocket broadcast
- frontend: React + React Three Fiber live viewer

## One-Time Setup

### 1) Backend (Python)

Open PowerShell in repository root:

```powershell
cd C:\Users\Natha\Documents\GitHub\RLPlayground\backend
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 2) Frontend (Node)

Open another PowerShell:

```powershell
cd C:\Users\Natha\Documents\GitHub\RLPlayground\frontend
npm install
```

## Run (Normal)

### Terminal A: Start backend

```powershell
cd C:\Users\Natha\Documents\GitHub\RLPlayground\backend
.\.venv\Scripts\python.exe train_and_broadcast.py
```

Expected log line:

```text
[ws] listening on ws://localhost:8001
```

### Terminal B: Start frontend

```powershell
cd C:\Users\Natha\Documents\GitHub\RLPlayground\frontend
npm run dev
```

Open the Vite URL shown in terminal (usually `http://localhost:5173` or `http://localhost:5174`).

## Run (High-Throughput Local Training)

If you want maximum local training speed, disable the websocket simulation/viewer and run train-only mode:

```powershell
cd C:\Users\Natha\Documents\GitHub\RLPlayground\backend
$env:RL_TRAIN_ONLY="1"
$env:RL_CHUNK_TIMESTEPS="65536"
$env:RL_SNAPSHOT_EVERY="10"
$env:RL_EVAL_EVERY="20"
\.\.venv\Scripts\python.exe train_and_broadcast.py
```

Optional runtime knobs:

- `RL_DEVICE`: Torch device, default `auto`
- `RL_CHUNK_TIMESTEPS`: PPO learn chunk per iteration
- `RL_SNAPSHOT_EVERY`: Opponent pool snapshot cadence
- `RL_EVAL_EVERY`: Elo evaluation cadence (set `0` to disable)
- `RL_EVAL_MATCHES`: Matches per sampled eval opponent
- `RL_EVAL_OPPONENTS`: Number of sampled opponents per eval pass

## Stop Servers

### Clean stop (recommended)

- Click terminal running backend and press `Ctrl + C`
- Click terminal running frontend and press `Ctrl + C`

## If a Port is Stuck

### Check which process is using backend port 8001

```powershell
netstat -ano | findstr :8001
```

### Kill that process (replace PID)

```powershell
taskkill /PID <PID> /F
```

### Check frontend port 5173

```powershell
netstat -ano | findstr :5173
```

### Kill frontend process by PID

```powershell
taskkill /PID <PID> /F
```

## Fast Restart

From two terminals:

```powershell
# terminal A
cd C:\Users\Natha\Documents\GitHub\RLPlayground\backend
.\.venv\Scripts\python.exe train_and_broadcast.py
```

```powershell
# terminal B
cd C:\Users\Natha\Documents\GitHub\RLPlayground\frontend
npm run dev
```

## Troubleshooting

### Backend says module missing

Run:

```powershell
cd C:\Users\Natha\Documents\GitHub\RLPlayground\backend
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### Frontend is running but not connected

- Confirm backend log shows websocket listening on `localhost:8001`
- Confirm frontend uses default websocket URL in `frontend/src/App.jsx`
- Refresh browser once backend is up

