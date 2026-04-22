# N-Body HPC Dashboard — Complete Setup Guide
## Ubuntu 20.04 / 22.04 / 24.04

---

## ONE-COMMAND SETUP (run this first)

```bash
sudo apt-get update && sudo apt-get install -y \
    openmpi-bin libopenmpi-dev \
    build-essential \
    nvidia-cuda-toolkit \
    python3-pip && \
pip3 install flask flask-cors --break-system-packages
```

---

## STEP-BY-STEP

### 1. Extract project
```bash
unzip nbody_hpc_project.zip
cd nbody
```

### 2. Build the simulation binary
```bash
chmod +x scripts/build.sh
bash scripts/build.sh
# Expected: [BUILD] ✓ SUCCESS
```

### 3. Quick test (verify binary works)
```bash
mpirun --allow-run-as-root -np 1 ./nbody 500 30 0.005 0
# Should print [step 0/30] ... GFLOPS=...
```

### 4. Start the dashboard backend (Terminal 1)
```bash
python3 dashboard/server.py
# Prints: N-Body Dashboard → http://localhost:5050
```

### 5. Open the dashboard (Terminal 2)
```bash
xdg-open http://localhost:5050
# OR open manually in your browser
```

---

## USING THE DASHBOARD

| Feature | How to use |
|---------|-----------|
| Theme | Click coloured dots (top right) — 5 themes |
| Bodies N | Slide to set particle count (100–20K) |
| Initial Condition | Plummer / Disk / Collision / Uniform |
| Run | Click ▶ RUN SIMULATION |
| Stop | Click ■ STOP (shows popup) |
| Galaxy view | Tab 1 — real-time trajectories |
| Energy charts | Tab 2 — KE/PE/TE conservation |
| Phase space | Tab 7 (Analysis) |
| Data Table | Tab 6 — full CSV rows |

---

## PERFORMANCE TIPS

- Start with N=2000, steps=100 for fast demo
- N=10000 on GPU: ~1-5 seconds per run
- N=50000 on GPU: ~60-120 seconds (normal)
- Use MPI=1 unless you have a cluster

---

## TROUBLESHOOTING

### "Binary not found"
```bash
bash scripts/build.sh
```

### "Cannot reach backend"
```bash
python3 dashboard/server.py
```

### Stop button not working
The new server uses process group kill — should work. If not:
```bash
pkill -f "mpirun"
```

### Port 5050 in use
```bash
PORT=5051 python3 dashboard/server.py
# Then open http://localhost:5051
```

### No GPU
The code automatically uses CPU+OpenMP. Works fine, just slower.

---

## WHAT'S NEW vs v1

- ✅ Fixed stop button (process group kill)
- ✅ Success popup with confetti animation
- ✅ Stop popup confirmation
- ✅ Error popup with message
- ✅ 5 colour themes (Cosmic/Nebula/Solar/Arctic/Matrix)
- ✅ 4 initial conditions (Plummer/Disk/Collision/Uniform)
- ✅ Galaxy view with animated trails
- ✅ Phase space chart (KE vs PE)
- ✅ Virial ratio chart
- ✅ Total energy deviation chart
- ✅ Animated progress bar
- ✅ Live stat cards with animation
- ✅ Confetti on completion
- ✅ Performance breakdown bars
- ✅ Optimized tiled CUDA kernel (10-20× faster)
- ✅ Estimated N<20000 runs fast enough for demo
