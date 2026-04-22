#!/usr/bin/env python3
"""
server.py  —  N-Body HPC Dashboard Backend
Fixed: stop button, SSE streaming, CORS, process management
Install: pip3 install flask flask-cors --break-system-packages
Run:     python3 dashboard/server.py
"""

import os, sys, csv, json, time, signal, subprocess, threading
from pathlib import Path
from flask import Flask, jsonify, request, Response, send_from_directory
from flask_cors import CORS

BASE    = Path(__file__).parent.parent.resolve()
BIN     = BASE / "nbody"
RESULTS = BASE / "results"
LOG     = RESULTS / "sim_log.csv"
RESULTS.mkdir(exist_ok=True)

app = Flask(__name__, static_folder=str(BASE / "dashboard"))
CORS(app, resources={r"/api/*": {"origins": "*"}})

# ── shared state ───────────────────────────────────────────────
class SimState:
    def __init__(self):
        self.lock     = threading.Lock()
        self.running  = False
        self.output   = []
        self.proc     = None   # subprocess.Popen object
        self.error    = None
        self.params   = {}
        self.done     = False

state = SimState()

# ── helpers ────────────────────────────────────────────────────
def parse_log():
    rows = []
    if not LOG.exists():
        return rows
    try:
        with open(LOG, newline="") as f:
            for row in csv.DictReader(f):
                d = {}
                for k, v in row.items():
                    try:    d[k] = float(v)
                    except: d[k] = 0.0
                rows.append(d)
    except Exception as e:
        pass
    return rows

def kill_proc():
    """Kill the running subprocess safely."""
    with state.lock:
        proc = state.proc
    if proc is None:
        return
    try:
        # Kill entire process group
        pgid = os.getpgid(proc.pid)
        os.killpg(pgid, signal.SIGTERM)
        time.sleep(0.3)
        try: os.killpg(pgid, signal.SIGKILL)
        except: pass
    except Exception:
        try: proc.kill()
        except: pass
    with state.lock:
        state.proc    = None
        state.running = False

def run_thread(n, steps, dt, np, ic):
    cmd = ["mpirun", "--allow-run-as-root", "-np", str(np),
           str(BIN), str(n), str(steps), str(dt), str(ic)]

    with state.lock:
        state.running = True
        state.output  = [f"$ {' '.join(cmd)}"]
        state.error   = None
        state.done    = False

    # Clear old log
    if LOG.exists():
        LOG.unlink()

    try:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, cwd=str(BASE),
            preexec_fn=os.setsid   # new process group for clean kill
        )
        with state.lock:
            state.proc = proc

        for line in proc.stdout:
            line = line.rstrip()
            with state.lock:
                if not state.running:
                    break
                state.output.append(line)

        proc.wait()
        rc = proc.returncode
        with state.lock:
            if rc != 0 and rc != -15:   # -15 = SIGTERM (user stop)
                state.error = f"Process exited with code {rc}"
            state.done = True

    except FileNotFoundError:
        with state.lock:
            state.error = "Binary not found. Run: bash scripts/build.sh"
            state.done  = True
    except Exception as e:
        with state.lock:
            state.error = str(e)
            state.done  = True
    finally:
        with state.lock:
            state.running = False
            state.proc    = None

# ── routes ─────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(str(BASE / "dashboard"), "index.html")

@app.route("/api/status")
def api_status():
    with state.lock:
        return jsonify({
            "running":     state.running,
            "done":        state.done,
            "binary_ok":   BIN.exists(),
            "has_results": LOG.exists(),
            "error":       state.error,
            "n_lines":     len(state.output),
            "params":      state.params,
        })

@app.route("/api/run", methods=["POST"])
def api_run():
    with state.lock:
        if state.running:
            return jsonify({"error": "Already running"}), 409

    d     = request.get_json(force=True) or {}
    n     = max(100,  min(int(d.get("n",     5000)), 50000))
    steps = max(10,   min(int(d.get("steps",  200)), 2000))
    dt    = float(d.get("dt",   0.005))
    np    = max(1,    min(int(d.get("np",       1)), 8))
    ic    = int(d.get("ic", 0))

    with state.lock:
        state.params = {"n":n,"steps":steps,"dt":dt,"np":np,"ic":ic}

    t = threading.Thread(target=run_thread, args=(n,steps,dt,np,ic), daemon=True)
    t.start()
    return jsonify({"ok": True, "n":n, "steps":steps, "dt":dt, "np":np, "ic":ic})

@app.route("/api/stop", methods=["POST"])
def api_stop():
    kill_proc()
    with state.lock:
        state.running = False
        state.done    = True
        state.output.append(">> Simulation stopped by user.")
    return jsonify({"ok": True})

@app.route("/api/output")
def api_output():
    offset = int(request.args.get("offset", 0))
    with state.lock:
        lines   = state.output[offset:]
        running = state.running
        error   = state.error
        done    = state.done
    return jsonify({"lines": lines, "running": running, "error": error, "done": done})

@app.route("/api/results")
def api_results():
    rows = parse_log()
    if not rows:
        return jsonify({"rows": [], "summary": {}})

    first_te = rows[0].get("te", 0)
    last_te  = rows[-1].get("te", 0)
    drift    = abs((last_te - first_te) / (abs(first_te) + 1e-30)) * 100

    gflops = 0.0
    with state.lock:
        for line in reversed(state.output):
            if "GFLOPS=" in line:
                try:
                    idx = line.index("GFLOPS=") + 7
                    gflops = float(line[idx:].split()[0])
                    break
                except: pass

    summary = {
        "steps":        len(rows),
        "energy_drift": round(drift, 5),
        "initial_te":   first_te,
        "final_te":     last_te,
        "gflops":       round(gflops, 2),
    }
    return jsonify({"rows": rows, "summary": summary})

@app.route("/api/stream")
def api_stream():
    """Server-Sent Events for real-time log streaming."""
    def gen():
        sent = 0
        while True:
            with state.lock:
                lines   = state.output[sent:]
                running = state.running
                error   = state.error
                done    = state.done
            for line in lines:
                yield f"data: {json.dumps({'line': line})}\n\n"
                sent += 1
            if error:
                yield f"data: {json.dumps({'error': error, 'done': True})}\n\n"
                break
            if done and not running:
                yield f"data: {json.dumps({'done': True})}\n\n"
                break
            time.sleep(0.25)

    return Response(gen(), mimetype="text/event-stream",
                    headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})

@app.route("/api/clear", methods=["POST"])
def api_clear():
    with state.lock:
        state.output = []
        state.error  = None
        state.done   = False
    if LOG.exists(): LOG.unlink()
    return jsonify({"ok": True})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    print(f"\n  ╔══════════════════════════════════════╗")
    print(f"  ║  N-Body Dashboard  →  http://localhost:{port}  ║")
    print(f"  ╚══════════════════════════════════════╝")
    print(f"  Binary : {BIN}  ({'✓ READY' if BIN.exists() else '✗ NOT BUILT'})")
    print(f"  Results: {RESULTS}\n")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
