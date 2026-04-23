# 🚀 N-Body HPC Simulation (Ubuntu Setup)

A high-performance **N-body gravitational simulation** built and executed in an Ubuntu environment using parallel computing and GPU acceleration.

---

## 📌 Features

* 🌌 Real-time galaxy simulation
* ⚡ High-performance computation (CUDA + MPI)
* 📊 Live performance metrics (GFLOPS, energy drift)
* 🎛️ Interactive dashboard controls
* 🧪 Plummer Sphere initial condition

---

## 🖥️ Tech Stack

* **OS**: Ubuntu (Linux)
* **Frontend**: HTML, CSS, JavaScript
* **Backend**: Python
* **Core Engine**: CUDA (C++)
* **Parallel Computing**: MPI

---

## 📂 Project Structure

```id="x3g6rb"
nbody2/
│── src/                # CUDA simulation code
│── dashboard/          # Web UI
│── scripts/            # Build scripts
│── SETUP.md
│── README.md
```

---

## ⚙️ Installation (Ubuntu)

### 1️⃣ Update system

```bash id="m6o8hc"
sudo apt update && sudo apt upgrade -y
```

---

### 2️⃣ Install dependencies

```bash id="j1r9yv"
sudo apt install build-essential python3 python3-pip mpich -y
```

---

### 3️⃣ Install CUDA Toolkit

Download from NVIDIA or install via:

```bash id="klj2sv"
nvcc --version
```

✔ Ensure CUDA is installed

---

### 4️⃣ Install Python packages

```bash id="bnm8zt"
pip3 install -r requirements.txt
```

---

### 5️⃣ Build the project

```bash id="v7o2hf"
cd scripts
bash build.sh
```

---

## ▶️ Run the Project

### Start backend server

```bash id="z7fdq1"
cd dashboard
python3 server.py
```

---

### Open browser

```id="9u8w3c"
http://localhost:5000
```

---

## 🎮 Usage

1. Set parameters:

   * Number of bodies (N)
   * Time steps
   * Timestep (Δt)
2. Select initial condition (Plummer Sphere)
3. Click **Run Simulation**
4. Observe real-time galaxy motion

---

## 📊 Concepts Used

* N-body gravitational simulation
* Leapfrog integration method
* Parallel processing (MPI / GPU)
* Performance optimization (GFLOPS)

---

## ⚠️ Requirements

* Ubuntu OS
* NVIDIA GPU (recommended)
* CUDA Toolkit
* Python 3

---

## 🚀 Future Enhancements

* Multi-GPU support
* 3D visualization
* Cloud deployment

---

## 👨‍💻 Author

**Jayakrishna**
GitHub: https://github.com/jayakrishna2004

---
