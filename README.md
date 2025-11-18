# 🤖 Robotic Arm PSO Optimization

A fully interactive **Particle Swarm Optimization (PSO)**–based robotic arm simulation built using **Streamlit**, **NumPy**, **Pandas**, and **Plotly**.  
It visualizes a 2-link robotic arm reaching any target point (**XE, YE, ZE**) using PSO and displays a live 3D animation, particle updates, convergence graph, and full particle table.

---


[![Deploy App](https://img.shields.io/badge/🚀_Open_App-Streamlit-blue?style=for-the-badge)](https://robotic-arm-pso-optimization.streamlit.app/)



## 🌟 Features

### 🔵 Real-Time PSO Simulation
- Live optimization updates  
- Adjustable PSO parameters (W, C1, C2, particles, iterations)  
- Global best & personal best tracking  
- Early stopping when fitness ≈ 0.0002  

### 🤖 3D Robot Arm Visualization
- Forward Kinematics for 2-link planar arm  
- Plotly 3D visualization  
- Shows best particle arm path  
- Shows all particle end-effectors  
- Highlights target point  

### 📊 Live Monitoring Dashboard
- Best-so-far metrics (θ1, θ2, D, fitness)  
- Dynamic convergence line chart  
- Full particle tracking table (positions, velocities, fitness)  
- Smooth frame-by-frame animation  

### 🎛 User Controls
- Select target XE, YE, ZE  
- Edit link lengths (L1, L2)  
- Start / Stop PSO anytime  
- Auto re-run with previous configuration  

---

## 🛠 Technologies Used
- Python 3  
- Streamlit  
- NumPy  
- Pandas  
- Plotly  

---

## 📂 Project Structure
```yaml
robotic-arm-pso-optimization/
│── app.py
│── requirements.txt
│── .gitignore
│── README.md
```

---

## ▶️ Run Locally

### 1️⃣ Clone the repository
```bash
git clone https://github.com/Ayush1202R/robotic-arm-pso-optimization.git
cd robotic-arm-pso-optimization
```

### 2️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Run the Streamlit app
```bash
streamlit run app.py
```

---

## 📌 Example Output
```yaml
Best so far → Theta1: 38.85°
Theta2: -35.95°
D: 44.00
Best Fitness Value: 0.00
```

---



## 🤝 Contributing
Pull requests and issues are welcome — improve the model, UI, or visualization.


