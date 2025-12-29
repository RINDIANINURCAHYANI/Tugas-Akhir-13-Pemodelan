import matplotlib
matplotlib.use("Agg")

import pandas as pd
import numpy as np
from flask import Flask, render_template
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import os

app = Flask(__name__)

# =========================
# Load Dataset
# =========================
df = pd.read_csv("bitcoin.csv")

price_mean = df['Close'].mean()
volume_mean = df['Volume_(BTC)'].mean()

# =========================
# Model Predator–Prey
# =========================
def bitcoin_model(t, z, alpha, beta, delta, gamma):
    price, volume = z
    dprice_dt = alpha * price - beta * price * volume
    dvolume_dt = delta * price * volume - gamma * volume
    return [dprice_dt, dvolume_dt]

alpha, beta, delta, gamma = 0.02, 0.00001, 0.00002, 0.03
initial_state = [price_mean, volume_mean]

t_eval = np.linspace(0, 200, 1000)
sol = solve_ivp(
    bitcoin_model,
    (0, 200),
    initial_state,
    args=(alpha, beta, delta, gamma),
    t_eval=t_eval
)

price = sol.y[0]
volume = sol.y[1]

# =========================
# Generate Plots (STATIC)
# =========================
os.makedirs("static/plots", exist_ok=True)

plt.figure(figsize=(8,4))
plt.plot(t_eval, price, label="Harga Bitcoin")
plt.plot(t_eval, volume, label="Volume Transaksi")
plt.title("Dinamika Harga Bitcoin dan Volume Transaksi")
plt.xlabel("Waktu")
plt.ylabel("Nilai")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("static/plots/timeseries.png")
plt.close()

plt.figure(figsize=(5,5))
plt.plot(price, volume)
plt.title("Phase Portrait Predator–Prey Bitcoin")
plt.xlabel("Harga Bitcoin")
plt.ylabel("Volume Transaksi")
plt.grid(True)
plt.tight_layout()
plt.savefig("static/plots/phase_portrait.png")
plt.close()

@app.route("/")
def index():
    return render_template(
        "index.html",
        price_avg=round(price_mean, 2),
        volume_avg=round(volume_mean, 2)
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
