import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import os

# ===============================
# Load Dataset
# ===============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "bitcoin.csv")

df = pd.read_csv(DATA_PATH)

# Normalisasi kolom
df.columns = df.columns.str.strip().str.lower()

print("Kolom dataset:", df.columns)

# Gunakan kolom yang BENAR
price_series = df['close']
volume_series = df['volume']

price_mean = price_series.mean()
volume_mean = volume_series.mean()

# ===============================
# Model Predator–Prey (Lotka–Volterra)
# ===============================
def bitcoin_predator_prey(t, z, alpha, beta, delta, gamma):
    price, volume = z
    dprice_dt = alpha * price - beta * price * volume
    dvolume_dt = delta * price * volume - gamma * volume
    return [dprice_dt, dvolume_dt]

# Parameter
alpha = 0.02
beta = 0.00001
delta = 0.00002
gamma = 0.03

initial_state = [price_mean, volume_mean]

# ===============================
# Simulasi
# ===============================
t_eval = np.linspace(0, 200, 1000)

sol = solve_ivp(
    bitcoin_predator_prey,
    (0, 200),
    initial_state,
    args=(alpha, beta, delta, gamma),
    t_eval=t_eval
)

t = sol.t
price = sol.y[0]
volume = sol.y[1]

# ===============================
# Visualisasi
# ===============================
plt.figure(figsize=(10, 5))
plt.plot(t, price, label="Harga Bitcoin (Prey)")
plt.plot(t, volume, label="Volume Transaksi (Predator)")
plt.xlabel("Waktu")
plt.ylabel("Nilai")
plt.title("Dinamika Interaksi Harga Bitcoin dan Volume Transaksi")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 6))
plt.plot(price, volume)
plt.xlabel("Harga Bitcoin")
plt.ylabel("Volume Transaksi")
plt.title("Phase Portrait Model Predator–Prey Bitcoin")
plt.grid(True)
plt.tight_layout()
plt.show()
