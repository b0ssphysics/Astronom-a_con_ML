import numpy as np
from scipy.integrate import simpson
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# --- Parámetros Físicos ---
# Nota: A 2600m (Bogotá), la densidad suele ser ~0.9 kg/m3, 
# pero mantengo tus valores para no alterar tu modelo original.
rho_bogota = 1.22 
h_atmos = 8000 
kappa = 0.01 
tiempo = np.linspace(0, 24, 96) 
smog_factor = 0.5 + 0.5 * np.sin(2 * np.pi * (tiempo - 6) / 24)**2

# --- Geometría ---
h_avion = np.linspace(10000, 100, 80) 
d = 14000 
F_emitido = 1.0 
F_referencia = 1.0 

magnitudes_tiempo = []
distancias = []

# --- Procesamiento ---
for sf in smog_factor:
    temp_mags = []
    for h in h_avion:
        L = np.sqrt(d**2 + h**2)
        seno_theta = h / L
        
        N = 1000
        s = np.linspace(0, L, N)
        z = s * seno_theta
        rho = rho_bogota * np.exp(-z / h_atmos)
        
        tau = simpson(y=(kappa * sf) * rho, x=s)
        F_observado = (F_emitido / (4 * np.pi * L**2)) * np.exp(-tau)
        m = -2.5 * np.log10(F_observado / F_referencia)
        temp_mags.append(m)
        
        if len(distancias) < len(h_avion):
            distancias.append(L / 1000)
    magnitudes_tiempo.append(temp_mags)

# --- Animación Optimizada ---
fig, ax = plt.subplots(figsize=(8, 5))
line, = ax.plot([], [], lw=2, color='#2c3e50')
ax.set_xlim(max(distancias), min(distancias))
ax.set_ylim(np.min(magnitudes_tiempo), np.max(magnitudes_tiempo))
ax.set_xlabel('Distancia al horizonte (km)')
ax.set_ylabel('Magnitud observada')
title = ax.set_title('')
ax.grid(True, linestyle='--', alpha=0.7)

def init():
    line.set_data([], [])
    return line,

def update(frame):
    line.set_data(distancias, magnitudes_tiempo[frame])
    title.set_text(f'Evolución de la extinción - Tiempo: {tiempo[frame]:.2f} h')
    return line, title

ani = FuncAnimation(fig, update, frames=len(tiempo), init_func=init, blit=True)

# Guardar como GIF usando Pillow (más compatible que ImageMagick)
ani.save('magnitud_observada.gif', writer='pillow', fps=10)
plt.close() # Evita que se duplique la imagen en notebooks

# --- Gráfica Estática Final ---
plt.figure(figsize=(9, 6))
cmap = plt.cm.plasma # Un mapa de color con mejor contraste
colors = cmap(np.linspace(0, 1, len(tiempo)))

for i, m in enumerate(magnitudes_tiempo):
    plt.plot(distancias, m, color=colors[i], alpha=0.2)

plt.gca().invert_xaxis()
plt.xlabel('Distancia al horizonte (km)')
plt.ylabel('Magnitud observada')
plt.title('Evolución de la Magnitud en 24 Horas')
plt.grid(True, which='both', linestyle=':', alpha=0.5)

sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=24))
sm.set_array([])
cbar = plt.colorbar(sm, ax=plt.gca())
cbar.set_label('Hora del día (h)')

# Guardar en formato PDF o PNG de alta resolución
plt.tight_layout() # Asegura que nada se corte
plt.savefig('reporte_extincion_24h.png', dpi=300)
plt.savefig('reporte_extincion_24h.pdf') # Formato vectorial ideal para la tarea
plt.show()
