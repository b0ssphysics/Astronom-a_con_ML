# Tarea 1 - Astronomía con ML

# La idea de éste código es modelar la masa de aire y extinción de luz 

#La forma en la que elegí modelar el sistema, ya que hay varias suposiciones físicas que hay que hacer.

#Primeramente, asumimos que el horizonte no es infinito, de manera que la masa de aire pueda modelarse de manera determinada. Además,
#como la observación se está haciendo desde la universidad de los Andes, entonces sabemos la Altitud del observatorio es conocida, 
# y se está ignorando el efecto de la curvatura de la tierra, ya que la curvatura aproximada en para 14km es approx 15 metros, 
# que comparado a 14km, hace que la corrección sea de segundo orden.

#paquetes necesarios
import numpy as np
from scipy.integrate import simpson
import matplotlib.pyplot as plt

#Constantes y parámetros físicos

rho_bogota=1.22 #kg/m3 Densidad del aire en Bogotá

h_atmos=8000 #m Altitud característica de la atmósfera
kappa = 0.01 #m2/kg Coeficiente de extinción de luz en el aire

tiempo = np.linspace(0, 24, 96) #s Tiempo de observación de 24 horas, con 15 minutos de resolución

#quiero una función que modela la fuerza del smog en función del tiempo del día, asumiendo que el smog es mayor en las horas pico de la mañana y la tarde
smog_factor = 0.5 + 0.5 * np.sin(2 * np.pi * (tiempo - 6) / 24)**2 #Factor de smog varía entre 0.5 y 1.0

#Geometría del problema

h_avion = np.linspace(10000,100, 80) #m Altitud del avión desde 10km hasta 100m, con 80 puntos para la integración
d = 14000 #m Distancia al horizonte

F_emitido = 1.0 # Flujo emitido por la estrella (Vega)
F_referencia = 1.0 # Flujo de referencia (Vega)

mangitudes = [] #Lista para almacenar las magnitudes calculadas
distancias = [] #Lista para almacenar las distancias al horizonte

magnitudes_tiempo = []  #Almacenar magnitudes para cada tiempo del día

for sf in smog_factor:
    mangitudes = []

    for h in h_avion:
        L = np.sqrt(d**2 + h**2) #m Longitud de la línea de visión del avión al horizonte, pitagorazo
        seno_theta = h / L #Seno del ángulo de observación

        #Integración line of sight

        N = 1000 #Número de puntos para la integración
        s = np.linspace(0, L, N) #Vector de posiciones a lo largo de la línea de visión
        z = s * seno_theta #Altitud en cada punto de la línea de visión
        rho = rho_bogota * np.exp(-z / h_atmos) #Densidad del aire en cada punto de la línea de visión

        #Cálculo de la masa de aire a lo largo de la línea de visión
        tau = simpson((kappa * sf) * rho, s) #Integral de la extinción a lo largo de la línea de visión

        #calculo de los flujos y magnitude
        F_observado = (F_emitido / (4 * np.pi * L**2 )) * np.exp(-tau)# Flujo observado después de la extinción

        m = -2.5 * np.log10(F_observado / F_referencia) # Cambio en magnitud debido a la extinción

        mangitudes.append(m)

        if len(distancias) < len(h_avion):
            distancias.append(L / 1000) #Convertir a km

    magnitudes_tiempo.append(mangitudes)


from matplotlib.animation import FuncAnimation
fig, ax = plt.subplots()
line, = ax.plot([], [], lw=2)

ax.set_xlim(max(distancias), min(distancias))
ax.set_ylim(
    np.min(magnitudes_tiempo),
    np.max(magnitudes_tiempo)
)

ax.set_xlabel('Distancia al horizonte (km)')
ax.set_ylabel('Magnitud observada')
title = ax.set_title('')
ax.grid()

def init():
    line.set_data([], [])
    title.set_text('')
    return line, title

def update(frame):
    y = magnitudes_tiempo[frame]
    line.set_data(distancias, y)
    title.set_text(f'Tiempo: {tiempo[frame]:.2f} h')
    return line, title

ani = FuncAnimation(
    fig,
    update,
    frames=len(tiempo),
    init_func=init,
    interval=200,
    blit=True
)

plt.show()



