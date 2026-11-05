import numpy as np
import matplotlib.pyplot as plt

class Miembro:
    def __init__(self, nombre, L, I, E=1.0, angulo=0, carga_dist=0):
        self.nombre = nombre
        self.L = L
        self.I = I
        self.E = E
        self.angulo = np.radians(angulo)
        self.carga_dist = carga_dist
        self.K = self.I / self.L if self.L != 0 else 0
        self.M_izq = 0.0
        self.M_der = 0.0

    def calcular_mep(self):
        if self.carga_dist != 0:
            mep = (self.carga_dist * self.L**2) / 12.0
            return -mep, mep
        return 0.0, 0.0

class Nodo:
    def __init__(self, nombre, apoyado=False):
        self.nombre = nombre
        self.miembros = []
        self.apoyado = apoyado
        self.momento_desequilibrado = 0.0

    def calcular_fd(self):
        suma_K = sum([m.K for m in self.miembros])
        fds = []
        for m in self.miembros:
            fds.append(m.K / suma_K if suma_K > 0 else 0)
        return fds

def cross_iterativo(nodos, miembros, iteraciones=4):
    for m in miembros:
        mep_izq, mep_der = m.calcular_mep()
        m.M_izq += mep_izq
        m.M_der += mep_der

    for _ in range(iteraciones):
        for nodo in nodos:
            if nodo.apoyado: continue
            M_des = sum([m.M_izq if m.nombre.startswith(nodo.nombre) else m.M_der for m in nodo.miembros])
            fds = nodo.calcular_fd()
            
            for i, m in enumerate(nodo.miembros):
                if m.nombre.startswith(nodo.nombre):
                    m.M_izq -= M_des * fds[i]
                else:
                    m.M_der -= M_des * fds[i]
                    
        for m in miembros:
            trans_izq = m.M_der * 0.5
            trans_der = m.M_izq * 0.5
            m.M_izq += trans_izq
            m.M_der += trans_der

def plot_diagramas(miembros, titulo):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    fig.suptitle(f"Diagramas para {titulo}")
    
    x_total = 0
    for m in miembros:
        x = np.linspace(x_total, x_total + m.L, 100)
        
        V = np.linspace((m.M_izq + m.M_der)/m.L, -(m.M_izq + m.M_der)/m.L, 100) - m.carga_dist * (x - x_total - m.L/2)
        M = np.linspace(m.M_izq, m.M_der, 100) + (m.carga_dist * (x - x_total) * (m.L - (x - x_total))) / 2
        
        ax1.plot(x, V, label=f'Cortante {m.nombre}')
        ax1.fill_between(x, V, alpha=0.3)
        
        ax2.plot(x, M, label=f'Momento {m.nombre}')
        ax2.fill_between(x, M, alpha=0.3)
        
        x_total += m.L
        
    ax1.set_ylabel("Fuerza Cortante (ton)")
    ax1.axhline(0, color='black', linewidth=1)
    ax1.grid(True)
    
    ax2.set_ylabel("Momento Flexionante (ton-m)")
    ax2.axhline(0, color='black', linewidth=1)
    ax2.grid(True)
    ax2.invert_yaxis()
    
    plt.show()

def resolver_ejemplo_9_2(cargas_dist=[0, 3, 3, 2]):
    h1, h2, h3 = 4.0, 3.0, 3.0
    L = 5.0
    
    col_AB = Miembro("AB", h1, 4.0)
    col_BC = Miembro("BC", h2, 3.0)
    col_CD = Miembro("CD", h3, 2.0)
    viga_BG = Miembro("BG", L, 1.0, carga_dist=cargas_dist[1])
    viga_CF = Miembro("CF", L, 1.0, carga_dist=cargas_dist[2])
    viga_DE = Miembro("DE", L, 1.0, carga_dist=cargas_dist[3])

    miembros_92 = [col_AB, col_BC, col_CD, viga_BG, viga_CF, viga_DE]
    
    plot_diagramas([viga_BG, viga_CF, viga_DE], "Ejemplo 9.2 - Vigas Principales")

def resolver_marco_inclinado(carga_P=3.0, angulo_CD=40, carga_dist_BC=2.0):
    L_AB = 3.0
    L_BC = 3.5
    L_CD = 2.0 / np.cos(np.radians(angulo_CD))
    
    col_AB = Miembro("AB", L_AB, 1.0)
    viga_BC = Miembro("BC", L_BC, 1.0, carga_dist=carga_dist_BC)
    col_CD = Miembro("CD", L_CD, 1.0, angulo=angulo_CD)
    
    nodoA = Nodo("A", apoyado=True)
    nodoB = Nodo("B")
    nodoC = Nodo("C")
    nodoD = Nodo("D", apoyado=True)
    
    nodoB.miembros = [col_AB, viga_BC]
    nodoC.miembros = [viga_BC, col_CD]
    
    nodos_incl = [nodoA, nodoB, nodoC, nodoD]
    miembros_incl = [col_AB, viga_BC, col_CD]
    
    cross_iterativo(nodos_incl, miembros_incl, iteraciones=4)
    
    plot_diagramas(miembros_incl, "Marco Inclinado con Carga Distribuida")

resolver_ejemplo_9_2(cargas_dist=[0, 3, 3, 2])
resolver_marco_inclinado(carga_P=3.0, angulo_CD=40, carga_dist_BC=2.5)
