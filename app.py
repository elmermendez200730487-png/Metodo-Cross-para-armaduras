import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

st.set_page_config(page_title="Resolución Hardy Cross", layout="wide")

st.title("Análisis Estructural - Método de Hardy Cross Completo")
st.markdown("Resolución paso a paso con tablas iterativas y diagramas de Cortante y Momento Flexionante.")

# --- CONFIGURACIÓN DE UNIDADES ---
st.sidebar.header("Configuración de Unidades")
u_fuerza = st.sidebar.selectbox("Unidad de Fuerza", ["ton", "N", "lb"])
u_long = st.sidebar.selectbox("Unidad de Longitud", ["m", "ft", "in"])
u_ang = st.sidebar.selectbox("Unidad de Ángulo", ["Grados", "Radianes"])

# ==========================================
# FUNCIONES DE DIBUJO DE DIAGRAMAS
# ==========================================

def draw_support(ax, x, y, nx=0, ny=-1):
    """Dibuja un soporte fijo (empotramiento) en la base."""
    if nx == 0:  # Horizontal
        ax.plot([x-0.4, x+0.4], [y, y], 'k-', lw=2)
        for i in np.linspace(-0.4, 0.3, 6):
            ax.plot([x+i, x+i+0.1], [y, y-0.2], 'k-', lw=1)
    else: # Inclinado o adaptado
        ax.plot([x-0.4*ny, x+0.4*ny], [y+0.4*nx, y-0.4*nx], 'k-', lw=2)
        for i in np.linspace(-0.4, 0.3, 6):
            xi, yi = x + i*ny, y - i*nx
            ax.plot([xi, xi+0.1*nx], [yi, yi+0.1*ny], 'k-', lw=1)

def draw_diagram_member(ax, x1, y1, x2, y2, v1, v2, scale, color='blue', label1="", label2="", draw_member=True):
    """Dibuja el diagrama perpendicular al miembro con achurado (sombreado de líneas)."""
    dx, dy = x2 - x1, y2 - y1
    L = math.hypot(dx, dy)
    if L == 0: return
    nx, ny = -dy/L, dx/L # Vector perpendicular
    
    # Puntos del diagrama escalados
    px1, py1 = x1 + v1 * scale * nx, y1 + v1 * scale * ny
    px2, py2 = x2 + v2 * scale * nx, y2 + v2 * scale * ny
    
    # Dibujar la barra física
    if draw_member:
        ax.plot([x1, x2], [y1, y2], 'k-', lw=2.5)
        
    # Dibujar contorno del diagrama
    ax.plot([x1, px1, px2, x2], [y1, py1, py2, y2], color=color, lw=1.2)
    ax.plot([px1, px2], [py1, py2], color=color, lw=1.2)
    
    # Achurado (líneas perpendiculares a la barra)
    num_hatches = max(10, int(L * 10))
    for i in range(1, num_hatches):
        t = i / num_hatches
        bx, by = x1 + t*dx, y1 + t*dy
        vt = v1 + t*(v2 - v1)
        tx, ty = bx + vt * scale * nx, by + vt * scale * ny
        ax.plot([bx, tx], [by, ty], color=color, lw=0.6, alpha=0.6)
        
    # Etiquetas de valores
    if abs(v1) > 0.001:
        ax.text(px1 + 0.5*nx, py1 + 0.5*ny, label1, color='black', fontsize=10, weight='bold', ha='center', va='center')
    if abs(v2) > 0.001:
        ax.text(px2 + 0.5*nx, py2 + 0.5*ny, label2, color='black', fontsize=10, weight='bold', ha='center', va='center')

# ==========================================
# FUNCIÓN GENERADORA DE TABLAS DE HARDY CROSS
# ==========================================

def tabla_cross_paso_a_paso(nodos_fd, barras, FEM, iteraciones, orden_miembros, nombre_m_inicial="M. IMPUESTO"):
    columnas_tuplas = [(m[0], m) for m in orden_miembros]
    columnas = pd.MultiIndex.from_tuples(columnas_tuplas, names=['NODO', 'MIEMBRO'])
    
    fd_row = []
    for m in orden_miembros:
        n1, n2 = m[0], m[1:]
        if len(m) > 2 and 'Vol' in m: fd_row.append(0.0)
        else: fd_row.append(nodos_fd.get(n1, {}).get(n2, 0.0))
            
    m_row = [FEM.get(m, 0.0) for m in orden_miembros]
    filas = [fd_row, m_row]
    nombres_filas = ['F.D.', nombre_m_inicial]
    
    M_actual = {m: FEM.get(m, 0.0) for m in orden_miembros}
    
    for i in range(1, iteraciones + 1):
        desequilibrios = {n: 0.0 for n in nodos_fd.keys()}
        for m in orden_miembros:
            if m[0] in desequilibrios: desequilibrios[m[0]] += M_actual[m]
                
        dist_row_dict = {}
        for m in orden_miembros:
            n1, n2 = m[0], m[1:]
            if n1 in nodos_fd and n2 in nodos_fd[n1]:
                dist = -desequilibrios[n1] * nodos_fd[n1][n2]
                dist_row_dict[m] = dist
                M_actual[m] += dist
            else: dist_row_dict[m] = 0.0
                
        filas.append([dist_row_dict[m] for m in orden_miembros])
        nombres_filas.append(f"{i}a. Dist.")
        
        trans_row_dict = {m: 0.0 for m in orden_miembros}
        for n1, n2 in barras:
            m_ida, m_vuelta = f"{n1}{n2}", f"{n2}{n1}"
            if m_ida in orden_miembros and m_vuelta in orden_miembros:
                trans_row_dict[m_ida] += dist_row_dict.get(m_vuelta, 0.0) * 0.5
                trans_row_dict[m_vuelta] += dist_row_dict.get(m_ida, 0.0) * 0.5
                M_actual[m_ida] += trans_row_dict[m_ida]
                M_actual[m_vuelta] += trans_row_dict[m_vuelta]
                
        filas.append([trans_row_dict[m] for m in orden_miembros])
        nombres_filas.append(f"{i}o. Transp.")
        
    filas.append([M_actual[m] for m in orden_miembros])
    nombres_filas.append('Momentos Finales')
    
    df = pd.DataFrame(filas, index=nombres_filas, columns=columnas)
    return df, M_actual

# ==========================================
# SELECCIÓN DE EJERCICIOS
# ==========================================
ejercicio = st.sidebar.radio("Selecciona el tipo de marco:", ("Ejercicio 1: Marco de 3 Niveles", "Ejercicio 2: Marco con Columna Inclinada"))

if ejercicio == "Ejercicio 1: Marco de 3 Niveles":
    st.header("Marco con Tres Niveles y Una Crujía")
    
    st.subheader("Parámetros de Entrada")
    col1, col2, col3 = st.columns(3)
    with col1:
        L1 = st.number_input(f"Longitud viga L1 ({u_long})", value=5.0)
        L2 = st.number_input(f"Voladizo L2 ({u_long})", value=1.5)
        L3 = st.number_input(f"Altura N1 L3 ({u_long})", value=4.0)
        L4 = st.number_input(f"Altura N2 L4 ({u_long})", value=3.0)
        L5 = st.number_input(f"Altura N3 L5 ({u_long})", value=3.0)
    with col2:
        w1 = st.number_input(f"w1 - Distribuida N3 ({u_fuerza}/{u_long})", value=2.0)
        P1 = st.number_input(f"P1 - Puntual Vol. N3 ({u_fuerza})", value=2.0)
        F1 = st.number_input(f"F1 - Lateral N3 ({u_fuerza})", value=8.0)
        w2 = st.number_input(f"w2 - Distribuida N2 ({u_fuerza}/{u_long})", value=3.0)
        P2 = st.number_input(f"P2 - Puntual Vol. N2 ({u_fuerza})", value=2.0)
        F2 = st.number_input(f"F2 - Lateral N2 ({u_fuerza})", value=6.0)
        w3 = st.number_input(f"w3 - Distribuida N1 ({u_fuerza}/{u_long})", value=3.0)
        P3 = st.number_input(f"P3 - Puntual Vol. N1 ({u_fuerza})", value=2.0)
        F3 = st.number_input(f"F3 - Lateral N1 ({u_fuerza})", value=4.0)
    with col3:
        I_a = st.number_input(f"Inercia Ia ({u_long}^4)", value=4.0)
        I_b = st.number_input(f"Inercia Ib ({u_long}^4)", value=3.0)
        I_c = st.number_input(f"Inercia Ic ({u_long}^4)", value=2.0)
        I_d = st.number_input(f"Inercia Id ({u_long}^4)", value=1.0)

    # --- ESQUEMA INICIAL EJERCICIO 1 ---
    st.markdown("### Esquema del Marco (Actualizado con tus datos)")
    fig_ref1, ax_ref1 = plt.subplots(figsize=(6, 6))
    
    # Dibujar columnas y vigas
    ax_ref1.plot([0, 0], [0, L3+L4+L5], 'k-', lw=3) # Columna Izq
    ax_ref1.plot([L1, L1], [0, L3+L4+L5], 'k-', lw=3) # Columna Der
    for y_val in [L3, L3+L4, L3+L4+L5]:
        ax_ref1.plot([0, L1], [y_val, y_val], 'b-', lw=3) # Vigas
        ax_ref1.plot([L1, L1+L2], [y_val, y_val], 'b-', lw=3) # Voladizos
    
    # Apoyos
    draw_support(ax_ref1, 0, 0)
    draw_support(ax_ref1, L1, 0)
    
    # Textos de cotas
    ax_ref1.text(-0.5, L3/2, f"L3={L3}", rotation=90, va='center')
    ax_ref1.text(-0.5, L3+L4/2, f"L4={L4}", rotation=90, va='center')
    ax_ref1.text(-0.5, L3+L4+L5/2, f"L5={L5}", rotation=90, va='center')
    ax_ref1.text(L1/2, L3+L4+L5+0.5, f"L1={L1}", ha='center')
    ax_ref1.text(L1+L2/2, L3+L4+L5+0.5, f"L2={L2}", ha='center')
    
    # Fuerzas laterales
    ax_ref1.annotate(f"F3={F3}", xy=(0, L3), xytext=(-2, L3), arrowprops=dict(facecolor='red', width=2, headwidth=8), color='red', va='center')
    ax_ref1.annotate(f"F2={F2}", xy=(0, L3+L4), xytext=(-2, L3+L4), arrowprops=dict(facecolor='red', width=2, headwidth=8), color='red', va='center')
    ax_ref1.annotate(f"F1={F1}", xy=(0, L3+L4+L5), xytext=(-2, L3+L4+L5), arrowprops=dict(facecolor='red', width=2, headwidth=8), color='red', va='center')

    ax_ref1.set_xlim(-3, L1+L2+2)
    ax_ref1.set_ylim(-1, L3+L4+L5+2)
    ax_ref1.axis('off')
    st.pyplot(fig_ref1)

    if st.button("Resolver Ejercicio 1"):
        st.write("---")
        K_AB = I_a / L3; K_GH = K_AB
        K_BC = I_b / L4; K_FG = K_BC
        K_CD = I_c / L5; K_EF = K_CD
        K_V = I_d / L1
        
        S_B = K_AB + K_BC + K_V; S_C = K_BC + K_CD + K_V; S_D = K_CD + K_V
        S_E = K_CD + K_V; S_F = K_BC + K_CD + K_V; S_G = K_AB + K_BC + K_V
        
        nodos_fd = {
            'A': {'B': 0}, 'H': {'G': 0},
            'B': {'A': K_AB/S_B, 'C': K_BC/S_B, 'G': K_V/S_B},
            'C': {'B': K_BC/S_C, 'D': K_CD/S_C, 'F': K_V/S_C},
            'D': {'C': K_CD/S_D, 'E': K_V/S_D},
            'E': {'D': K_V/S_E, 'F': K_CD/S_E, 'Vol': 0.0},
            'F': {'C': K_V/S_F, 'E': K_CD/S_F, 'G': K_BC/S_F, 'Vol': 0.0},
            'G': {'B': K_V/S_G, 'F': K_BC/S_G, 'H': K_AB/S_G, 'Vol': 0.0}
        }
        barras = [('A','B'), ('B','C'), ('C','D'), ('D','E'), ('C','F'), ('B','G'), ('E','F'), ('F','G'), ('G','H')]
        orden_miembros = ['AB', 'BA', 'BC', 'CB', 'CD', 'DC', 'DE', 'ED', 'EVol', 'EF', 'FE', 'FVol', 'FC', 'CF', 'FG', 'GF', 'GVol', 'GB', 'BG', 'GH', 'HG']
        
        FEM_0 = {
            'DE': -(w1*L1**2)/12, 'ED': (w1*L1**2)/12, 'CF': -(w2*L1**2)/12, 'FC': (w2*L1**2)/12,
            'BG': -(w3*L1**2)/12, 'GB': (w3*L1**2)/12, 'EVol': -(w1*L2**2)/2 - P1*L2,
            'FVol': -(w2*L2**2)/2 - P2*L2, 'GVol': -(w3*L2**2)/2 - P3*L2
        }
        
        # TABLAS DE DISTRIBUCIÓN PASO A PASO
        st.subheader("Tablas de Iteración (Hardy Cross)")
        df_estado0, M0 = tabla_cross_paso_a_paso(nodos_fd, barras, FEM_0, 15, orden_miembros, "M. INICIAL")
        with st.expander("Ver Tabla Estado 0 (Cargas Externas)"): st.dataframe(df_estado0.style.format("{:.4f}"))
            
        FEM_1 = {'AB': -100, 'BA': -100, 'GH': -100, 'HG': -100}
        df_estado1, M1 = tabla_cross_paso_a_paso(nodosAquí tienes el código completamente corregido. He analizado todos los problemas que mencionaste y he aplicado las siguientes soluciones:

1. **Marcos de inicio restaurados:** He vuelto a codificar los esquemas visuales de referencia para **ambos ejercicios** al inicio de cada sección. Ahora muestran la geometría completa, los apoyos, las cargas distribuidas y las cotas para que identifiques qué es cada variable.
2. **Pasos y Tablas recuperados:** He vuelto a incluir la visualización en pantalla de los pasos intermedios (`st.dataframe(...)`). Ahora, al darle a resolver, verás las tablas iterativas de Hardy Cross (Estado 0, Estado 1, etc.) antes de los momentos finales.
3. **Corrección de los Gráficos de Diagramas:** El error por el cual los diagramas de cortante y momento no aparecían se debía a que los gráficos estaban dentro de un bloque `try/except` que fallaba silenciosamente al calcular las escalas si los valores iniciales por defecto daban cero, y a que Streamlit a veces corta el renderizado si no se cierra correctamente la figura. Lo he reestructurado para garantizar que se grafiquen con el estilo sombreado (achurado) que me pediste.

Reemplaza todo el contenido de tu archivo `app.py` con este código:

```python
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

st.set_page_config(page_title="Resolución Hardy Cross Completo", layout="wide")

st.title("Análisis Estructural - Método de Hardy Cross")
st.markdown("Resolución paso a paso con tablas iterativas y diagramas de Cortante y Momento.")

# --- CONFIGURACIÓN DE UNIDADES ---
st.sidebar.header("Configuración de Unidades")
u_fuerza = st.sidebar.selectbox("Unidad de Fuerza", ["ton", "N", "lb"])
u_long = st.sidebar.selectbox("Unidad de Longitud", ["m", "ft", "in"])
u_ang = st.sidebar.selectbox("Unidad de Ángulo", ["Grados", "Radianes"])

# ==========================================
# FUNCIONES DE DIBUJO DE DIAGRAMAS FINALES
# ==========================================
def draw_support(ax, x, y, nx=0, ny=-1):
    if nx == 0:  # Horizontal
        ax.plot([x-0.4, x+0.4], [y, y], 'k-', lw=2)
        for i in np.linspace(-0.4, 0.3, 6):
            ax.plot([x+i, x+i+0.1], [y, y-0.2], 'k-', lw=1)
    else: # Inclinado
        ax.plot























































































































                                     
