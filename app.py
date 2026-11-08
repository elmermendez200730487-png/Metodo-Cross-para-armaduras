import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

st.set_page_config(page_title="Hardy Cross - Marcos", layout="wide")

# ==========================================
# FUNCIONES DE DIBUJO
# ==========================================
def draw_support(ax, x, y, nx=0, ny=-1):
    if nx == 0:
        ax.plot([x-0.4, x+0.4], [y, y], 'k-', lw=2)
        for i in np.linspace(-0.4, 0.3, 6):
            ax.plot([x+i, x+i+0.1], [y, y-0.2], 'k-', lw=1)
    else:
        ax.plot([x-0.4*ny, x+0.4*ny], [y+0.4*nx, y-0.4*nx], 'k-', lw=2)
        for i in np.linspace(-0.4, 0.3, 6):
            xi, yi = x + i*ny, y - i*nx
            ax.plot([xi, xi+0.1*nx], [yi, yi+0.1*ny], 'k-', lw=1)

def draw_distributed_load(ax, x1, x2, y, label):
    ax.plot([x1, x2], [y+0.5, y+0.5], 'r-', lw=1)
    for px in np.linspace(x1, x2, int((x2-x1)*3)+2):
        ax.annotate('', xy=(px, y), xytext=(px, y+0.5),
                    arrowprops=dict(arrowstyle='->', color='red', lw=1))
    ax.text((x1+x2)/2, y+0.7, label, color='red', ha='center', weight='bold')

def draw_diagram_member(ax, x1, y1, x2, y2, v1, v2, scale, color='blue', label1="", label2="", draw_member=True):
    dx, dy = x2 - x1, y2 - y1
    L = math.hypot(dx, dy)
    if L == 0: return
    nx, ny = -dy/L, dx/L
    
    px1, py1 = x1 + v1 * scale * nx, y1 + v1 * scale * ny
    px2, py2 = x2 + v2 * scale * nx, y2 + v2 * scale * ny
    
    if draw_member:
        ax.plot([x1, x2], [y1, y2], 'k-', lw=2.5)
        
    ax.plot([x1, px1, px2, x2], [y1, py1, py2, y2], color=color, lw=1.2)
    
    num_hatches = max(5, int(L * 8))
    for i in range(1, num_hatches):
        t = i / num_hatches
        bx, by = x1 + t*dx, y1 + t*dy
        vt = v1 + t*(v2 - v1)
        tx, ty = bx + vt * scale * nx, by + vt * scale * ny
        ax.plot([bx, tx], [by, ty], color=color, lw=0.6, alpha=0.6)
        
    if abs(v1) > 0.001: ax.text(px1 + 0.3*nx, py1 + 0.3*ny, label1, color='black', fontsize=9, weight='bold', ha='center', va='center')
    if abs(v2) > 0.001: ax.text(px2 + 0.3*nx, py2 + 0.3*ny, label2, color='black', fontsize=9, weight='bold', ha='center', va='center')

# ==========================================
# FUNCIÓN GENERADORA DE TABLAS CROSS
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
    
    return pd.DataFrame(filas, index=nombres_filas, columns=columnas), M_actual

st.title("Análisis Estructural - Método de Hardy Cross")

ejercicio = st.sidebar.radio("Selecciona el marco:", ("Ejercicio 1: 3 Niveles", "Ejercicio 2: Columna Inclinada"))

if ejercicio == "Ejercicio 1: 3 Niveles":
    st.header("Marco con Tres Niveles y Una Crujía")
    
    # DIBUJO DEL ESQUEMA INICIAL EJERCICIO 1
    fig_ref1, ax_ref1 = plt.subplots(figsize=(8, 7))
    # Marco principal y voladizos
    ax_ref1.plot([0, 0], [0, 9], 'b-', lw=2); ax_ref1.plot([5, 5], [0, 9], 'b-', lw=2)
    ax_ref1.plot([0, 7], [3, 3], 'b-', lw=2); ax_ref1.plot([0, 7], [6, 6], 'b-', lw=2); ax_ref1.plot([0, 7], [9, 9], 'b-', lw=2)
    draw_support(ax_ref1, 0, 0); draw_support(ax_ref1, 5, 0)
    
    # Textos de Nodos e Inercias
    ax_ref1.text(-0.3, 0, 'A', weight='bold'); ax_ref1.text(5.2, 0, 'H', weight='bold')
    ax_ref1.text(-0.3, 3, 'B', weight='bold'); ax_ref1.text(-0.3, 6, 'C', weight='bold'); ax_ref1.text(-0.3, 9, 'D', weight='bold')
    ax_ref1.text(4.7, 3, 'G', weight='bold'); ax_ref1.text(4.7, 6, 'F', weight='bold'); ax_ref1.text(4.7, 9, 'E', weight='bold')
    
    ax_ref1.text(0.2, 1.5, 'aI', color='teal', weight='bold'); ax_ref1.text(5.2, 1.5, 'aI', color='teal', weight='bold')
    ax_ref1.text(0.2, 4.5, 'bI', color='teal', weight='bold'); ax_ref1.text(5.2, 4.5, 'bI', color='teal', weight='bold')
    ax_ref1.text(0.2, 7.5, 'cI', color='teal', weight='bold'); ax_ref1.text(5.2, 7.5, 'cI', color='teal', weight='bold')
    ax_ref1.text(2.5, 2.7, 'dI', color='teal', weight='bold'); ax_ref1.text(2.5, 5.7, 'dI', color='teal', weight='bold'); ax_ref1.text(2.5, 8.7, 'dI', color='teal', weight='bold')

    # Cargas Distribuidas y Puntuales
    draw_distributed_load(ax_ref1, 0, 7, 9, '$w_1$'); draw_distributed_load(ax_ref1, 0, 7, 6, '$w_2$'); draw_distributed_load(ax_ref1, 0, 7, 3, '$w_3$')
    ax_ref1.annotate('$P_1$', xy=(7, 9), xytext=(7, 10), arrowprops=dict(facecolor='red', width=1, headwidth=5), color='red', ha='center')
    ax_ref1.annotate('$P_2$', xy=(7, 6), xytext=(7, 7), arrowprops=dict(facecolor='red', width=1, headwidth=5), color='red', ha='center')
    ax_ref1.annotate('$P_3$', xy=(7, 3), xytext=(7, 4), arrowprops=dict(facecolor='red', width=1, headwidth=5), color='red', ha='center')
    
    # Fuerzas Laterales
    ax_ref1.annotate('$F_1$', xy=(0, 9), xytext=(-2, 9), arrowprops=dict(facecolor='red', width=1, headwidth=5), color='red', va='center')
    ax_ref1.annotate('$F_2$', xy=(0, 6), xytext=(-2, 6), arrowprops=dict(facecolor='red', width=1, headwidth=5), color='red', va='center')
    ax_ref1.annotate('$F_3$', xy=(0, 3), xytext=(-2, 3), arrowprops=dict(facecolor='red', width=1, headwidth=5), color='red', va='center')
    
    # Cotas
    ax_ref1.text(7.5, 1.5, '$L_3$', va='center'); ax_ref1.text(7.5, 4.5, '$L_4$', va='center'); ax_ref1.text(7.5, 7.5, '$L_5$', va='center')
    ax_ref1.text(2.5, -1, '$L_1$', ha='center'); ax_ref1.text(6, -1, '$L_2$', ha='center')
    
    ax_ref1.set_xlim(-3, 9); ax_ref1.set_ylim(-2, 11); ax_ref1.axis('off')
    st.pyplot(fig_ref1)
    
    # Parámetros (sin Inercia explícita como magnitud, usando los coeficientes)
    col1, col2, col3 = st.columns(3)
    with col1:
        L1 = st.number_input("L1 (m)", value=5.0); L2 = st.number_input("L2 (m)", value=1.5)
        L3 = st.number_input("L3 (m)", value=4.0); L4 = st.number_input("L4 (m)", value=3.0); L5 = st.number_input("L5 (m)", value=3.0)
    with col2:
        w1 = st.number_input("w1 (Ton/m)", value=2.0); P1 = st.number_input("P1 (Ton)", value=2.0); F1 = st.number_input("F1 (Ton)", value=8.0)
        w2 = st.number_input("w2 (Ton/m)", value=3.0); P2 = st.number_input("P2 (Ton)", value=2.0); F2 = st.number_input("F2 (Ton)", value=6.0)
        w3 = st.number_input("w3 (Ton/m)", value=3.0); P3 = st.number_input("P3 (Ton)", value=2.0); F3 = st.number_input("F3 (Ton)", value=4.0)
    with col3:
        a_coef = st.number_input("Coeficiente Inercia a", value=4.0)
        b_coef = st.number_input("Coeficiente Inercia b", value=3.0)
        c_coef = st.number_input("Coeficiente Inercia c", value=2.0)
        d_coef = st.number_input("Coeficiente Inercia d", value=1.0)

    if st.button("Resolver Ejercicio 1"):
        # Cálculo de rigideces proporcionales
        K_AB = a_coef / L3; K_GH = K_AB
        K_BC = b_coef / L4; K_FG = K_BC
        K_CD = c_coef / L5; K_EF = K_CD
        K_V = d_coef / L1
        
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
        
        st.subheader("Desglose de Tablas de Iteración (Hardy Cross)")
        df_estado0, M0 = tabla_cross_paso_a_paso(nodos_fd, barras, FEM_0, 10, orden_miembros, "M. INICIAL")
        st.markdown("**Estado 0 (Cargas Reales)**")
        st.dataframe(df_estado0.style.format("{:.3f}"))
        
        FEM_1 = {'AB': -100, 'BA': -100, 'GH': -100, 'HG': -100}
        df_estado1, M1 = tabla_cross_paso_a_paso(nodos_fd, barras, FEM_1, 10, orden_miembros, "M. IMPUESTO (\u03941)")
        st.markdown("**Estado 1 (Desplazamiento Nivel 1)**")
        st.dataframe(df_estado1.style.format("{:.3f}"))
        
        FEM_2 = {'BC': -100, 'CB': -100, 'FG': -100, 'GF': -100}
        df_estado2, M2 = tabla_cross_paso_a_paso(nodos_fd, barras, FEM_2, 10, orden_miembros, "M. IMPUESTO (\u03942)")
        st.markdown("**Estado 2 (Desplazamiento Nivel 2)**")
        st.dataframe(df_estado2.style.format("{:.3f}"))
        
        FEM_3 = {'CD': -100, 'DC': -100, 'EF': -100, 'FE': -100}
        df_estado3, M3 = tabla_cross_paso_a_paso(nodos_fd, barras, FEM_3, 10, orden_miembros, "M. IMPUESTO (\u03943)")
        st.markdown("**Estado 3 (Desplazamiento Nivel 3)**")
        st.dataframe(df_estado3.style.format("{:.3f}"))
        
        def calc_cortantes(M_dict):
            V1 = -(M_dict.get('AB',0) + M_dict.get('BA',0) + M_dict.get('GH',0) + M_dict.get('HG',0)) / L3
            V2 = -(M_dict.get('BC',0) + M_dict.get('CB',0) + M_dict.get('FG',0) + M_dict.get('GF',0)) / L4
            V3 = -(M_dict.get('CD',0) + M_dict.get('DC',0) + M_dict.get('EF',0) + M_dict.get('FE',0)) / L5
            return np.array([V1, V2, V3])

        V0, V1, V2, V3 = calc_cortantes(M0), calc_cortantes(M1), calc_cortantes(M2), calc_cortantes(M3)
        Matriz_V = np.column_stack((V1, V2, V3))
        Vector_F = np.array([F1 + F2 + F3, F1 + F2, F1]) - V0
        
        X = np.linalg.solve(Matriz_V, Vector_F)
        M_final = {k: M0[k] + X[0]*M1[k] + X[1]*M2[k] + X[2]*M3[k] for k in orden_miembros}
        
        st.subheader("Momentos Finales")
        df_final = pd.DataFrame([M_final], index=['Mom. Final'])
        df_final.columns = pd.MultiIndex.from_tuples([(m[0], m) for m in orden_miembros], names=['NODO', 'MIEMBRO'])
        st.dataframe(df_final.style.format("{:.4f}"))
        
        st.subheader("Diagramas de Cortante y Momento")
        fig_res, (ax_v, ax_m) = plt.subplots(1, 2, figsize=(16, 8))
        coords = {'A':(0,0), 'B':(0,L3), 'C':(0,L3+L4), 'D':(0,L3+L4+L5), 'H':(L1,0), 'G':(L1,L3), 'F':(L1,L3+L4), 'E':(L1,L3+L4+L5)}
        
        # Factor de escala visual
        max_m = max([abs(v) for v in M_final.values()]) or 1.0
        scale_m = (L1 * 0.15) / max_m
        
        ax_m.set_title("Diagrama de Momento Flexionante")
        draw_support(ax_m, coords['A'][0], coords['A'][1]); draw_support(ax_m, coords['H'][0], coords['H'][1])
        tramos_m = [('A','B'), ('B','C'), ('C','D'), ('H','G'), ('G','F'), ('F','E'), ('B','G'), ('C','F'), ('D','E')]
        for n1, n2 in tramos_m:
            v1, v2 = M_final.get(n1+n2, 0), M_final.get(n2+n1, 0)
            draw_diagram_member(ax_m, coords[n1][0], coords[n1][1], coords[n2][0], coords[n2][1], v1, -v2, scale_m, color='purple', label1=f"{v1:.2f}", label2=f"{v2:.2f}")

        ax_v.set_title("Diagrama de Fuerza Cortante")
        draw_support(ax_v, coords['A'][0], coords['A'][1]); draw_support(ax_v, coords['H'][0], coords['H'][1])
        V_vals = {}
        for n1, n2, L, w in [('A','B',L3,0), ('B','C',L4,0), ('C','D',L5,0), ('H','G',L3,0), ('G','F',L4,0), ('F','E',L5,0), ('B','G',L1,w3), ('C','F',L1,w2), ('D','E',L1,w1)]:
            v_izq = w*L/2.0 - (M_final.get(n1+n2,0) + M_final.get(n2+n1,0))/L
            v_der = -w*L/2.0 - (M_final.get(n1+n2,0) + M_final.get(n2+n1,0))/L
            V_vals[n1+n2] = (v_izq, v_der)
        
        max_v = max([abs(v[0]) for v in V_vals.values()] + [abs(v[1]) for v in V_vals.values()]) or 1.0
        scale_v = (L1 * 0.15) / max_v
        
        for (n1, n2), (v_izq, v_der) in V_vals.items():
            draw_diagram_member(ax_v, coords[n1][0], coords[n1][1], coords[n2][0], coords[n2][1], v_izq, v_der, scale_v, color='green', label1=f"{v_izq:.2f}", label2=f"{abs(v_der):.2f}")
            
        for ax in (ax_v, ax_m): 
            ax.set_xlim(-2, L1+3); ax.set_ylim(-1, L3+L4+L5+2)
            ax.axis('off')
        st.pyplot(fig_res)


elif ejercicio == "Ejercicio 2: Columna Inclinada":
    st.header("Marco con una Columna Inclinada")
    
    # DIBUJO DEL ESQUEMA INICIAL EJERCICIO 2
    fig_ref2, ax_ref2 = plt.subplots(figsize=(8, 6))
    ax_ref2.plot([0, 0], [0, 4], 'b-', lw=2); ax_ref2.plot([0, 5], [4, 4], 'b-', lw=2); ax_ref2.plot([5, 8], [4, 0], 'b-', lw=2)
    draw_support(ax_ref2, 0, 0); draw_support(ax_ref2, 8, 0)
    
    ax_ref2.text(-0.3, 0, 'A', weight='bold'); ax_ref2.text(-0.3, 4, 'B', weight='bold')
    ax_ref2.text(5.2, 4, 'C', weight='bold'); ax_ref2.text(8.3, 0, 'D', weight='bold')
    
    ax_ref2.text(0.2, 2, 'I', color='teal', weight='bold'); ax_ref2.text(2.5, 3.7, 'I', color='teal', weight='bold'); ax_ref2.text(6.7, 2.2, 'I', color='teal', weight='bold')
    
    draw_distributed_load(ax_ref2, 0, 5, 4, '$w$')
    ax_ref2.annotate('$F$', xy=(0, 4), xytext=(-2, 4), arrowprops=dict(facecolor='red', width=1, headwidth=5), color='red', va='center')
    
    # Cotas y ángulos
    ax_ref2.text(-1, 2, '$a$', va='center', color='black'); ax_ref2.text(2.5, 5.2, '$b$', ha='center', color='black'); ax_ref2.text(6.5, 5.2, '$c$', ha='center', color='black')
    ax_ref2.plot([5, 5], [4, 5.5], 'k--', lw=0.5); ax_ref2.plot([8, 8], [0, 5.5], 'k--', lw=0.5)
    
    # Angulo Theta
    ax_ref2.plot([7, 9], [0, 0], 'k-', lw=1)
    theta_arc = np.linspace(np.pi - np.arctan(4/3), np.pi, 20)
    ax_ref2.plot(8 + 0.8*np.cos(theta_arc), 0.8*np.sin(theta_arc), 'r-', lw=1.5)
    ax_ref2.text(7.3, 0.4, r'$\theta$', color='red', weight='bold')
    
    ax_ref2.set_xlim(-3, 10); ax_ref2.set_ylim(-1, 6); ax_ref2.axis('off')
    st.pyplot(fig_ref2)
    
    # Parámetros (Inercia como coeficiente común I=1 implícito en los cálculos)
    col1, col2 = st.columns(2)
    with col1:
        a = st.number_input("Altura columna a (m)", value=3.0)
        b = st.number_input("Longitud viga b (m)", value=3.5)
        c = st.number_input("Proyección columna c (m)", value=2.0)
    with col2:
        w = st.number_input("Carga distribuida w (Ton/m)", value=2.0)
        F = st.number_input("Fuerza F (Ton)", value=3.0)

    if st.button("Resolver Ejercicio 2"):
        L_CD = math.hypot(c, a)
        K_AB = 1.0 / a; K_BC = 1.0 / b; K_CD = 1.0 / L_CD
        
        FD_BA = K_AB/(K_AB+K_BC); FD_BC = K_BC/(K_AB+K_BC)
        FD_CB = K_BC/(K_BC+K_CD); FD_CD = K_CD/(K_BC+K_CD)
        
        nodos_fd = {'A': {'B': 0.0}, 'D': {'C': 0.0}, 'B': {'A': FD_BA, 'C': FD_BC}, 'C': {'B': FD_CB, 'D': FD_CD}}
        barras = [('A','B'), ('B','C'), ('C','D')]; orden_miembros = ['AB', 'BA', 'BC', 'CB', 'CD', 'DC']
        
        FEM_0 = {'BC': -(w * b**2)/12, 'CB': (w * b**2)/12}
        
        st.subheader("Desglose de Tablas de Iteración (Hardy Cross)")
        df_estado0, M0 = tabla_cross_paso_a_paso(nodos_fd, barras, FEM_0, 6, orden_miembros, "M. INICIAL")
        st.markdown("**Estado 0 (Cargas Reales)**")
        st.dataframe(df_estado0.style.format("{:.3f}"))
        
        M_AB_imp = -100.0
        M_CD_imp = (a**2 * M_AB_imp) / (L_CD**2)
        FEM_1 = {'AB': M_AB_imp, 'BA': M_AB_imp, 'CD': M_CD_imp, 'DC': M_CD_imp}
        df_estado1, M1 = tabla_cross_paso_a_paso(nodos_fd, barras, FEM_1, 6, orden_miembros, "M. IMPUESTO")
        st.markdown("**Estado 1 (Desplazamiento Lateral)**")
        st.dataframe(df_estado1.style.format("{:.3f}"))
        
        V0 = -(M0.get('AB',0) + M0.get('BA',0))/a - (M0.get('CD',0) + M0.get('DC',0))/a
        V1 = -(M1.get('AB',0) + M1.get('BA',0))/a - (M1.get('CD',0) + M1.get('DC',0))/a
        
        if V1 != 0:
            X = (F - V0) / V1
            M_final = {k: M0[k] + X*M1[k] for k in orden_miembros}
            
            st.subheader("Momentos Finales")
            df_final = pd.DataFrame([M_final], index=['Mom. Final'])
            df_final.columns = pd.MultiIndex.from_tuples([(m[0], m) for m in orden_miembros], names=['NODO', 'MIEMBRO'])
            st.dataframe(df_final.style.format("{:.4f}"))
            
            st.subheader("Diagramas de Cortante y Momento")
            fig_res2, (ax_v, ax_m) = plt.subplots(1, 2, figsize=(14, 6))
            coords = {'A':(0,0), 'B':(0,a), 'C':(b,a), 'D':(b+c,0)}
            
            max_m = max([abs(v) for v in M_final.values()]) or 1.0
            scale_m = (b * 0.15) / max_m
            
            ax_m.set_title("Diagrama de Momento Flexionante")
            draw_support(ax_m, coords['A'][0], coords['A'][1]); draw_support(ax_m, coords['D'][0], coords['D'][1])
            for n1, n2 in [('A','B'), ('B','C'), ('C','D')]:
                v1, v2 = M_final.get(n1+n2, 0), M_final.get(n2+n1, 0)
                draw_diagram_member(ax_m, coords[n1][0], coords[n1][1], coords[n2][0], coords[n2][1], v1, -v2, scale_m, color='purple', label1=f"{v1:.3f}", label2=f"{v2:.3f}")
            
            ax_v.set_title("Diagrama de Fuerza Cortante")
            draw_support(ax_v, coords['A'][0], coords['A'][1]); draw_support(ax_v, coords['D'][0], coords['D'][1])
            V_vals = {}
            for n1, n2, L, load in [('A','B',a,0), ('B','C',b,w), ('C','D',L_CD,0)]:
                v_izq = load*L/2.0 - (M_final.get(n1+n2,0) + M_final.get(n2+n1,0))/L
                v_der = -load*L/2.0 - (M_final.get(n1+n2,0) + M_final.get(n2+n1,0))/L
                V_vals[n1+n2] = (v_izq, v_der)
            
            max_v = max([abs(v[0]) for v in V_vals.values()] + [abs(v[1]) for v in V_vals.values()]) or 1.0
            scale_v = (b * 0.15) / max_v
            
            for (n1, n2), (v_izq, v_der) in V_vals.items():
                draw_diagram_member(ax_v, coords[n1][0], coords[n1][1], coords[n2][0], coords[n2][1], v_izq, v_der, scale_v, color='green', label1=f"{v_izq:.3f}", label2=f"{abs(v_der):.3f}")
            
            for ax in (ax_v, ax_m): 
                ax.set_xlim(-2, b+c+3); ax.set_ylim(-1, a+2)
                ax.axis('off')
            st.pyplot(fig_res2)



















































































                                     
