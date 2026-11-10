import sys
sys.setrecursionlimit(5000)

import matplotlib
matplotlib.use('Agg')

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

st.set_page_config(page_title="Hardy Cross - Marcos", layout="wide")

# Resetear la memoria si cambian los parámetros estructurales principales
def reset_state():
    st.session_state.resuelto = False

if 'resuelto' not in st.session_state:
    st.session_state.resuelto = False

# ==========================================
# FUNCIONES DE DIBUJO DE DIAGRAMAS
# ==========================================
def draw_support(ax, x, y, nx=0, ny=-1):
    hx, hy = [], []
    ax.plot([x-0.4, x+0.4], [y, y], 'k-', lw=2)
    for i in np.linspace(-0.4, 0.3, 6):
        hx.extend([x+i, x+i+0.1, np.nan])
        hy.extend([y, y-0.2, np.nan])
    if hx: ax.plot(hx, hy, 'k-', lw=1)

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
    ax.fill_between([x1, x2], [y1, y2], [py1, py2], color=color, alpha=0.2)
        
    if abs(v1) > 0.001: ax.text(px1 + 0.3*nx, py1 + 0.3*ny, label1, color='black', fontsize=9, weight='bold', ha='center', va='center')
    if abs(v2) > 0.001: ax.text(px2 + 0.3*nx, py2 + 0.3*ny, label2, color='black', fontsize=9, weight='bold', ha='center', va='center')

# ==========================================
# FUNCIÓN GENERADORA DE TABLAS CROSS
# ==========================================
def tabla_cross_paso_a_paso(nodos_fd, barras, FEM, iteraciones, orden_miembros, nombre_m_inicial):
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

st.title("Análisis Estructural - Método de Hardy Cross Dinámico")

# ==========================================
# CONFIGURACIÓN LATERAL
# ==========================================
st.sidebar.header("Configuración de la Estructura")
u_fuerza = st.sidebar.selectbox("Unidad de Fuerza", ["ton", "N", "lb"])
u_long = st.sidebar.selectbox("Unidad de Longitud", ["m", "ft", "in"])

num_niveles = st.sidebar.radio("Número de niveles:", (1, 2, 3), index=2, on_change=reset_state)
tiene_voladizos = st.sidebar.checkbox("Incluir voladizos a la derecha", value=True, on_change=reset_state)

# ==========================================
# ESQUEMA DINÁMICO
# ==========================================
st.header(f"Marco de {num_niveles} Nivel(es) {'con' if tiene_voladizos else 'sin'} Voladizos")

fig_ref, ax_ref = plt.subplots(figsize=(8, 7))
ax_ref.plot([0, 0], [0, num_niveles*3], 'b-', lw=2)  # Columna Izquierda
ax_ref.plot([5, 5], [0, num_niveles*3], 'b-', lw=2)  # Columna Derecha

draw_support(ax_ref, 0, 0); draw_support(ax_ref, 5, 0)
ax_ref.text(-0.3, 0, 'A', weight='bold'); ax_ref.text(5.2, 0, 'H', weight='bold')

for i in range(1, num_niveles + 1):
    y = i * 3
    ax_ref.plot([0, 5], [y, y], 'b-', lw=2) # Vigas principales
    if tiene_voladizos:
        ax_ref.plot([5, 7], [y, y], 'b-', lw=2) # Voladizos
        
    # Nombres de nodos según nivel
    if i == 1:
        ax_ref.text(-0.3, y, 'B', weight='bold'); ax_ref.text(4.7, y, 'G', weight='bold')
    elif i == 2:
        ax_ref.text(-0.3, y, 'C', weight='bold'); ax_ref.text(4.7, y, 'F', weight='bold')
    elif i == 3:
        ax_ref.text(-0.3, y, 'D', weight='bold'); ax_ref.text(4.7, y, 'E', weight='bold')

ax_ref.set_xlim(-2, 9); ax_ref.set_ylim(-1.5, num_niveles*3 + 2.5); ax_ref.axis('off')
st.pyplot(fig_ref, clear_figure=True)
plt.close(fig_ref)

# ==========================================
# ENTRADA DE DATOS (SOLO ACTIVOS)
# ==========================================
st.subheader("Geometría e Inercia")
colG1, colG2, colG3 = st.columns(3)
with colG1:
    L_viga = st.number_input("Long. Viga Central (m)", value=5.0, on_change=reset_state)
    coef_viga = st.number_input("Inercia Vigas (I)", value=1.0, on_change=reset_state)
with colG2:
    if tiene_voladizos:
        L_vol = st.number_input("Long. Voladizo (m)", value=1.5, on_change=reset_state)
    else:
        L_vol = 0.0
with colG3:
    H_niveles = []; coef_cols = []
    for i in range(1, num_niveles + 1):
        H_niveles.append(st.number_input(f"Altura Nivel {i} (m)", value=3.0, key=f"H{i}", on_change=reset_state))
        coef_cols.append(st.number_input(f"Inercia Columna Nivel {i} (I)", value=4.0 if i==1 else (3.0 if i==2 else 2.0), key=f"I{i}", on_change=reset_state))

st.subheader("Cargas (0 para anular)")
cargas_W = []; cargas_P = []; cargas_F = []
for i in range(1, num_niveles + 1):
    st.markdown(f"**Nivel {i}:**")
    colC1, colC2, colC3 = st.columns(3)
    with colC1: cargas_W.append(st.number_input(f"Distribuida (w{i})", value=2.0, key=f"w{i}", on_change=reset_state))
    with colC2: cargas_P.append(st.number_input(f"Puntual en extremo voladizo (P{i})", value=2.0 if tiene_voladizos else 0.0, disabled=not tiene_voladizos, key=f"p{i}", on_change=reset_state))
    with colC3: cargas_F.append(st.number_input(f"Fuerza Horizontal (F{i})", value=8.0 if i==3 else (6.0 if i==2 else 4.0), key=f"f{i}", on_change=reset_state))

if st.button("Resolver Estructura"):
    st.session_state.resuelto = True

if st.session_state.resuelto:
    st.write("---")
    # ==========================================
    # LÓGICA DINÁMICA HARDY CROSS
    # ==========================================
    # 1. Definir miembros activos y Rigideces
    K_cols = [coef_cols[i] / H_niveles[i] for i in range(num_niveles)]
    K_viga = coef_viga / L_viga
    
    nodos_activos = ['A', 'H', 'B', 'G']
    barras_activas = [('A','B'), ('B','G'), ('G','H')]
    orden_miembros = ['AB', 'BA', 'BG', 'GB', 'GH', 'HG']
    if tiene_voladizos: orden_miembros.insert(4, 'GVol')
    
    if num_niveles >= 2:
        nodos_activos.extend(['C', 'F'])
        barras_activas.extend([('B','C'), ('C','F'), ('F','G')])
        orden_miembros.extend(['BC', 'CB', 'CF', 'FC', 'FG', 'GF'])
        if tiene_voladizos: orden_miembros.insert(orden_miembros.index('FC')+1, 'FVol')
        
    if num_niveles == 3:
        nodos_activos.extend(['D', 'E'])
        barras_activas.extend([('C','D'), ('D','E'), ('E','F')])
        orden_miembros.extend(['CD', 'DC', 'DE', 'ED', 'EF', 'FE'])
        if tiene_voladizos: orden_miembros.insert(orden_miembros.index('ED')+1, 'EVol')

    nodos_fd = {n: {} for n in nodos_activos}
    nodos_fd['A']['B'] = 0; nodos_fd['H']['G'] = 0
    
    # Asignar K a los nodos
    S_nodos = {n: 0.0 for n in nodos_activos if n not in ['A','H']}
    
    # Nivel 1 (B, G)
    nodos_fd['B']['A'] = K_cols[0]; nodos_fd['B']['G'] = K_viga; S_nodos['B'] += K_cols[0] + K_viga
    nodos_fd['G']['H'] = K_cols[0]; nodos_fd['G']['B'] = K_viga; S_nodos['G'] += K_cols[0] + K_viga
    if num_niveles >= 2:
        nodos_fd['B']['C'] = K_cols[1]; S_nodos['B'] += K_cols[1]
        nodos_fd['G']['F'] = K_cols[1]; S_nodos['G'] += K_cols[1]
    
    # Nivel 2 (C, F)
    if num_niveles >= 2:
        nodos_fd['C']['B'] = K_cols[1]; nodos_fd['C']['F'] = K_viga; S_nodos['C'] += K_cols[1] + K_viga
        nodos_fd['F']['G'] = K_cols[1]; nodos_fd['F']['C'] = K_viga; S_nodos['F'] += K_cols[1] + K_viga
        if num_niveles == 3:
            nodos_fd['C']['D'] = K_cols[2]; S_nodos['C'] += K_cols[2]
            nodos_fd['F']['E'] = K_cols[2]; S_nodos['F'] += K_cols[2]
            
    # Nivel 3 (D, E)
    if num_niveles == 3:
        nodos_fd['D']['C'] = K_cols[2]; nodos_fd['D']['E'] = K_viga; S_nodos['D'] += K_cols[2] + K_viga
        nodos_fd['E']['F'] = K_cols[2]; nodos_fd['E']['D'] = K_viga; S_nodos['E'] += K_cols[2] + K_viga

    # Calcular Factores de Distribución (FD)
    for n in S_nodos:
        for adj in nodos_fd[n]:
            nodos_fd[n][adj] = nodos_fd[n][adj] / S_nodos[n]
        if tiene_voladizos:
            nodos_fd[n]['Vol'] = 0.0

    # 2. Momentos de Empotramiento (FEM_0)
    FEM_0 = {}
    vigas_por_nivel = {1: ('B','G'), 2: ('C','F'), 3: ('D','E')}
    for i in range(1, num_niveles + 1):
        n1, n2 = vigas_por_nivel[i]
        w = cargas_W[i-1]
        FEM_0[f"{n1}{n2}"] = -(w * L_viga**2)/12
        FEM_0[f"{n2}{n1}"] = (w * L_viga**2)/12
        if tiene_voladizos:
            P = cargas_P[i-1]
            FEM_0[f"{n2}Vol"] = -(w * L_vol**2)/2 - P*L_vol

    # Estados de Ladeo
    FEM_sway = []
    cols_por_nivel = {1: ('AB','BA','GH','HG'), 2: ('BC','CB','FG','GF'), 3: ('CD','DC','EF','FE')}
    for i in range(1, num_niveles + 1):
        f_sway = {}
        for c in cols_por_nivel[i]:
            f_sway[c] = -100.0
        FEM_sway.append(f_sway)

    st.markdown("### 1. Desglose de Tablas (Cargas Reales y Ladeos)")
    df_0, M0 = tabla_cross_paso_a_paso(nodos_fd, barras_activas, FEM_0, 10, orden_miembros, "M. INICIAL")
    with st.expander("Tabla Estado 0 (Cargas Reales)"):
        st.dataframe(df_0.round(3))
        
    M_sways = []
    for i in range(num_niveles):
        df_sway, m_sway = tabla_cross_paso_a_paso(nodos_fd, barras_activas, FEM_sway[i], 10, orden_miembros, f"M. Ladeo Nivel {i+1}")
        M_sways.append(m_sway)
        with st.expander(f"Tabla Estado Ladeo Nivel {i+1}"):
            st.dataframe(df_sway.round(3))

    st.markdown("### 2. Ecuaciones de Cortante y Factores de Corrección")
    def calc_cortante_nivel(M_dict, nivel):
        c = cols_por_nivel[nivel]
        return -(M_dict.get(c[0],0) + M_dict.get(c[1],0) + M_dict.get(c[2],0) + M_dict.get(c[3],0)) / H_niveles[nivel-1]

    # Construir Matriz V y Vector F
    V_matrix = np.zeros((num_niveles, num_niveles))
    F_vector = np.zeros(num_niveles)
    
    for fila in range(num_niveles): # Fila i representa el corte en el nivel i+1
        corte_idx = fila + 1
        # Fuerzas reales actuantes en el corte
        f_actuante = sum(cargas_F[corte_idx-1:])
        # Cortante del Estado 0 en este nivel
        v0 = calc_cortante_nivel(M0, corte_idx)
        F_vector[fila] = f_actuante - v0
        
        # Llenar matriz con los cortantes generados por cada ladeo
        for col in range(num_niveles):
            V_matrix[fila, col] = calc_cortante_nivel(M_sways[col], corte_idx)
            
    X_correcciones = np.linalg.solve(V_matrix, F_vector)
    
    st.write("Factores de corrección calculados:")
    for i, x in enumerate(X_correcciones):
        st.latex(rf"X_{i+1} = {x:.4f}")

    # Momentos Finales
    M_final = {k: M0.get(k,0) for k in orden_miembros}
    for i in range(num_niveles):
        for k in orden_miembros:
            M_final[k] += X_correcciones[i] * M_sways[i].get(k, 0)

    st.markdown("### 3. Momentos Finales")
    df_final = pd.DataFrame([M_final], index=['Mom. Final'])
    df_final.columns = pd.MultiIndex.from_tuples([(m[0], m) for m in orden_miembros], names=['NODO', 'MIEMBRO'])
    st.dataframe(df_final.astype(float).round(4))

    st.markdown("### 4. Reacciones en la Base")
    R_Ax = -(M_final.get('AB',0) + M_final.get('BA',0)) / H_niveles[0]
    R_Hx = -(M_final.get('GH',0) + M_final.get('HG',0)) / H_niveles[0]
    colr1, colr2 = st.columns(2)
    colr1.latex(rf"R_{{Ax}} = {R_Ax:.4f} \text{{ {u_fuerza}}}")
    colr2.latex(rf"R_{{Hx}} = {R_Hx:.4f} \text{{ {u_fuerza}}}")

    # ==========================================
    # DIAGRAMAS FINALES
    # ==========================================
    st.markdown("### 5. Diagramas de Cortante y Momento")
    fig_res, (ax_v, ax_m) = plt.subplots(1, 2, figsize=(16, max(6, num_niveles*3)))
    
    # Coordenadas
    coords = {'A':(0,0), 'H':(L_viga, 0)}
    y_acum = 0
    for i in range(1, num_niveles + 1):
        y_acum += H_niveles[i-1]
        nI, nD = vigas_por_nivel[i]
        coords[nI] = (0, y_acum)
        coords[nD] = (L_viga, y_acum)
        if tiene_voladizos:
            coords[f"{nD}Vol"] = (L_viga + L_vol, y_acum)

    # Escala para dibujos
    max_m = max([abs(v) for v in M_final.values() if not np.isnan(v)], default=0.0)
    scale_m = (L_viga * 0.15) / max_m if max_m > 1e-4 else 1.0

    ax_m.set_title("Diagrama de Momento Flexionante")
    draw_support(ax_m, coords['A'][0], coords['A'][1]); draw_support(ax_m, coords['H'][0], coords['H'][1])
    
    # Dibujar momentos en columnas y vigas
    for c in barras_activas:
        v1, v2 = M_final.get(c[0]+c[1], 0), M_final.get(c[1]+c[0], 0)
        draw_diagram_member(ax_m, coords[c[0]][0], coords[c[0]][1], coords[c[1]][0], coords[c[1]][1], v1, -v2, scale_m, color='purple', label1=f"{v1:.2f}", label2=f"{v2:.2f}")
    
    # Dibujar momentos en voladizos
    if tiene_voladizos:
        for i in range(1, num_niveles + 1):
            nD = vigas_por_nivel[i][1]
            v1 = M_final.get(f"{nD}Vol", 0)
            draw_diagram_member(ax_m, coords[nD][0], coords[nD][1], coords[f"{nD}Vol"][0], coords[f"{nD}Vol"][1], v1, 0, scale_m, color='purple', label1=f"{v1:.2f}")

    ax_v.set_title("Diagrama de Fuerza Cortante")
    draw_support(ax_v, coords['A'][0], coords['A'][1]); draw_support(ax_v, coords['H'][0], coords['H'][1])
    
    # Cortantes de vigas (incluyendo distribuidas)
    V_vals = {}
    for i in range(1, num_niveles + 1):
        nI, nD = vigas_por_nivel[i]
        w = cargas_W[i-1]
        v_izq = w*L_viga/2.0 - (M_final.get(nI+nD,0) + M_final.get(nD+nI,0))/L_viga
        v_der = -w*L_viga/2.0 - (M_final.get(nI+nD,0) + M_final.get(nD+nI,0))/L_viga
        V_vals[nI+nD] = (v_izq, v_der)
        
        if tiene_voladizos:
            P = cargas_P[i-1]
            V_vals[f"{nD}Vol"] = (w*L_vol + P, P)

    # Cortantes de Columnas
    for i in range(1, num_niveles + 1):
        h = H_niveles[i-1]
        c = cols_por_nivel[i]
        for col in [c[:2], c[2:]]: # (AB, BA) y (GH, HG)
            v_izq = -(M_final.get(col[0],0) + M_final.get(col[1],0))/h
            V_vals[col[0]] = (v_izq, v_izq)

    max_v = max([abs(v[0]) for v in V_vals.values()] + [abs(v[1]) for v in V_vals.values()], default=0.0)
    scale_v = (L_viga * 0.15) / max_v if max_v > 1e-4 else 1.0

    # Dibujar cortantes
    for mem, (v_izq, v_der) in V_vals.items():
        if "Vol" in mem:
            nD = mem.replace("Vol","")
            draw_diagram_member(ax_v, coords[nD][0], coords[nD][1], coords[mem][0], coords[mem][1], v_izq, v_der, scale_v, color='green', label1=f"{v_izq:.2f}", label2=f"{v_der:.2f}")
        else:
            n1, n2 = mem[0], mem[1]
            draw_diagram_member(ax_v, coords[n1][0], coords[n1][1], coords[n2][0], coords[n2][1], v_izq, v_der, scale_v, color='green', label1=f"{v_izq:.2f}", label2=f"{abs(v_der):.2f}")

    for ax in (ax_v, ax_m): 
        ax.set_xticks([]); ax.set_yticks([])
        ax.axis('off')
        
    st.pyplot(fig_res, clear_figure=True)
    plt.close(fig_res)
