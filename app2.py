import streamlit as st
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Configuración de símbolos
x, y = sp.symbols('x y')

# Funciones disponibles para sympify
funciones = {
    'sin': sp.sin, 'cos': sp.cos, 'tan': sp.tan,
    'log': sp.log, 'ln': sp.log, 'exp': sp.exp, 'sqrt': sp.sqrt,
    'Abs': sp.Abs, 'asin': sp.asin, 'acos': sp.acos, 'atan': sp.atan,
    'pi': sp.pi, 'e': sp.E
}

# Título
st.title("🌐 Visualizador de Superficies y Gradientes")

# Entrada de función
funcion_input = st.text_input("Ingresa una función explícita de dos variables (f(x, y)):", "x**2 + y**2")

# Parámetros de malla
st.sidebar.header("⚙️ Parámetros de la malla")
x_min = st.sidebar.number_input("x mínimo", value=-5.0)
x_max = st.sidebar.number_input("x máximo", value=5.0)
y_min = st.sidebar.number_input("y mínimo", value=-5.0)
y_max = st.sidebar.number_input("y máximo", value=5.0)
res = st.sidebar.slider("Resolución", 20, 200, 50)

# Punto para evaluación de gradiente
st.sidebar.header("📍 Punto para evaluar gradiente")
x0 = st.sidebar.number_input("x₀", value=1.0)
y0 = st.sidebar.number_input("y₀", value=1.0)

if funcion_input:
    try:
        # Convertir a expresión simbólica
        f = sp.sympify(funcion_input, locals=funciones)

        # Derivadas parciales
        fx = sp.diff(f, x)
        fy = sp.diff(f, y)

        # Mostrar fórmulas
        st.subheader("📘 Función y derivadas")
        st.latex(f"f(x, y) = {sp.latex(f)}")
        st.latex(f"\\frac{{\\partial f}}{{\\partial x}} = {sp.latex(fx)}")
        st.latex(f"\\frac{{\\partial f}}{{\\partial y}} = {sp.latex(fy)}")


        # Vector gradiente simbólico
        gradiente = sp.Matrix([fx, fy])

        st.subheader("🧮 Vector Gradiente Simbólico")
        st.latex(f"\\nabla f(x, y) = \\left( {sp.latex(fx)},\ {sp.latex(fy)} \\right)")
        #st.code(f"∇f(x, y) = ({fx}, {fy})", language="python")


        # Mostrar como código
        #st.code(f"""f(x, y) = {funcion_input}
#∂f/∂x = {fx}
#∂f/∂y = {fy}""", language="python")

        # Funciones evaluables
        f_np = sp.lambdify((x, y), f, "numpy")
        fx_np = sp.lambdify((x, y), fx, "numpy")
        fy_np = sp.lambdify((x, y), fy, "numpy")

        # Crear malla
        X_vals = np.linspace(x_min, x_max, res)
        Y_vals = np.linspace(y_min, y_max, res)
        X, Y = np.meshgrid(X_vals, Y_vals)

        # Evaluar funciones
        with np.errstate(all='ignore'):
            Z = f_np(X, Y)
            U = fx_np(X, Y)
            V = fy_np(X, Y)

        Z = np.nan_to_num(Z, nan=np.nan, posinf=np.nan, neginf=np.nan)
        U = np.nan_to_num(U)
        V = np.nan_to_num(V)

        # Gráfica 3D
        st.subheader("🗻 Superficie 3D")
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.9)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("f(x, y)")
        fig.colorbar(surf, ax=ax, shrink=0.5)
        st.pyplot(fig)

        # Campo vectorial 2D
        st.subheader("🧭 Campo de Gradiente (2D)")
        fig2, ax2 = plt.subplots(figsize=(6, 6))
        ax2.quiver(X, Y, U, V, color='blue')
        ax2.set_title("Campo de Vectores del Gradiente")
        ax2.set_xlabel("x")
        ax2.set_ylabel("y")
        ax2.axis('equal')
        st.pyplot(fig2)

        # Líneas de nivel
        st.subheader("📉 Contornos de Nivel")
        fig3, ax3 = plt.subplots(figsize=(6, 5))
        cp = ax3.contourf(X, Y, Z, levels=30, cmap='viridis')
        ax3.set_title("Líneas de nivel")
        ax3.set_xlabel("x")
        ax3.set_ylabel("y")
        fig3.colorbar(cp)
        st.pyplot(fig3)

        # Evaluación en punto (gradiente)
        fx_val = fx_np(x0, y0)
        fy_val = fy_np(x0, y0)
        st.subheader("📌 Evaluación del Gradiente")
        st.write(f"En el punto (x₀, y₀) = ({x0}, {y0}):")
        st.latex(f"\\nabla f(x_0, y_0) = ({fx_val:.4f}, {fy_val:.4f})")

    except Exception as e:
        st.error(f"❌ Error al procesar la función: {e}")
