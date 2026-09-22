# Archivo base para el despliegue del Agente en Streamlit
import streamlit as st
from sklearn.linear_model import LinearRegression
import numpy as np

# En streamlit vamos a agregar un titolo para la pagina web
st.title("Configuracion inicial")
# Agregamos un textbox a la pagina web
st.write("Primera prueba de uso de streamlit y ambiente de MA2026")

# Usar un slider que perm9ite ingresar por un slider el parametro de inversion
gasto=st.slider("Seleccine nivel de gasto en publicicdad", 10,200,50) # de 10 a 200 en brincos de 50

# Variables del modelo
variable_x = np.array([[10], [20], [30], [40],[50]])
variable_y = np.array([15,25,35,45,55])
modelo_lr = LinearRegression()

# Entrenamiento de nuestro modelo LR
modelo_lr.fit(variable_x,variable_y)

# En streamlit, tenemos un boton que dice Predecir y al dar click activara las lineas de codigo del i
if st.button("Predecir"):
    resultado = modelo_lr.predict([[gasto]])

    # Streamlit muestra un mensaje de exito usando success
    st.success(f"Las ventas proyectadas para una inversion de ${gasto} son: ${resultado[0]}")
