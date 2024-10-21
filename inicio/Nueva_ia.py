import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt

# Definir x como un array de valores
x = np.array([0, 1, 2, 3, 4], dtype=float)

# Calcular las derivadas y la respuesta
derivada = np.array([x**1, x**3, x**5, x**8, x**10], dtype=float).T  # Transponer para que tenga la forma correcta
respuesta = np.array([1, 3*x**2, 5*x**4, 8*x**7, 10*x**9], dtype=float)  # No transponer aquí

# Asegúrate de que respuesta tenga la forma correcta
respuesta = respuesta.reshape(-1, 1)  # Cambiar la forma a (5, 1)

# Capa oculta con 64 neuronas y función de activación ReLU 
capa_oculta = tf.keras.layers.Dense(units=64, activation='relu', input_shape=[1]) 
capa_salida = tf.keras.layers.Dense(units=1) 

# Modelo secuencial con capas ocultas y de salida 
modelo = tf.keras.Sequential([capa_oculta, capa_salida]) 
modelo.compile( 
    optimizer=tf.keras.optimizers.Adam(0.1), 
    loss='mean_squared_error' 
)   

print("Comenzando entrenamiento...") 
historial = modelo.fit(derivada, respuesta, epochs=50, verbose=False) 
print("Modelo entrenado!") 

# Gráfica de la magnitud de pérdida
plt.xlabel("# Época") 
plt.ylabel("Magnitud de pérdida") 
plt.plot(historial.history["loss"]) 
plt.show() 

print("Hagamos una predicción!") 

# Realizar una predicción, por ejemplo, para x=12
resultado = modelo.predict(np.array([[12.0]])) 
print("El resultado es " + str(resultado))