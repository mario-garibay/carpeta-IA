import tensorflow as tf 

import numpy as np 

celsius = np.array([-40, -10, 0, 8, 15, 22, 38, 42, 52, 62], dtype=float) 

fahrenheit = np.array([-40, -14, 32, 46, 59, 72, 100, 108, 126, 144], dtype=float) 


# Capa oculta con 2 neuronas y función de activación ReLU 

capa_oculta = tf.keras.layers.Dense(units=64, activation='relu', input_shape=[1]) 

# Capa de salida con 1 neurona 

capa_salida = tf.keras.layers.Dense(units=1) 


# Modelo secuencial con capas ocultas y de salida 

modelo = tf.keras.Sequential([capa_oculta, capa_salida]) 

modelo.compile( 

    optimizer=tf.keras.optimizers.Adam(0.1), 

    loss='mean_squared_error' 

)   

print("Comenzando entrenamiento...") 

historial = modelo.fit(celsius, fahrenheit, epochs=400, verbose=False) 

print("Modelo entrenado!") 

import matplotlib.pyplot as plt 

plt.xlabel("# Epoca") 

plt.ylabel("Magnitud de pérdida") 

plt.plot(historial.history["loss"]) 

plt.show() 

print("Hagamos una predicción!") 

resultado = modelo.predict(np.array([[100.0]])) 

print("El resultado es " + str(resultado) + " fahrenheit!") 