import tensorflow as tf 

import numpy as np 

tiempo = np.array([0,15,30,45,60,75,90,105,120,135,150,165,180,195,210,225,240,255,270,285,300,315,330,345,360,375,390,405,420,435], dtype=float) 

trafico = np.array([10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200,210,220,230,240,250,260,270,280,290,300], dtype=float) 


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

historial = modelo.fit(tiempo,trafico, epochs=50, verbose=False) 

print("Modelo entrenado!") 

import matplotlib.pyplot as plt 

plt.xlabel("# Epoca") 

plt.ylabel("Magnitud de pérdida") 

plt.plot(historial.history["loss"]) 

plt.show() 

print("Hagamos una predicción!") 

resultado = modelo.predict(np.array([[150.0]])) 

print("El resultado es " + str(resultado) + " trafico de red!") 