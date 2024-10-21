# primer programa python 
## mario diaz 




edad=18

if edad >= 18:
    print("eres mayor de edad")
else:
    print("eres menos de edad")


---------------------------------------------------------------------------------------------
red neuronal

import tensorflow as tf
import numpy as numpy

celsius = numpy.array([-40,-10,0,8,15,22,38,42],dtype=float)
fahrenheit=numpy.array([-40,-14,32,46,59,72,100,108],dtype=float)

capa = tf.keras.layers.Dense(units=1, input_shape=[1])
modelo = tf.keras.Sequential([capa])

modelo.compile(
    optimizer =tf.keras.optimizers.Adam(0.01),
    loss='mean_squared_error'
)

print("comenzando entrenamiento...")
historial = modelo.fit(celsius,fahrenheit,epochs=500,verbose=False)
print("modelo entrenado!")

import matplotlib.pyplot as plt
plt.xlabel("# epoca")
plt.ylabel("magnitud de perdida")
plt.plot(historial.history["loss"])
plt.show()



------------------------------------------------------------------------------------------------------------------