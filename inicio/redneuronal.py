import tensorflow as tf
import numpy as np

celsius = np.array([-40,-10,0,8,15,22,38,42,62,82,92,100],dtype=float)
fahrenheit=np.array([-40,-14,32,46,59,72,100,108,144,180,198,2],dtype=float)

capa = tf.keras.layers.Dense(units=1,activation='relu', input_shape=[1])
modelo = tf.keras.Sequential([capa])

modelo.compile(
    optimizer =tf.keras.optimizers.Adam(0.1),
    loss='mean_squared_error'
)

print("comenzando entrenamiento...")
historial = modelo.fit(celsius,fahrenheit,epochs=400,verbose=False)
print("modelo entrenado!")

import matplotlib.pyplot as plt
plt.xlabel("# epoca")
plt.ylabel("magnitud de perdida")
plt.plot(historial.history["loss"])
plt.show()

print("hagamos una prediccion!")
resultado = modelo.predict(numpy.array([[100.0]]))
print("el resultado es " + str(resultado)+"fahrenheit")

