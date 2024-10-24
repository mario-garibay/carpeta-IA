import sympy as sp
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Paso 1: Definir el problema y recolectar datos
# Crear un DataFrame ficticio para ilustrar
data = pd.DataFrame({
    'feature1': ["2*x + 3", "x**2 + x + 1", "x*2","3*x-7","X*2 + 3"],
    'feature2': ["x", "x", "x" ,"x","X"],
    'target': ["2", "2*x + 1", "2" ,"3","2"]
})

# Guardar el DataFrame como un CSV (solo para fines ilustrativos)
data.to_csv('tus_datos.csv', index=False)

# Leer el archivo CSV (lo harías con tus datos reales)
data = pd.read_csv('tus_datos.csv')
data_clean = data.dropna()

# Mostrar el DataFrame limpio
print(data_clean)

# Paso 2: Dividir datos y entrenar el modelo
# Aquí debes crear características numéricas reales
# Una opción es usar una representación simple:
def extract_features(expr_str, var_str):
    expr = sp.sympify(expr_str)
    var = sp.Symbol(var_str)
    return [expr.coeff(var, i) for i in range(3)]

data_clean['features'] = data_clean.apply(lambda row: extract_features(row['feature1'], row['feature2']), axis=1)
X = list(data_clean['features'])
y = data_clean['target'].apply(lambda expr: float(sp.sympify(expr).subs(sp.Symbol('x'), 1))).tolist()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluar la precisión del modelo
score = model.score(X_test, y_test)
print(f"Precisión del modelo: {score}")

# Paso 3: Función para predecir nueva entrada
def predecir_nueva_entrada(feature1, feature2):
    expr = sp.sympify(feature1)
    nueva_entrada = [expr.coeff(sp.Symbol(feature2), i) for i in range(3)]
    prediccion = model.predict([nueva_entrada])
    return prediccion

# Paso 4: Pedir al usuario una función y calcular su derivada
def resolver_funcion():
    expr_str = input("Introduce la expresión matemática (ej. 'x**2 + 3*x + 2'): ")
    var_str = input("Introduce la variable (ej. 'x'): ")
    x_val = float(input("Introduce el valor de x: "))
    
    # Preparar la expresión y la variable
    expr = sp.sympify(expr_str)
    var = sp.Symbol(var_str)

    # Calcular derivada simbólica
    derivada = sp.diff(expr, var)

    # Evaluar la función y la derivada en el valor dado
    funcion_val = expr.subs(var, x_val)
    derivada_val = derivada.subs(var, x_val)

    # Imprimir resultados
    print(f"Función: {expr}")
    print(f"Derivada: {derivada}")
    print(f"Valor de la función en x={x_val}: {funcion_val}")
    print(f"Valor de la derivada en x={x_val}: {derivada_val}")

# Paso 5: Llamar a la función para resolver
resolver_funcion()

# Paso 6: Ejemplo de uso de la predicción
feature1 = input("Introduce la función (ej. '2*x + 3'): ")
feature2 = input("Introduce la variable (ej. 'x'): ")
resultado = predecir_nueva_entrada(feature1, feature2)
print(f"Predicción: {resultado}")
