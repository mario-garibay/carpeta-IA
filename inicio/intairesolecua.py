from sympy import symbols, Eq, solve

x = symbols('x')
int=input("ingrese dato1")
int=input("dato2")
int=input("dato3")
resultado=('res ') 
ecuacion = Eq(input)
soluciones = solve(ecuacion, x)
print(f"Las soluciones son: {soluciones}")
