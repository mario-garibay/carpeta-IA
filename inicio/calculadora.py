#calculadora
print("Hola, como estan mi nombre es mario ")


num1 = int(input("numero 1: "))
num2 = int(input("numero 2: "))
operacion = input("introduce una operacion(+ - * /)")


match operacion:
      case "+":
            res = num1+num2 
      case "-":
            res = num1-num2
      case "*":
            res = num1*num2
      case "/":
            res = num1/num2

print(f"el resultado de {num1} {operacion} {num2} = {res}")   