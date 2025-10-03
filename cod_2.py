import math
import matplotlib.pyplot as plt
import numpy as np
import re

# ----------------------------------------------------------------------
# FUNÇÕES AUXILIARES PARA CÁLCULO CÚBICO (Método de Newton)
# ----------------------------------------------------------------------

def f_valor(x, c_x3, c_x2, c_x1, c_constante):
    """Calcula o valor da função f(x) no ponto x."""
    return (c_x3 * x**3) + (c_x2 * x**2) + (c_x1 * x) + c_constante

def f_derivada(x, c_x3, c_x2, c_x1):
    """Calcula o valor da derivada f'(x) no ponto x."""
    # Derivada de ax³ + bx² + cx + d é 3ax² + 2bx + c
    return (3 * c_x3 * x**2) + (2 * c_x2 * x) + c_x1

def newton_raphson(c_x3, c_x2, c_x1, c_constante, chute_inicial, max_iter=100, tolerancia=1e-6):
    """Encontra uma raiz real usando o Método de Newton-Raphson."""
    x = chute_inicial
    for _ in range(max_iter):
        fx = f_valor(x, c_x3, c_x2, c_x1, c_constante)
        fx_derivada = f_derivada(x, c_x3, c_x2, c_x1)
        
        # Evita divisão por zero se a derivada for muito próxima de zero
        if abs(fx_derivada) < 1e-10:
            return None 

        x_novo = x - (fx / fx_derivada)
        
        if abs(x_novo - x) < tolerancia:
            return x_novo
        
        x = x_novo
        
    return None 
    
# ----------------------------------------------------------------------
# FUNÇÃO 1: EXTRAÇÃO DE COEFICIENTES (Mantida, com 4 coeficientes)
# ----------------------------------------------------------------------

def extrair_coeficientes_da_string(funcao_str):
    """Extrai coeficientes (x³, x², x, constante) de forma simples (apenas inteiros)."""
    
    string_limpa = funcao_str.replace(' ', '').lower()
    
    if not re.search(r'[0-9x\^]', string_limpa):
         print("\nERRO: Entrada inválida. A função deve conter termos numéricos e/ou 'x', 'x^2' ou 'x^3'.")
         return None, None, None, None
         
    if not string_limpa.startswith(('+', '-')):
        string_limpa = '+' + string_limpa
        
    c_x3, c_x2, c_x1, c_constante = 0.0, 0.0, 0.0, 0.0

    try:
        # Padrão Regex para o termo cúbico (x^3) _________
        termo_cubico = re.search(r'([+-]?)(\d*)x\^3', string_limpa) 
        if termo_cubico:
            sinal = termo_cubico.group(1)
            numero = termo_cubico.group(2)
            c_x3 = 1.0 if numero == '' and sinal in ('', '+') else (-1.0 if numero == '' else float(sinal + numero))
        
        # Padrão Regex para o termo quadrático (x^2) _______
        termo_quadratico = re.search(r'([+-]?)(\d*)x\^2', string_limpa) 
        if termo_quadratico:
            sinal = termo_quadratico.group(1)
            numero = termo_quadratico.group(2)
            c_x2 = 1.0 if numero == '' and sinal in ('', '+') else (-1.0 if numero == '' else float(sinal + numero))

        # Padrão Regex para o termo linear (x) ____________
        termo_linear = re.search(r'([+-]?)(\d*)x(?![\^])', string_limpa) 
        if termo_linear:
            sinal = termo_linear.group(1)
            numero = termo_linear.group(2)
            c_x1 = 1.0 if numero == '' and sinal in ('', '+') else (-1.0 if numero == '' else float(sinal + numero))

        # Padrão Regex para o termo constante ______________
        termo_constante = re.search(r'([+-]\d+)$', string_limpa) 
        if termo_constante:
            c_constante = float(termo_constante.group(1))

    except Exception:
        print("\nERRO: O formato da função digitada não é válido ou a conversão falhou.")
        return None, None, None, None

    if c_x3 == 0.0 and c_x2 == 0.0 and c_x1 == 0.0:
        print("\nAVISO: Função constante (Grau Zero). A análise será limitada.")

    return c_x3, c_x2, c_x1, c_constante

# ----------------------------------------------------------------------
# FUNÇÃO 2: CÁLCULO DAS RAÍZES
# ----------------------------------------------------------------------

def calcular_raizes(c_x3, c_x2, c_x1, c_constante, grau_funcao):
    """Calcula raízes reais para 1º, 2º e 3º grau (usando Método de Newton para 3º)."""
    
    raizes_encontradas = []
    print("\n--- Cálculo de Raízes ---")

    if grau_funcao.startswith("1º Grau"):
        # ... (Lógica de 1º Grau)
        if c_x1 == 0:
            print("Função inválida para 1º grau (b=0). Não é possível calcular a raiz.")
            return raizes_encontradas
        
        valor_raiz = -c_constante / c_x1
        raizes_encontradas.append(valor_raiz)
        print(f"Raiz (Zero da Função): x = {valor_raiz:.3f}")

    elif grau_funcao.startswith("2º Grau"):
        # ... (Lógica de 2º Grau/Bhaskara)
        a = c_x2; b = c_x1; c = c_constante
        
        discriminante_delta = b**2 - 4 * a * c
        print(f"Discriminante (Δ): {discriminante_delta:.3f}")

        if discriminante_delta >= 0:
            denominador = 2 * a
            raiz_delta = math.sqrt(discriminante_delta)
            
            x_raiz1 = (-b + raiz_delta) / denominador
            x_raiz2 = (-b - raiz_delta) / denominador
            
            raizes_encontradas.extend([x_raiz1, x_raiz2])
            
            if abs(discriminante_delta) < 1e-9: # Usando tolerância para delta=0
                print(f"Resultado: Δ ≈ 0. Raiz única: x = {x_raiz1:.3f}")
            else:
                 print(f"Resultado: Δ > 0. Duas raízes: x'={x_raiz1:.3f} | x''={x_raiz2:.3f}")
        else:
            print("Resultado: Δ < 0. Não há raízes reais.")
        
    elif grau_funcao.startswith("3º Grau"):
        print("MÉTODO: Newton-Raphson e Deflação Polinomial.")
        
        # 1. Encontra a Primeira Raiz Real (r1)
        # Tenta chutar a partir do 0 (ponto mais simples)
        raiz1 = newton_raphson(c_x3, c_x2, c_x1, c_constante, chute_inicial=0.0)
        
        if raiz1:
            raizes_encontradas.append(raiz1)
            print(f"Raiz 1 (Encontrada por Newton): x1 = {raiz1:.3f}")

            # 2. Deflação Polinomial (Divisão Sintética)
            # Divide o polinômio original (grau 3) pela raiz (x - r1) para obter um polinômio de grau 2.
            # Coeficientes do Novo Polinômio de 2º Grau (A'x² + B'x + C')
            
            # A' é sempre o c_x3 original
            a_novo = c_x3
            
            # B' = c_x2 + r1 * A'
            b_novo = c_x2 + raiz1 * a_novo
            
            # C' = c_x1 + r1 * B'
            c_novo = c_x1 + raiz1 * b_novo
            
            # Nota: O resto (c_constante + r1 * C') deve ser zero (ou próximo de zero)
            
            # 3. Resolve o Novo Polinômio (grau 2) com Bhaskara
            
            delta_novo = b_novo**2 - 4 * a_novo * c_novo
            
            if delta_novo >= 0:
                raiz_delta_novo = math.sqrt(delta_novo)
                denominador_novo = 2 * a_novo
                
                raiz2 = (-b_novo + raiz_delta_novo) / denominador_novo
                raiz3 = (-b_novo - raiz_delta_novo) / denominador_novo
                
                raizes_encontradas.extend([raiz2, raiz3])
                print(f"Raízes 2 e 3 (Bhaskara do Polinômio Deflacionado): x2={raiz2:.3f} | x3={raiz3:.3f}")
            else:
                 print("As duas raízes restantes são complexas (Delta < 0 no polinômio deflacionado).")
        else:
            print("Não foi possível encontrar uma raiz real pelo Método de Newton. Tente outro chute inicial.")
        
    return raizes_encontradas

# ----------------------------------------------------------------------
# FUNÇÃO 3: PLOTAGEM DO GRÁFICO
# ----------------------------------------------------------------------

def plotar_grafico_da_funcao(c_x3, c_x2, c_x1, c_constante, raizes, grau_funcao):
    """Gera gráfico para funções até o 3º grau."""
    
    valores_x = np.linspace(-10, 10, 200) 
    
    valores_y = (c_x3 * valores_x**3) + (c_x2 * valores_x**2) + (c_x1 * valores_x) + c_constante

    plt.figure(figsize=(8, 6))
    
    rotulo = f"f(x) = {c_x3}x³ + {c_x2}x² + {c_x1}x + {c_constante}"
    plt.plot(valores_x, valores_y, label=rotulo, color='blue')

    if raizes:
        coordenadas_y_raizes = [0] * len(raizes)
        # Plota apenas raízes reais, convertendo para float (necessário para a plotagem)
        raizes_reais = [r for r in raizes if isinstance(r, (int, float))]
        
        plt.scatter(raizes_reais, coordenadas_y_raizes, color='red', zorder=5, label='Raízes Reais')

    # Estética
    plt.axhline(0, color='black', linewidth=0.8, linestyle='-')
    plt.axvline(0, color='black', linewidth=0.8, linestyle='-')
    plt.title(f'Gráfico de uma Função de {grau_funcao}')
    plt.xlabel('Eixo X')
    plt.ylabel('f(x)')
    plt.grid(True)
    plt.legend()
    plt.show()

# ----------------------------------------------------------------------
# FUNÇÃO PRINCIPAL: ORQUESTRAÇÃO SIMPLIFICADA
# ----------------------------------------------------------------------

def iniciar_analise_da_funcao():
    print("--- Análise de Funções Polinomiais (Até x^3) ---")
    print("FORMATO ESPERADO: ax^3+bx^2+cx+d. Coeficientes devem ser INTEIROS.")
    
    entrada_funcao = input("\nDigite a função (ex: x^3-4x ou -2x^2+3x+1): ")

    c_x3, c_x2, c_x1, c_constante = extrair_coeficientes_da_string(entrada_funcao)

    if c_x3 is None:
        return

    if c_x3 != 0:
        grau_funcao = "3º Grau (Função Cúbica)"
    elif c_x2 != 0:
        grau_funcao = "2º Grau (Função Quadrática)"
    elif c_x1 != 0:
        grau_funcao = "1º Grau (Função Afim)"
    else:
        grau_funcao = "Grau Zero (Função Constante)"

    print(f"\nGrau Identificado: {grau_funcao}")
    
    raizes_finais = []
    if grau_funcao != "Grau Zero (Função Constante)":
        # Passa todos os 4 coeficientes para a função de cálculo de raízes
        raizes_finais = calcular_raizes(c_x3, c_x2, c_x1, c_constante, grau_funcao)

    plotar_grafico_da_funcao(c_x3, c_x2, c_x1, c_constante, raizes_finais, grau_funcao)

# ----------------------------------------------------------------------
# EXECUÇÃO
# ----------------------------------------------------------------------
iniciar_analise_da_funcao()