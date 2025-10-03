import math
import matplotlib.pyplot as plt
import numpy as np
import re

# ----------------------------------------------------------------------
# FUNÇÃO 1: EXTRAÇÃO DE COEFICIENTES
# ----------------------------------------------------------------------

def extrair_coeficientes_da_string(funcao_str):
    # 1. Limpeza e Padronização
    string_limpa = funcao_str.replace(' ', '').lower()
    
    # Validação Básica: Deve conter algo que pareça um termo de função.
    if not re.search(r'[0-9x\^]', string_limpa):
         print("\nERRO: Entrada inválida. A função deve conter termos numéricos (coeficientes) e/ou as variáveis 'x' e 'x^2'.")
         return None, None, None
         
    # Adiciona um '+' inicial para padronizar o Regex.
    if not string_limpa.startswith(('+', '-')):
        string_limpa = '+' + string_limpa
        
    # Inicializa coeficientes
    coef_a, coef_b, coef_c = 0.0, 0.0, 0.0

    try:
        # Padrão Regex para o termo quadrático (ax^2) ________
        #Captura o sinal, nº inteiro incluindo vazio, e x^2
        termo_quadratico = re.search(r'([+-]?)(\d*)x\^2', string_limpa) 
        if termo_quadratico:
            sinal = termo_quadratico.group(1)
            numero = termo_quadratico.group(2)
            
            if numero == '':
                coef_a = 1.0 if sinal in ('', '+') else -1.0
            else:
                coef_a = float(sinal + numero)

        # Padrão Regex para o termo linear (bx) ___________
        # regex captura sinal, nº inteiro incluindo vazio, x, e exclui ^
        termo_linear = re.search(r'([+-]?)(\d*)x(?![\^])', string_limpa) 
        if termo_linear:
            sinal = termo_linear.group(1)
            numero = termo_linear.group(2)
            
            if numero == '':
                coef_b = 1.0 if sinal in ('', '+') else -1.0
            else:
                coef_b = float(sinal + numero)

        # Padrão Regex para o termo constante (c)____________
        #captura sinal, e o número. Exige pelo menos 1 inteiro. $ garante que a captura será feita se estiver no fim da #string
        termo_constante= re.search(r'([+-]\d+)$', string_limpa) 
        if termo_constante:
            coef_c = float(termo_constante.group(1))

    except Exception:
        print("\nERRO: O formato da função digitada não é válido ou a conversão falhou.")
        return None, None, None

    if coef_a == 0.0 and coef_b == 0.0:
        print("\nAVISO: Função constante (Grau Zero). A análise será limitada.")

    return coef_a, coef_b, coef_c

# ----------------------------------------------------------------------
# FUNÇÃO 2: CÁLCULO DAS RAÍZES
# ----------------------------------------------------------------------

def calcular_raizes(coef_a, coef_b, coef_c, grau_funcao):
    raizes_encontradas = []
    print("\n--- Cálculo de Raízes ---")

    if grau_funcao.startswith("1º Grau"):
        if coef_b == 0:
            print("Função inválida para 1º grau (b=0). Não é possível calcular a raiz.")
            return raizes_encontradas
        
        valor_raiz = -coef_c / coef_b
        raizes_encontradas.append(valor_raiz)
        print(f"Raiz (Zero da Função): x = {valor_raiz:.3f}")

    elif grau_funcao.startswith("2º Grau"):
        discriminante_delta = coef_b**2 - 4 * coef_a * coef_c
        print(f"Discriminante (Δ): {discriminante_delta:.3f}")

        if discriminante_delta < 0:
            print("Resultado: Δ < 0. Não há raízes reais.")
        elif discriminante_delta == 0:
            x_unica = -coef_b / (2 * coef_a)
            raizes_encontradas.append(x_unica)
            print(f"Resultado: Δ = 0. Raiz única: x = {x_unica:.3f}")
        else:
            raiz_delta = math.sqrt(discriminante_delta) #calculo raiz quadrada
            denominador = 2 * coef_a
            
            x_raiz1 = (-coef_b + raiz_delta) / denominador
            x_raiz2 = (-coef_b - raiz_delta) / denominador
            raizes_encontradas.extend([x_raiz1, x_raiz2])
            print(f"Resultado: Δ > 0. Duas raízes: x'={x_raiz1:.3f} | x''={x_raiz2:.3f}")
        
    return raizes_encontradas

# ----------------------------------------------------------------------
# FUNÇÃO 3: PLOTAGEM DO GRÁFICO
# ----------------------------------------------------------------------

def plotar_grafico_da_funcao(coef_a, coef_b, coef_c, raizes, grau_funcao):
    valores_x = np.linspace(-10, 10, 100) #função de espaço linear do numpy.Usa os parâmetros (início,fim,quantidade)
    valores_y = coef_a * valores_x**2 + coef_b * valores_x + coef_c

    plt.figure(figsize=(8, 6)) #largura/altura
    plt.plot(valores_x, valores_y, 
             label=f'f(x) = {coef_a}x² + {coef_b}x + {coef_c}', 
             color='blue')

    if raizes:
        #cria uma lista com zeros do mesmo tamanho da lista de raízes. Ex (0,0)
        coordenadas_y_raizes = [0] * len(raizes)
        plt.scatter(raizes, coordenadas_y_raizes, color='red', zorder=5, label='Raízes')

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
    print("--- Análise de Funções Polinomiais (ax^2 + bx + c) ---")
    print("FORMATO ESPERADO: ax^2+bx+c. Coeficientes devem ser INTEIROS.")
    
    # Loop ÚNICO, sem repetição em caso de erro. SIMPLICIDADE
    entrada_funcao = input("\nDigite a função (ex: x^2-5x+6 ou 2x+4): ")

    # 1. Extração dos Coeficientes
    coef_a, coef_b, coef_c = extrair_coeficientes_da_string(entrada_funcao)

    # 2. Checagem de Falha
    if coef_a is None:
        return # Encerra o programa em caso de erro

    # 3. Identificação do Grau
    if coef_a != 0:
        grau_funcao = "2º Grau (Função Quadrática)"
    elif coef_b != 0:
        grau_funcao = "1º Grau (Função Afim)"
    else:
        grau_funcao = "Grau Zero (Função Constante)"

    print(f"\nGrau Identificado: {grau_funcao}")
    
    # 4. Cálculo das Raízes
    raizes_finais = []
    if coef_a != 0 or coef_b != 0:
        raizes_finais = calcular_raizes(coef_a, coef_b, coef_c, grau_funcao)
    
    # 5. Plotagem do Gráfico
    plotar_grafico_da_funcao(coef_a, coef_b, coef_c, raizes_finais, grau_funcao)

# ----------------------------------------------------------------------
# EXECUÇÃO
# ----------------------------------------------------------------------
iniciar_analise_da_funcao()