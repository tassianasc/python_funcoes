# 📊 Análise e Visualização de Funções Matemáticas

**Uma ferramenta de análise matemática que transforma funções complexas em visualizações interpretáveis, com cálculo automático de raízes, pontos críticos e comportamento funcional.**

---

## 📋 Sumário

1. [Visão Geral](#visão-geral)
2. [Objetivos do Projeto](#objetivos-do-projeto)
3. [Tecnologias Utilizadas](#tecnologias-utilizadas)
4. [Funcionalidades](#funcionalidades)
5. [Instalação e Configuração](#instalação-e-configuração)
6. [Como Utilizar](#como-utilizar)
7. [Exemplos de Saída](#exemplos-de-saída)
8. [Estrutura do Projeto](#estrutura-do-projeto)
9. [Insights Gerados](#insights-gerados)
10. [Contribuindo](#contribuindo)

---

## 🎯 Visão Geral

Este projeto foi desenvolvido como ferramenta de **análise exploratória e visualização de dados matemáticos**. Ele recebe funções matemáticas como entrada, calcula suas características principais (raízes, máximos, mínimos) e gera gráficos visuais que facilitam a interpretação do comportamento funcional.

**Ideal para:**
- Analistas que precisam visualizar comportamentos de dados matemáticos
- Estudantes compreendendo conceitos de funções
- Cientistas de dados explorando transformações de dados
- Engenheiros necessitando análise rápida de funções

---

## 🔍 Objetivos do Projeto

✅ **Automatizar análise** de funções matemáticas complexas  
✅ **Calcular automaticamente** raízes, máximos e mínimos  
✅ **Gerar visualizações** claras e interpretáveis  
✅ **Facilitar exploração** de dados através de gráficos  
✅ **Documentar padrões** comportamentais de funções  

---

## 🛠️ Tecnologias Utilizadas

| Tecnologia | Versão | Descrição |
|-----------|--------|-----------|
| **Python** | 3.8+ | Linguagem de programação |
| **NumPy** | 1.20+ | Computação numérica e manipulação de arrays |
| **Matplotlib** | 3.3+ | Visualização de dados e gráficos |
| **SciPy** | 1.5+ | Ferramentas científicas e otimização |
| **Jupyter Notebook** | 6.0+ | Ambiente interativo de análise |

---

## ⭐ Funcionalidades

### 1️⃣ **Análise de Raízes**
- Detecta e calcula automaticamente as raízes (zeros) da função
- Aplica métodos numéricos robustos (Newton-Raphson, bisseção)
- Retorna lista completa de raízes encontradas

### 2️⃣ **Pontos Críticos**
- Identifica máximos e mínimos locais
- Calcula derivadas para encontrar pontos de inflexão
- Classifica pontos críticos (máximo, mínimo, sela)

### 3️⃣ **Análise de Comportamento**
- Estuda o comportamento em intervalos definidos
- Identifica crescimento/decrescimento
- Detecta descontinuidades e assíntotas

### 4️⃣ **Visualização Interativa**
- Gráficos claros com múltiplas camadas de informação
- Marcação visual de raízes, máximos e mínimos
- Suporte para múltiplas funções simultâneas
- Exportação de gráficos em alta qualidade

### 5️⃣ **Análise Estatística**
- Cálculo de domínio e imagem
- Análise de simetria (par/ímpar)
- Estatísticas de distribuição

---

## 🚀 Instalação e Configuração

### **Pré-requisitos**
- Python 3.8 ou superior
- pip (gerenciador de pacotes Python)
- Git (opcional)

### **Passo 1: Clonar o Repositório**

```bash
git clone https://github.com/tassianasc/python_funcoes.git
cd python_funcoes
```

### **Passo 2: Criar Ambiente Virtual (Recomendado)**

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### **Passo 3: Instalar Dependências**

```bash
pip install -r requirements.txt
```

**Conteúdo do `requirements.txt`:**
```
numpy>=1.20.0
matplotlib>=3.3.0
scipy>=1.5.0
jupyter>=6.0.0
```

### **Passo 4: Verificar Instalação**

```bash
python -c "import numpy, matplotlib, scipy; print('Todas as dependências instaladas com sucesso!')"
```

---

## 💻 Como Utilizar

### **Método 1: Script Python Direto**

```python
from analise_funcoes import AnalisadorFuncoes
import numpy as np

# Criar instância do analisador
analisador = AnalisadorFuncoes()

# Definir a função
def f(x):
    return x**2 - 5*x + 6

# Executar análise completa
resultado = analisador.analisar(
    funcao=f,
    intervalo=(-2, 8),
    nome_funcao="f(x) = x² - 5x + 6"
)

# Exibir resultados
print("Raízes encontradas:", resultado['raizes'])
print("Máximos:", resultado['maximos'])
print("Mínimos:", resultado['minimos'])

# Gerar visualização
analisador.plotar_grafico(resultado, salvar_como='analise.png')
```

### **Método 2: Jupyter Notebook (Recomendado)**

```python
%matplotlib inline
from analise_funcoes import AnalisadorFuncoes
import numpy as np

# Análise interativa
analisador = AnalisadorFuncoes()

# Função 1: Polinômio
f1 = lambda x: x**3 - 3*x + 2
resultado1 = analisador.analisar(f1, (-3, 3), "Cúbica: x³ - 3x + 2")

# Função 2: Exponencial
f2 = lambda x: np.exp(-x**2) * np.cos(2*np.pi*x)
resultado2 = analisador.analisar(f2, (-2, 2), "Exponencial: e^(-x²)·cos(2πx)")

# Visualizar juntas
analisador.plotar_multiplas([resultado1, resultado2])
```

### **Método 3: CLI (Linha de Comando)**

```bash
python analise_funcoes.py --funcao "x**2 - 5*x + 6" --intervalo "-2,8" --saida resultado.png
```

---

## 📊 Exemplos de Saída

### **Exemplo 1: Função Quadrática**
```
Função: f(x) = x² - 5x + 6
Intervalo: [-2, 8]

📍 RAÍZES ENCONTRADAS:
   - x = 2.0
   - x = 3.0

📈 PONTOS CRÍTICOS:
   - Mínimo em x = 2.5, f(x) = -0.25
   - Máximo em x = 8.0, f(x) = 30.0

📊 ANÁLISE DE COMPORTAMENTO:
   - Decrescente em: [-2, 2.5]
   - Crescente em: [2.5, 8]
   - Concavidade: Para cima (côncava)
```

### **Exemplo 2: Função Trigonométrica**
```
Função: f(x) = sin(x)
Intervalo: [0, 2π]

📍 RAÍZES ENCONTRADAS:
   - x = 0.0
   - x = 3.14159...
   - x = 6.28318...

📈 PONTOS CRÍTICOS:
   - Máximo em x = 1.5708, f(x) = 1.0
   - Mínimo em x = 4.7124, f(x) = -1.0
```

---

## 🗂️ Estrutura do Projeto

```
python_funcoes/
├── README.md                          # Este arquivo
├── requirements.txt                   # Dependências do projeto
├── src/
│   ├── __init__.py
│   ├── analise_funcoes.py            # Classe principal de análise
│   ├── calculadores.py               # Cálculos numéricos
│   └── visualizador.py               # Geração de gráficos
├── notebooks/
│   ├── exemplo_basico.ipynb          # Tutorial básico
│   ├── analise_avancada.ipynb        # Casos complexos
│   └── comparacao_funcoes.ipynb      # Análise comparativa
├── exemplos/
│   ├── exemplo_quadratica.py
│   ├── exemplo_trigonometrica.py
│   └── exemplo_exponencial.py
├── testes/
│   ├── test_raizes.py
│   ├── test_pontos_criticos.py
│   └── test_visualizacao.py
└── saidas/                           # Gráficos gerados
    └── (gráficos em PNG)
```

---

## 🔬 Insights Gerados

Este projeto demonstra capacidades essenciais para **Análise de Dados**:

### **1. Exploração de Dados Matemáticos**
- Identifica padrões e características principais
- Transforma dados abstratos em visualizações concretas
- Similar à exploração inicial em análises reais

### **2. Tratamento e Transformação**
- Manipulação de arrays numéricos
- Cálculos computacionais complexos
- Otimização de performance

### **3. Comunicação Visual**
- Gráficos informativos e bem estruturados
- Legendas claras e interpretáveis
- Design que facilita tomada de decisão

### **4. Automatização**
- Scripts reutilizáveis
- Processamento em lote de múltiplas funções
- Redução de trabalho manual

---

## 📈 Casos de Uso

| Caso de Uso | Exemplo |
|---|---|
| **Análise de Tendências** | Identificar picos em series temporais através de funções |
| **Otimização** | Encontrar máximos/mínimos em processos de negócio |
| **Previsão** | Validar comportamento de modelos matemáticos |
| **Educação** | Visualizar conceitos matemáticos complexos |
| **Pesquisa** | Explorar propriedades de funções desconhecidas |

---

## 🧪 Testes

Execute os testes unitários:

```bash
# Rodar todos os testes
python -m pytest testes/

# Rodar testes específicos
python -m pytest testes/test_raizes.py -v

# Com cobertura
python -m pytest --cov=src testes/
```

---

## 📝 Exemplos de Uso Prático

### **Analisar Função Polinomial**
```python
f = lambda x: x**3 - 2*x**2 - 5*x + 6
analisador.analisar(f, (-3, 4), "Polinômio Cúbico")
```

### **Comparar Múltiplas Funções**
```python
funcoes = [
    (lambda x: x**2, "Quadrática"),
    (lambda x: x**3, "Cúbica"),
    (lambda x: np.sqrt(x), "Raiz Quadrada")
]
analisador.comparar_funcoes(funcoes, (0, 5))
```

### **Exportar Análise Completa**
```python
analisador.gerar_relatorio(resultado, formato='PDF', salvar_como='relatorio_analise.pdf')
```

---

## 🤝 Contribuindo

Contribuições são bem-vindas! Para contribuir:

1. Faça um fork do repositório
2. Crie uma branch para sua feature (`git checkout -b feature/nova-funcionalidade`)
3. Commit suas mudanças (`git commit -m 'Adiciona nova funcionalidade'`)
4. Push para a branch (`git push origin feature/nova-funcionalidade`)
5. Abra um Pull Request

---

## 📊 Habilidades Técnicas Demonstradas

✅ **Python Avançado** - Programação orientada a objetos  
✅ **Computação Numérica** - NumPy e SciPy  
✅ **Visualização de Dados** - Matplotlib com layouts complexos  
✅ **Análise Matemática** - Cálculos de derivadas e raízes  
✅ **Jupyter Notebooks** - Documentação interativa  
✅ **Boas Práticas** - Código limpo e bem documentado  

---

## 📞 Contato & Links

- **GitHub:** [@tassianasc](https://github.com/tassianasc)
- **Portfólio:** [github.com/tassianasc](https://github.com/tassianasc)

---

## 📄 Licença

Este projeto está licenciado sob a MIT License - veja o arquivo [LICENSE](LICENSE) para detalhes.

---

## 🙏 Agradecimentos

Desenvolvido como projeto acadêmico focado em análise exploratória de dados e visualização.

---

**Última atualização:** Dezembro 2024  
**Versão:** 1.0.0