# Agente para Coberturas con Futuros de Energía 🤖⚡

Este repositorio desarrolla un agente inteligente para operar en el mercado de futuros de energía colombiano (XM). El proyecto utiliza algoritmos de aprendizaje profundo por refuerzo (DRL) para tomar decisiones de inversión automáticas basadas en datos del mercado eléctrico.

## 📋 Descripción General

El bot propuesto es un **agente autónomo** que realiza análisis del mercado de futuros de energía e implementa estrategias de cobertura mediante redes neuronales profundas. El proyecto forma parte de un trabajo integrador para la Maestría en Analítica e Inteligencia de Negocios de la Universidad del Valle.

## 🎯 Objetivos Principales

- Automatizar decisiones de cobertura en mercados de futuros de energía
- Integrar múltiples fuentes de datos (precios spot, futuros, demanda, generación, noticias)
- Implementar estrategias de cobertura basadas en aprendizaje por refuerzo
- Generar eficiencias en cuánto a costo de compra de electricidad

## 📁 Estructura del Proyecto

```
bot_futuros/
├── src/                              # Código fuente principal
│   ├── 01_load_data.py              # Carga de datos desde múltiples fuentes
│   ├── 02_clean_data.py             # Limpieza, procesamiento y feature engineering
│   ├── requirements.txt             # Dependencias del proyecto
│   │
│   ├── agents/                      # Agentes de trading
│   │   ├── option1/                 # Primera estrategia
│   │   └── option2/                 # Segunda estrategia
│   │
│   ├── models/                      # Modelos predictivos
│   │   ├── news/                    # Análisis de noticias con NLP
│   │   ├── option1/                 # Modelos para estrategia 1
│   │   └── option2/                 # Modelos para estrategia 2
│   │
│   ├── poc/                         # Pruebas de concepto
│   │   ├── copilot/                 # Implementación con Copilot
│   │   ├── gemini1/                 # Implementación con Gemini (opción 1)
│   │   └── gemini2/                 # Implementación con Gemini (opción 2)
│   │
│   └── utils/                       # Utilidades y funciones auxiliares
│       ├── load_data_energia.py     # Carga datos del mercado eléctrico (XM)
│       ├── load_data_futuros.py     # Carga precios de futuros
│       ├── preprocesamiento_datos.py# Procesamiento de variables del sistema
│       ├── simulation_demand.py     # Simulación de demanda por comprador
│       ├── get_news.py              # Obtención de noticias desde Google News
│       └── model_news.py            # Etiquetado de noticias (NLP)
│
├── data/                            # Datos (no incluidos en repo)
│   ├── 1_raw/                       # Datos brutos del mercado
│   ├── 2_silver/                    # Datos procesados
│   └── 3_gold/                      # Datos finales listos para ML
│
├── results/                         # Resultados de entrenamientos
│   ├── option1/                     # Resultados de estrategia 1
│   │   ├── ddpg/                    # Modelo DDPG
│   │   └── ppo/                     # Modelo PPO
│   ├── option2/                     # Resultados de estrategia 2
│   │   ├── ddpg/                    # Modelo DDPG
│   │   └── ppo/                     # Modelo PPO
│   └── poc/                         # Pruebas de concepto
│       ├── resultados_img_v1/       # Visualizaciones versión 1
│       └── resultados_img_v2/       # Visualizaciones versión 2
│
├── notebooks/                       # Jupyter notebooks (análisis exploratorio)
├── docs/                            # Documentación
├── paper1/                          # Artículo/reporte del proyecto
└── ref_code/                        # Código de referencia
```

## 🔄 Pipeline de Datos

### 1️⃣ Extracción de Datos (`01_load_data.py`)

El script carga datos de múltiples fuentes:

#### Mercado Eléctrico (XM)
- `DEMANDA`: Demanda eléctrica por horas (0-7, 7-17, 17-23)
- `PRECIOS`: Precios spot por horas
- `PRECIOS_PONDERADOS`: Precios ponderados del mercado
- `PRECIOS_BILATERALES`: Transacciones bilaterales
- `APORTES_HIDRICOS`: Energía hidráulica disponible
- `DISPONIBILIDAD_REAL`: Capacidad de generación disponible
- `GENERACION_REAL`: Generación real (térmica, hidráulica, solar)

#### Mercado de Futuros
- Contratos de futuros de energía: **ELM**, **MTB**, **DTB**, **NTB**
- Múltiples vencimientos (0 a 6 meses)

#### Noticias
- Google News sobre energía en Colombia
- Clasificación por tipo (bullish/bearish)

**Funciones principales:**
- `limpiar_fuente_en_titulo()`: Limpia metadatos de títulos
- `normalize_text()`: Normaliza texto para análisis NLP
- `process_news()`: Procesa y etiqueta noticias
- `load_data_files()`: Orquesta carga de todos los datos

**Rango de datos:** 2022-01-01 hasta 2026-01-31

### 2️⃣ Procesamiento de Datos (`02_clean_data.py`)

El script realiza transformaciones exhaustivas:

#### Limpieza de Outliers
- Detección basada en **mediana móvil (7 días)**
- Umbral de anomalía: **3 desviaciones estándar**
- Corrección: Interpolación lineal de anomalías
- Se aplica a: Precios spot, precios ponderados, aportes hídricos

#### Feature Engineering
- **Ratios de Cobertura**: `Disponibilidad / Demanda` por franja horaria
- **Medias Móviles**: `AportesHidricos_GWh_MA7`
- **Retornos Logarítmicos**: `ln(Precio_t / Precio_t-1)` para análisis de volatilidad
- **Beta Móvil (30 días)**: Relación entre futuros y spot por contrato
  - Contratos ELM: Beta vs Precio Ponderado
  - Contratos MTB: Beta vs Precio 0-7
  - Contratos DTB: Beta vs Precio 7-17
  - Contratos NTB: Beta vs Precio 17-23
- **Base de Futuros**: `Precio_Futuro - Precio_Spot`
- **Demanda Adelantada**: Proyecciones 1-6 meses (shift de demanda mensual)

#### Simulación de Demanda
- Desagregación de demanda por comprador
- Genera columnas: `Demanda_Comprador_Dia_01Meses_Adelante`, etc.

#### Procesamiento de Noticias
- Filtra por relevancia
- Aplica media móvil de 30 días a polaridad
- Completa fechas faltantes con forward fill

### 3️⃣ Datasets Finales (Carpeta `data/3_gold`)

Se generan **4 datasets específicos** por tipo de contrato de futuros:

| Archivo | Contrato | Rango Horario |
|---------|----------|--------------|
| `dataset_SISTEMA_ELM.csv` | ELM | Full Day (24h) |
| `dataset_SISTEMA_MTB.csv` | MTB | Madrugada (0-7) |
| `dataset_SISTEMA_DTB.csv` | DTB | Día (7-17) |
| `dataset_SISTEMA_NTB.csv` | NTB | Noche (17-23) |

**Variables incluidas en cada dataset:**
- Fechas y variables temporales
- Precios spot (ponderado y por franja)
- Precios de futuros (0-6 meses)
- Base de futuros y beta móvil
- Demanda (actual y adelantada)
- Generación (térmica, hidráulica, solar)
- Aportes hídricos (valor y media móvil)
- Disponibilidad real
- Sentimiento de noticias
- Ratios de cobertura
- Retornos logarítmicos

**Archivos adicionales:**
- `precios_LIQUIDACION.csv`: Precios promedios mensuales
- `fechas_transacciones.csv`: Fechas de transacciones activas
- `demanda_COMPRADOR.csv`: Demanda desagregada por comprador

## 🤖 Modelos de Trading

### Estrategias Exploradas

#### **Option 1 & 2**: Agentes con Aprendizaje por Refuerzo
- **PPO (Proximal Policy Optimization)**: 
  - Algoritmo de política mejorado
  - Estable y robusto para trading
  - Óptimo para espacios de acción continuos/discretos
  
- **DDPG (Deep Deterministic Policy Gradient)**:
  - Actor-Crítico determinista
  - Excelente para control continuo
  - Manejo eficiente de acciones continuas

### Ambiente del Agente

El agente RL interactúa con el mercado diariamente:

- **Estados**: Precios, demanda, generación, noticias, betas, ratios
- **Acciones**: Posiciones en contratos de futuros 
  - Compra (Long)
  - Venta (Short)
  - Posición Neutra
- **Recompensa**: Utilidad ajustada por riesgo, penalizaciones por exposición
- **Horizonte**: revisión diario en mercado de futuros de energía

### Variables del Agente

**Entrada (Observaciones):**
```
[Precio_Spot, Precio_Futuro, Demanda, Generacion, 
 Beta_Movil, Base_Futuros, AportesHidricos, 
 Sentimiento_Noticias, RatioCobertura, Retornos_Historicos, ...]
```

**Salida (Acciones):**
```
Posicion ∈ {-1 (Short), 0 (Neutro), 1 (Long)}
```

## 📊 Requisitos

El proyecto requiere librerías especializadas en:

### Dependencias Principales

```
# Deep Learning & RL
keras==3.11.3
tensorflow==2.20.0
torch==2.9.1
torchvision==0.24.1
gymnasium==1.2.2
tensorboard==2.20.0

# Data Science
numpy==2.3.3
pandas==2.3.3
scipy==1.16.2
scikit-learn==1.7.2

# Time Series Forecasting
statsmodels==0.14.6
pmdarima==2.1.1
prophet==1.3.0
skforecast==0.20.1

# NLP
spacy==3.8.11
gnews==0.4.3

# Data Sources
pydataxm==0.3.16
sodapy==2.2.0

# Visualization
matplotlib==3.10.6
seaborn==0.13.2
plotly==6.5.2

# Utilities
pytz==2025.2
openpyxl==3.1.5
xlrd==2.0.2
```

### Instalación

```bash
cd src
pip install -r requirements.txt
```

Nota: El modelo spaCy español se instala automáticamente desde el wheel incluido en `requirements.txt`.

## 🚀 Uso

### Paso 1: Preparación del Entorno

```bash
# Clonar repositorio
git clone https://github.com/shaka03/bot_futuros.git
cd bot_futuros

# Crear entorno virtual (recomendado)
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar dependencias
cd src
pip install -r requirements.txt
```

### Paso 2: Carga de Datos

```bash
cd src
python 01_load_data.py
```

**Salida esperada:**
- Datos en `data/2_silver/` (datos procesados)
- Archivos CSV con datos de mercado eléctrico y futuros

### Paso 3: Procesamiento y Feature Engineering

```bash
python 02_clean_data.py
```

**Salida esperada:**
- Datasets finales en `data/3_gold/` (listos para ML)
- 4 archivos principales: `dataset_SISTEMA_ELM.csv`, etc.
- Archivos auxiliares: precios de liquidación, fechas de transacciones

### Paso 4: Entrenamiento de Agentes

```bash
# Entrenar agentes en src/agents/
python agents/option1/train_ppo.py    # Entrenar PPO opción 1
python agents/option1/train_ddpg.py   # Entrenar DDPG opción 1

# O en option2
python agents/option2/train_ppo.py
python agents/option2/train_ddpg.py
```

### Paso 5: Evaluación y Análisis

```bash
# Ejecutar pruebas de concepto
python poc/copilot/analyze.py
python poc/gemini1/predict.py

# Ver resultados en results/
# Visualizaciones en results/poc/resultados_img_v1/ y v2/
```

## 📈 Resultados

Los resultados se almacenan en la carpeta `results/`:

### Estructura de Resultados
```
results/
├── option1/
│   ├── ddpg/          # Modelos, pesos y métricas DDPG
│   └── ppo/           # Modelos, pesos y métricas PPO
├── option2/
│   ├── ddpg/          # Modelos, pesos y métricas DDPG
│   └── ppo/           # Modelos, pesos y métricas PPO
└── poc/
    ├── resultados_img_v1/  # Gráficos v1 (Sharpe, retornos, etc.)
    └── resultados_img_v2/  # Gráficos v2 (comparativas mejoradas)
```

### Métricas Evaluadas

- **Retorno Acumulado**: Ganancia total del período
- **Sharpe Ratio**: Retorno ajustado por volatilidad
- **Maximum Drawdown**: Pérdida máxima desde pico
- **Win Rate**: Porcentaje de días ganadores
- **Profit Factor**: Ganancia/Pérdida total
- **Volatilidad**: Desviación estándar de retornos

## 🔧 Tecnologías Principales

| Componente | Tecnología | Propósito |
|-----------|-----------|----------|
| **Data Sources** | XM (pydataxm), Google News | Obtener datos reales |
| **Data Processing** | Pandas, NumPy, SciPy | Limpieza y transformación |
| **Feature Engineering** | statsmodels, custom | Ingeniería de características |
| **ML/RL** | TensorFlow, PyTorch, Gymnasium | Entrenar agentes |
| **Time Series** | statsmodels, Prophet, skforecast | Pronósticos |
| **NLP** | spaCy (Spanish) | Análisis de noticias |
| **Visualization** | Plotly, Matplotlib, Seaborn | Reportes y gráficos |

## 📝 Notas Importantes

### 1. **Datos Reales**
El proyecto utiliza datos del **Mercado Eléctrico Colombiano (XM)**:
- Precios horarios del mercado mayorista
- Demanda de energía
- Generación real de plantas
- Transacciones bilaterales

### 2. **Series de Tiempo**
- Datos históricos: **2022-01-01 a 2026-01-31**
- Frecuencia: **Diaria**
- Granularidad: **Horaria** (agregada a diaria)

### 3. **Demanda Simulada**
- Se desagrega la demanda total por comprador
- Simula comportamiento realista usando `simulation_demand.py`
- Permite proyecciones adelantadas

### 4. **Procesamiento de Noticias**
- Se etiquetan automáticamente por polaridad
- Utiliza NLP con spaCy (modelo español)
- Media móvil de 30 días para suavizar

### 5. **Contratos de Futuros**
- **ELM**: Contrato Full Day (todo el día)
- **MTB**: Contrato Madrugada (0-7 horas)
- **DTB**: Contrato Día (7-17 horas)
- **NTB**: Contrato Noche (17-23 horas)

### 6. **Validación Temporal**
- Desplazamiento de 1 día en datasets finales para evitar *look-ahead bias*
- Información histórica únicamente (sin datos futuros)

## 📚 Referencias

- **Mercado Eléctrico Colombiano**: https://www.xm.com.co/
- **Gymnasium**: Environment Toolkit para RL - https://gymnasium.farama.org/
- **scikit-learn**: ML Library - https://scikit-learn.org/
- **statsmodels**: Time Series - https://www.statsmodels.org/
- **Prophet**: Facebook's Forecasting - https://facebook.github.io/prophet/
- **spaCy**: NLP Library - https://spacy.io/

## 🤝 Contribuciones

Para contribuir al proyecto:

1. Fork el repositorio
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## ⚠️ Disclaimer

Este software es una prueba de concepto académica para la aplicación de Deep Reinforcement Learning en mercados energéticos. No constituye una recomendación de inversión financiera. Los mercados de futuros de energía son altamente volátiles y riesgosos.

Proyecto de investigación para la **Maestría en Analítica e Inteligencia de Negocios**.

---

**Última actualización**: Enero 2026

**Estado**: En Desarrollo 🚧
