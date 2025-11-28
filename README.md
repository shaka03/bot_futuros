# Agente de Cobertura de Futuros de Energía en Colombia

Este repositorio contiene una implementación modular de **Multi-Agent Deep Deterministic Policy Gradient (MADDPG)** con la técnica **Dual Q (Twin Critic)**, diseñada específicamente para la gestión de riesgo y cobertura (hedging) de contratos de futuros de energía en el mercado colombiano. Este trabajo se propone para el trabajo integrador de la Maestría en Analítica e Inteligencia de Negocios (MAIN) de la Universidad del Valle.

El sistema entrena múltiples agentes autónomos (uno por cada tipo de contrato: ELM, ELS, MTB, DTB, NTB) que aprenden a cooperar o actuar individualmente para maximizar el PnL ajustado al riesgo, gestionando decisiones de posición y *roll-over*.

## 🚀 Características Principales

* **Arquitectura Dual Q (Twin Critic):** Implementación de dos redes críticas para reducir la sobreestimación de los valores Q (inspirado en TD3), mejorando la estabilidad del aprendizaje.
* **Gestión de Roll-Over:** Los agentes deciden autónomamente si mantienen la posición hasta el vencimiento (liquidación financiera) o hacen *roll-over* al siguiente vencimiento (M2) basándose en los días restantes (DTM) y costos de spread.
* **Curva de Futuros Continua:** Procesamiento de datos inteligente que construye curvas "Front Month" (M1) y "Next Month" (M2) para evitar discontinuidades en el aprendizaje.
* **Prevención de Leakage:** División estricta de Train/Test y validación temporal para asegurar que el agente solo actúe con información disponible en el tiempo $t$.
* **Replay Buffer Optimizado:** Implementación de memoria de experiencias basada en `numpy` pre-alojado para máxima eficiencia computacional.

## 📂 Estructura del Proyecto

El código está diseñado de forma modular:

| Archivo | Descripción |
| :--- | :--- |
| `config.py` | Hiperparámetros globales (Tasas de aprendizaje, costos de transacción, aversión al riesgo). |
| `data_processor.py` | ETL de datos: Alineación de precios Spot y Futuros, y cálculo de días al vencimiento. |
| `environment.py` | Entorno compatible con Gym. Simula el mercado, calcula PnL, costos y recompensas. |
| `networks.py` | Definición de redes neuronales (Actor y Críticos Duales) en PyTorch. |
| `maddpg.py` | Lógica del algoritmo de aprendizaje, actualización de gradientes y *Soft Updates*. |
| `buffer.py` | Gestión eficiente de la memoria de repetición (Replay Buffer). |
| `visualization.py` | Módulo de análisis gráfico para comparar PnL y comportamiento de los agentes. |
| `main.py` | Script principal: Carga datos, entrena el modelo y ejecuta el Backtest. |

## ⚙️ Instalación y Requisitos

Se requiere **Python 3.8+**. Instala las dependencias necesarias:

```bash
pip install torch pandas numpy gymnasium matplotlib seaborn
```

## 📊 Datos Requeridos

El sistema espera dos archivos CSV en el directorio raíz:

datos_energia.csv: Precios de bolsa (Spot).

Columnas clave: FechaHora, Valor (Precio), CodigoVariable (debe contener 'PB_Nal').

datos_futuros.csv: Precios de cierre de contratos derivados.

Columnas clave: Fecha, Tipo (ELM, ELS...), Mes, Año, Precio.

## 🧠 Lógica del Agente

### **Espacio de Estado (Observación)**

Cada agente recibe un vector de 5 dimensiones:

1. Precio Spot: Precio actual en bolsa.
2. Precio Futuro M1: Contrato con vencimiento más cercano.
3. Precio Futuro M2: Siguiente contrato (para referencia de roll).
4. DTM (Days to Maturity): Días restantes para el vencimiento de M1.
5. Posición Actual: Inventario actual del agente.

### **Espacio de Acción**

El agente emite 2 valores continuos `[-1, 1]`:

1. Hedge Ratio Delta: Cuánto cambiar la posición actual (Compra/Venta).
2. Roll Decision: Probabilidad de ejecutar un roll-over anticipado si DTM < 5 días.

### **Función de Recompensa**

El objetivo es maximizar el PnL penalizando la volatilidad (riesgo):

$$R = PnL_{diario} - \lambda \cdot (PnL_{diario})^2 - Costos_{transaccion}$$

Donde $\lambda$ es el factor de aversión al riesgo definido en `config.py`.

## ▶️ Ejecución

Para iniciar el entrenamiento y posterior validación:

```Bash
python main.py
```

El script realizará lo siguiente:

1. Procesará los CSV.
2. Entrenará los agentes durante el número de episodios definidos en Config.
3. Ejecutará un Backtest sobre el 20% final de los datos (nunca vistos).
4. Generará gráficos de resultados en la carpeta `resultados_img/`.

## 📈 Resultados y Visualización

Al finalizar, revisa la carpeta resultados_img/ para ver:

* `pnl_comparison.png`: Curva de equidad comparando Dual Q Agent vs Benchmark (Buy & Hold).
* `behavior_{CONTRACT}.png`: Gráfico de doble eje que muestra cómo el agente ajusta su Hedge Ratio en respuesta a los movimientos del Precio Spot.
* `scatter_{CONTRACT}.png`: Mapa de dispersión para analizar la correlación entre precio y posición.

## ⚠️ Disclaimer

Este software es una prueba de concepto académica para la aplicación de Deep Reinforcement Learning en mercados energéticos. No constituye una recomendación de inversión financiera. Los mercados de futuros de energía son altamente volátiles y riesgosos.
