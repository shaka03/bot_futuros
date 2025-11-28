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
