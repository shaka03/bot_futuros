# Estructura de la Carpeta de Datos (Data Pipeline)

Esta carpeta organiza el flujo de información para el entrenamiento y evaluación del agente de Deep Reinforcement Learning.

---

## 📂 1. raw
**Propósito:** Almacenamiento de datos fuente sin ninguna modificación.
* **Precios de Futuros:** Archivos `.csv`, `.xls` y `.xlsx` con los precios de cierre de los contratos de energía y variables de entorno.
* **Variables de Estado:** Datos físicos y de mercado (aportes, demanda, niveles de embalse) que servirán como inputs para que el agente observe el entorno.
* **Estado:** Inmutable. No se deben editar estos archivos directamente.

---

## 📂 2. silver
**Propósito:** Datos limpios y transformados listos para el consumo del modelo.
* **Procesamiento:** Incluye la limpieza de nulos y el alineamiento temporal (joins) entre precios y variables de estado.
* **Uso:** Esta es la carpeta desde la cual el entorno (Environment) cargará los datos para el agente.

---

## 📂 3. gold
**Propósito:** Resultados finales y productos de valor.
* **Salidas del Agente:** Logs de las acciones tomadas por el agente, evolución de la recompensa (reward) y métricas de desempeño del portafolio.
* **Backtesting:** Resultados de las simulaciones finales comparadas con el benchmark del mercado.

---

> **Nota:** El flujo de datos debe ser siempre unidireccional: **Raw → Silver → Gold**.

---

# Diccionario de Datos - agente de cobertura con futuros de electricidad

Este documento proporciona una descripción detallada de los datasets relacionados con el mercado de energía eléctrica, incluyendo aportes hídricos, demanda, disponibilidad, generación y precios, que están en la carpeta 📂 **silver**.

---

## 1. datos_APORTES_HIDRICOS.csv
**Descripción:** Contiene los registros diarios de los aportes hídricos al sistema, medidos en energía (kWh).

| Columna | Descripción | Tipo de Variable |
| :--- | :--- | :--- |
| Fecha | Fecha del registro (YYYY-MM-DD). | Temporal (Date) |
| AportesHidricos_kWh | Cantidad de energía aportada por recursos hídricos en kWh. | Numérica (Float) |

---

## 2. datos_DEMANDA.csv
**Descripción:** Registra la demanda de energía eléctrica desglosada por bloques horarios y el total diario.

| Columna | Descripción | Tipo de Variable |
| :--- | :--- | :--- |
| Fecha | Fecha del registro (YYYY-MM-DD). | Temporal (Date) |
| Demanda_kWh_0-7 | Demanda de energía en el bloque de 00:00 a 07:00 (kWh). | Numérica (Float) |
| Demanda_kWh_7-17 | Demanda de energía en el bloque de 07:00 a 17:00 (kWh). | Numérica (Float) |
| Demanda_kWh_17-23 | Demanda de energía en el bloque de 17:00 a 24:00 (kWh). | Numérica (Float) |
| Demanda_kWh_Dia | Demanda total de energía en el día (kWh). | Numérica (Float) |

---

## 3. datos_DEMANDA_COMPRADOR.csv
**Descripción:** Registra la demanda de energía eléctrica desglosada por bloques horarios y el total diario, para un comprador no regulado de electricidad.

| Columna | Descripción | Tipo de Variable |
| :--- | :--- | :--- |
| Fecha | Fecha del registro (YYYY-MM-DD). | Temporal (Date) |
| Demanda_kWh_0-7 | Demanda de energía en el bloque de 00:00 a 07:00 (kWh). | Numérica (Float) |
| Demanda_kWh_7-17 | Demanda de energía en el bloque de 07:00 a 17:00 (kWh). | Numérica (Float) |
| Demanda_kWh_17-23 | Demanda de energía en el bloque de 17:00 a 24:00 (kWh). | Numérica (Float) |
| Demanda_kWh_Dia | Demanda total de energía en el día (kWh). | Numérica (Float) |

---

## 4. datos_DISPONIBILIDAD_REAL.csv
**Descripción:** Indica la disponibilidad real de energía en el sistema por bloques horarios.

| Columna | Descripción | Tipo de Variable |
| :--- | :--- | :--- |
| Fecha | Fecha del registro (YYYY-MM-DD). | Temporal (Date) |
| Disponibilidad_kWh_0-7 | Energía disponible en el bloque de 00:00 a 07:00 (kWh). | Numérica (Float) |
| Disponibilidad_kWh_7-17 | Energía disponible en el bloque de 07:00 a 17:00 (kWh). | Numérica (Float) |
| Disponibilidad_kWh_17-23 | Energía disponible en el bloque de 17:00 a 24:00 (kWh). | Numérica (Float) |
| Disponibilidad_kWh_Dia | Energía total disponible en el día (kWh). | Numérica (Float) |

---

## 5. datos_GENERACION_REAL.csv
**Descripción:** Detalla la generación real de energía según la fuente de origen.

| Columna | Descripción | Tipo de Variable |
| :--- | :--- | :--- |
| Fecha | Fecha del registro (YYYY-MM-DD). | Temporal (Date) |
| Generacion_Cogenerador_kWh | Energía generada por cogeneradores (kWh). | Numérica (Float) |
| Generacion_Eolica_kWh | Energía generada por fuente eólica (kWh). | Numérica (Float) |
| Generacion_Hidraulica_kWh | Energía generada por fuente hidráulica (kWh). | Numérica (Float) |
| Generacion_Solar_kWh | Energía generada por fuente solar (kWh). | Numérica (Float) |
| Generacion_Termica_kWh | Energía generada por fuente térmica (kWh). | Numérica (Float) |
| Generacion_Total_kWh | Generación total de energía en el día (kWh). | Numérica (Float) |

---

## 6. datos_NIVELES_EMBALSE.csv
**Descripción:** Registro diario del nivel de energía almacenada en los embalses.

| Columna | Descripción | Tipo de Variable |
| :--- | :--- | :--- |
| Fecha | Fecha del registro (YYYY-MM-DD). | Temporal (Date) |
| NivelEmbalse | Nivel de energía almacenada en los embalses (kWh). | Numérica (Float) |

---

## 7. datos_PRECIOS.csv
**Descripción:** Precios de la energía eléctrica en la bolsa, desglosados por bloques horarios.

| Columna | Descripción | Tipo de Variable |
| :--- | :--- | :--- |
| Fecha | Fecha del registro (YYYY-MM-DD). | Temporal (Date) |
| Precio_COP/kWh_0-7 | Precio de la energía en el bloque 0-7 (COP/kWh). | Numérica (Float) |
| Precio_COP/kWh_7-17 | Precio de la energía en el bloque 7-17 (COP/kWh). | Numérica (Float) |
| Precio_COP/kWh_17-23 | Precio de la energía en el bloque 17-23 (COP/kWh). | Numérica (Float) |
| Precio_COP/kWh_Dia | Precio promedio diario de la energía (COP/kWh). | Numérica (Float) |

---

## 8. datos_PRECIOS_PONDERADOS.csv
**Descripción:** Precio promedio ponderado de la energía en el mercado.

| Columna | Descripción | Tipo de Variable |
| :--- | :--- | :--- |
| Fecha | Fecha del registro (YYYY-MM-DD). | Temporal (Date) |
| Precio_Ponderado_COP/kWh | Precio promedio ponderado diario (COP/kWh). | Numérica (Float) |

---

## 9. precios_FUTUROS.csv
**Descripción:** Cotizaciones de los contratos de futuros de energía eléctrica.

| Columna | Descripción | Tipo de Variable |
| :--- | :--- | :--- |
| Nemotecnico | Código identificador del contrato de futuro. | Categórica (String) |
| Tipo | Tipo de contrato o mercado. | Categórica (String) |
| Mes | Mes de vencimiento del contrato. | Categórica (String) |
| Año | Año de vencimiento del contrato. | Numérica (Integer) |
| Fecha | Fecha de la cotización (YYYY-MM-DD). | Temporal (Date) |
| FechaVencimientoContrato | Fecha de de vencimiento del comtrato (YYYY-MM-DD). | Temporal (Date) |
| Precio | Precio de cierre del contrato de futuro (COP/kWh). | Numérica (Float) |