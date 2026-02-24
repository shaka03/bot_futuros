# Estructura de la Carpeta de Datos (Data Pipeline)

Esta carpeta organiza el flujo de información para el entrenamiento y evaluación del agente de Deep Reinforcement Learning.

---

## 📂 1_raw
**Propósito:** Almacenamiento de datos fuente sin ninguna modificación.
* **Precios de Futuros:** Archivos `.csv`, `.xls` y `.xlsx` con los precios de cierre de los contratos de energía y variables de entorno.
* **Variables de Estado:** Datos físicos y de mercado (aportes, demanda, niveles de embalse) que servirán como inputs para que el agente observe el entorno.
* **Estado:** Inmutable. No se deben editar estos archivos directamente.

---

## 📂 2_silver
**Propósito:** Datos limpios y transformados para procesamiento de lo datos.
* **Procesamiento:** Incluye la limpieza de nulos y el alineamiento temporal entre precios y variables de estado.
* **Uso:** Esta es la carpeta donde se tienen los datos pre-procesados.

---

## 📂 3_gold
**Propósito:** datos limpios y procesados listos para el consumo del agente.
* **Procesamiento:** Incluye limpieza de anomalías, creación de nuevas variables y join de los precios y las variables de estado.
* **Uso:** Esta es la carpeta donde se almacenará el dataset limpio y listo para que el agente tome los datos.

---

## 📂 4_platinum
**Propósito:** Resultados finales y productos de valor.
* **Salidas del Agente:** Logs de las acciones tomadas por el agente, evolución de la recompensa (reward) y métricas de desempeño del portafolio.
* **Backtesting:** Resultados de las simulaciones finales comparadas con el benchmark del mercado.

---

> **Nota:** El flujo de datos debe ser siempre unidireccional: **1_raw → 2_silver → 3_gold → 4_platinum**.

---

# Diccionario de Datos - 2_silver - agente de cobertura con futuros de electricidad

Este documento proporciona una descripción detallada de los datasets relacionados con el mercado de energía eléctrica, incluyendo aportes hídricos, demanda, disponibilidad, generación y precios, que están en la carpeta 📂 **2_silver**.

---

## 1. datos_APORTES_HIDRICOS.csv
**Descripción:** Contiene los registros diarios de los aportes hídricos al sistema, medidos en energía (kWh).

| Columna | Descripción | Tipo de Variable |
| :--- | :--- | :--- |
| Fecha | Fecha del registro (YYYY-MM-DD). | Temporal (Date) |
| AportesHidricos_GWh | Cantidad de energía aportada por recursos hídricos en GWh. | Numérica (Float) |

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

## 6. datos_PRECIOS.csv
**Descripción:** Precios de la energía eléctrica en la bolsa, desglosados por bloques horarios.

| Columna | Descripción | Tipo de Variable |
| :--- | :--- | :--- |
| Fecha | Fecha del registro (YYYY-MM-DD). | Temporal (Date) |
| Precio_COP/kWh_0-7 | Precio de la energía en el bloque 0-7 (COP/kWh). | Numérica (Float) |
| Precio_COP/kWh_7-17 | Precio de la energía en el bloque 7-17 (COP/kWh). | Numérica (Float) |
| Precio_COP/kWh_17-23 | Precio de la energía en el bloque 17-23 (COP/kWh). | Numérica (Float) |
| Precio_COP/kWh_Dia | Precio promedio diario de la energía (COP/kWh). | Numérica (Float) |

---

## 7. datos_PRECIOS_PONDERADOS.csv
**Descripción:** Precio promedio ponderado de la energía en el mercado.

| Columna | Descripción | Tipo de Variable |
| :--- | :--- | :--- |
| Fecha | Fecha del registro (YYYY-MM-DD). | Temporal (Date) |
| Precio_Ponderado_COP/kWh | Precio promedio ponderado diario (COP/kWh). | Numérica (Float) |

---

## 8. precios_FUTUROS.csv
**Descripción:** Precios de cierre de los contratos de futuros de energía eléctrica.

| Columna | Descripción | Tipo de Variable |
| :--- | :--- | :--- |
| Nemotecnico | Código identificador del contrato de futuro. | Categórica (String) |
| Tipo | Tipo de contrato o mercado. | Categórica (String) |
| Mes | Mes de vencimiento del contrato. | Categórica (String) |
| Año | Año de vencimiento del contrato. | Numérica (Integer) |
| Fecha | Fecha de la cotización (YYYY-MM-DD). | Temporal (Date) |
| FechaVencimientoContrato | Fecha de de vencimiento del comtrato (YYYY-MM-DD). | Temporal (Date) |
| Precio | Precio de cierre del contrato de futuro (COP/kWh). | Numérica (Float) |

---

## 9. datos_NOTICIAS.csv
**Descripción:** Registro de noticias del sector energético recolectadas de diversas fuentes, incluyendo el texto normalizado y una categorización numérica.

| Columna | Descripción | Tipo de Variable |
| :--- | :--- | :--- |
| Titulo | Título original de la noticia tal como aparece en la fuente. | Categórica (String) |
| Fuente | Nombre del medio de comunicación o portal que publica la noticia. | Categórica (String) |
| Fecha | Fecha de publicación del registro. | Temporal (Date) |
| Hora | Hora exacta de publicación de la noticia. | Temporal (Time) |
| url | Enlace directo a la fuente original de la información. | Categórica (String) |
| Titulo_norm | Título procesado (limpieza de caracteres, minúsculas) para análisis de texto. | Categórica (String) |
| Tipo_noticia | Valor numérico que representa la clasificación o el sentimiento de la noticia. | Numérica (Float) |

---

# 10. datos_PRECIOS_BILATERALES.csv
**Descripción:** Información sobre los precios de energía pactados en contratos bilaterales, segmentados por bloques de horas para el mercado colombiano.

| Columna | Descripción | Tipo de Variable |
| :--- | :--- | :--- |
| Fecha | Fecha a la que corresponde el registro de precios. | Temporal (Date) |
| Precio_Bilateral_COP/kWh_0-7 | Precio promedio en pesos por kilovatio hora para el bloque de 00:00 a 07:00. | Numérica (Float) |
| Precio_Bilateral_COP/kWh_7-17 | Precio promedio en pesos por kilovatio hora para el bloque de 07:00 a 17:00. | Numérica (Float) |
| Precio_Bilateral_COP/kWh_17-23 | Precio promedio en pesos por kilovatio hora para el bloque de 17:00 a 23:00. | Numérica (Float) |
| Precio_Bilateral_COP/kWh_Dia | Precio promedio ponderado total para el día completo. | Numérica (Float) |

---

## 11. precios_FUTUROS_WIDE.csv
**Descripción:** Matriz de precios de cierre para contratos de futuros de energía eléctrica (series DTB, ELM, MTB, NTB) con diferentes horizontes de vencimiento.

| Columna | Descripción | Tipo de Variable |
| :--- | :--- | :--- |
| Fecha | Fecha de negociación o valoración de los contratos. | Temporal (Date) |
| DTB_Vencimiento_XXMeses | Precios de futuros serie DTB con vencimiento a XX meses (0 a 23 meses). | Numérica (Float) |
| ELM_Vencimiento_XXMeses | Precios de futuros serie ELM con vencimiento a XX meses (0 a 71 meses). | Numérica (Float) |
| MTB_Vencimiento_XXMeses | Precios de futuros serie MTB con vencimiento a XX meses (0 a 23 meses). | Numérica (Float) |
| NTB_Vencimiento_XXMeses | Precios de futuros serie NTB con vencimiento a XX meses (0 a 23 meses). | Numérica (Float) |

---

# Diccionario de Datos - 3_gold - agente de cobertura con futuros de electricidad

Este documento proporciona una descripción detallada del dataset limpio, con las variables de precios y de estado, que están en la carpeta 📂 **3_gold**.

---

## 1. dataset_SISTEMA.csv
**Descripción:** Archivo consolidado que integra variables del mercado spot, fundamentales del sistema (demanda/generación) e indicadores financieros.

| Columna | Descripción | Tipo de Variable |
| :--- | :--- | :--- |
| Fecha | Fecha de operación del sistema. | Temporal (Date) |
| [SERIE]_Vencimiento_00-06Meses | Precios de futuros de corto plazo (ELM, MTB, DTB, NTB). | Numérica (Float) |
| Precio_COP/kWh_[Bloque] | Precios spot de energía en bolsa por bloques horarios. | Numérica (Float) |
| Demanda_kWh_[Bloque] | Demanda de energía eléctrica por bloques y total diario. | Numérica (Float) |
| AportesHidricos_GWh | Energía que ingresa a los embalses por caudales. | Numérica (Float) |
| Generacion_Termica_kWh | Energía producida por plantas térmicas. | Numérica (Float) |
| Generacion_Hidraulica_kWh | Energía producida por centrales hidroeléctricas. | Numérica (Float) |
| Disponibilidad_kWh_[Bloque] | Oferta total disponible en el sistema. | Numérica (Float) |
| Ratio_Cobertura_[Bloque] | Relación Disponibilidad / Demanda. | Numérica (Float) |
| Retorno_[Variable] | Variación porcentual (logarítmica) de precios y futuros. | Numérica (Float) |
| Base_[SERIE]_Vencimiento_XX | Basis: Diferencia entre el precio spot y el futuro. | Numérica (Float) |
| Beta_MA30_[SERIE]_Vencimiento | Sensibilidad (Beta) del contrato con media móvil de 30 días. | Numérica (Float) |
			
---

## 2. datos_DEMANDA_COMPRADOR.csv
**Descripción:** Registra la demanda de energía eléctrica de un comprador específico del mercado, discriminada por franjas horarias.

| Columna | Descripción | Tipo de Variable |
| :--- | :--- | :--- |
| Fecha | Fecha del registro (YYYY-MM-DD). | Temporal (Date) |
| Demanda_kWh_0-7 | Demanda de energía del comprador en la franja 0-7. | Numérica (Float) |
| Demanda_kWh_7-17 | Demanda de energía del comprador en la franja 7-17. | Numérica (Float) |
| Demanda_kWh_17-23 | Demanda de energía del comprador en la franja 17-23. | Numérica (Float) |
| Demanda_kWh_Dia | Demanda total de energía del comprador en el día. | Numérica (Float) |

---

## 3. fechas_transacciones.csv
**Descripción:** Corresponde a las fechas en las que hacen las transacciones en el mercado de futuros de electricidad.

| Columna | Descripción | Tipo de Variable |
| :--- | :--- | :--- |
| Fecha | Fecha de la transacción (YYYY-MM-DD). | Temporal (Date) |

---

## 4. precios_FUTUROS.csv
**Descripción:** Precios de cierre de los contratos de futuros de energía eléctrica negociados en el mercado.

| Columna | Descripción | Tipo de Variable |
| :--- | :--- | :--- |
| Nemotecnico | Código identificador del contrato de futuro. | Categórica (String) |
| Tipo | Tipo de contrato o mercado (ej. ELM). | Categórica (String) |
| Mes | Mes de vencimiento del contrato. | Categórica (String) |
| Año | Año de vencimiento del contrato. | Numérica (Integer) |
| Fecha | Fecha de la cotización (YYYY-MM-DD). | Temporal (Date) |
| FechaVencimientoContrato | Fecha de vencimiento del contrato (YYYY-MM-DD). | Temporal (Date) |
| Precio | Precio de cierre del contrato de futuro (COP/kWh). | Numérica (Float) |

---

## 5. datos_Precios.csv
**Descripción:** Precios de spot por franja horario en el mercado eléctrico colombiano.

| Columna | Descripción | Tipo de Variable |
| :--- | :--- | :--- |
| Fecha | Fecha del registro (YYYY-MM-DD). | Temporal (Date) |
| Precio_COP/kWh_0-7 | Precio de bolsa promedio en la franja horaria 0-7. | Numérica (Float) |
| Precio_COP/kWh_7-17 | Precio de bolsa promedio en la franja horaria 7-17. | Numérica (Float) |
| Precio_COP/kWh_17-23 | Precio de bolsa promedio en la franja horaria 17-23. | Numérica (Float) |
| Precio_COP/kWh_Dia | Precio de bolsa promedio aritmético del día. | Numérica (Float) |

---

## 6. precios_liquidacion.csv
**Descripción:** Son los precios en los cuales se liquidan los contratos futuros de electricidad.

| Columna | Descripción | Tipo de Variable |
| :--- | :--- | :--- |
| FechaVencimiento | Fecha de vencimiento (YYYY-MM-DD). | Temporal (Date) |
| Precio_COP/kWh_0-7 | Precio de liquidación en la franja horaria 0-7. | Numérica (Float) |
| Precio_COP/kWh_7-17 | Precio de liquidación en la franja horaria 7-17. | Numérica (Float) |
| Precio_COP/kWh_17-23 | Precio de liquidación en la franja horaria 17-23. | Numérica (Float) |
| Precio_COP/kWh_Dia | Precio de liquidación total día. | Numérica (Float) |