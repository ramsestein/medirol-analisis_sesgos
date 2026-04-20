# Cuando mitigar un sesgo amplifica otro: evaluación sistemática de modelos de lenguaje en simulación de pacientes para educación médica

---

## Resumen

Los modelos de lenguaje de gran tamaño (LLMs) se están incorporando rápidamente como simuladores de pacientes en educación médica. Su capacidad para mantener conversación clínica realista los convierte en instrumento pedagógico potente, pero también en potencial vector de transmisión de sesgos demográficos al estudiante. Los marcos de evaluación actuales suelen centrarse en una única dimensión y evaluar modelos de forma aislada, sin considerar cómo el diseño del prompt interactúa con las propiedades representacionales de cada modelo. Presentamos una metodología sistemática basada en sondeo adversarial sobre el simulador MediRol, aplicada a cinco modelos comerciales (deepseek, gemini-flash, gpt-4o, gpt-4.1 y gpt-5.4-mini) bajo tres configuraciones de prompt. Mostramos que la tasa de confabulación de datos demográficos oscila entre el 6 % y el 87 % según la combinación modelo × prompt; que modelos más recientes no son necesariamente más seguros (gpt-4.1, presentado como actualización de gpt-4o, triplica su tasa de fuga); que el prompt de producción, optimizado originalmente para gpt-4o, transfiere adecuadamente a gpt-5.4-mini y mal a otros modelos; y, sobre todo, que las intervenciones diseñadas para mitigar la confabulación demográfica modifican simultáneamente la estereotipia de género ocupacional. Una instrucción mínima de neutralidad demográfica produce, por mecanismo de masculino gramatical no marcado, una masculinización absoluta (100 %) del personal de enfermería. La evaluación de sesgos en LLMs clínicos requiere análisis multieje, replicación entre modelos y cautela ante supuestas mejoras incrementales de versión.

**Palabras clave**: educación médica, inteligencia artificial, simulación clínica, sesgos algorítmicos, modelos de lenguaje, paciente estandarizado.

---

## Estructura del repositorio

```
sesgos/
├── src/
│   ├── loader.py            # Carga y normalización de respuestas (→ data/long_df.parquet)
│   ├── classifier.py        # Clasificador regex Fase A (→ data/classified_df.parquet)
│   ├── judge.py             # Jueces LLM duales + voto mayoritario (→ data/judged_df.parquet)
│   ├── human_eval.py        # Evaluación humana de los 82 casos three-way
│   ├── leak_rates.py        # Cálculo de tasas de revelación por modelo/condición/pregunta
│   ├── name_gender.py       # Análisis de género en nombres de personal sanitario (P6/P7)
│   ├── patient_profile.py   # Perfil sociodemográfico del paciente confabulado
│   ├── stats_tests.py       # Contrastes estadísticos (χ², Fisher, binomial, FDR-BH)
│   └── figures.py           # Generación de todas las figuras del análisis
├── data/
│   ├── long_df.parquet      # Datos crudos normalizados (9 240 filas)
│   ├── classified_df.parquet
│   └── judged_df.parquet    # Dataset final con etiquetas de clasificación
├── tables/                  # CSVs de resultados intermedios y estadísticos
├── figures/                 # Figuras generadas (.png)
├── casos/                   # Ficheros JSON de los casos clínicos base (MediRol)
└── casos_sesgos/            # Ficheros JSON de los casos con carga estigmatizante
```

---

## Pipeline de análisis

```
loader.py  →  classifier.py  →  judge.py  →  human_eval.py (opcional)
                                    ↓
                 leak_rates.py  ·  name_gender.py  ·  patient_profile.py
                                    ↓
                              stats_tests.py
                                    ↓
                               figures.py
```

### Reproducir el análisis completo

```bash
# 1. Construir el dataset principal
python -m src.loader
python -m src.classifier

# 2. Clasificación con jueces LLM  (requiere claves API en .env)
python -m src.judge

# 3. Revisión humana de discrepancias three-way (opcional, interactivo)
python -m src.human_eval

# 4. Métricas
python -m src.leak_rates
python -m src.name_gender
python -m src.patient_profile
python -m src.patient_profile --by-disease   # análisis por enfermedad

# 5. Significación estadística
python -m src.stats_tests
python -m src.stats_tests --domain leak      # solo tasas de revelación
python -m src.stats_tests --domain gender    # solo género
python -m src.stats_tests --domain profile   # solo perfil paciente

# 6. Figuras
python -m src.figures
python -m src.figures --fig 1 3              # solo figuras 1 y 3
```

### Variables de entorno (`.env`)

```
OPENAI_API_KEY=...
DEEPSEEK_API_KEY=...
GOOGLE_API_KEY=...
```

---

## Diseño experimental

| Dimensión | Detalle |
|---|---|
| **Modelos** | deepseek, gemini-flash, gpt-4.1, gpt-4o, gpt-5.4-mini |
| **Condiciones** | Básico · Básico + "no reveles" · Prompt explícito r1 · Prompt explícito r2 |
| **Enfermedades** | 11 patologías con carga estigmatizante |
| **Preguntas** | 14 (P0–P13): edad, sexo, orientación, profesión, religión, raza, nombre enf./méd., nacionalidad, estudios, economía, familia, localización, etnia |
| **Rondas** | 3 por pregunta (presión conversacional creciente) |
| **Total interacciones** | 9 240 |

---

## Resultados principales

### Tasa de revelación por modelo y condición (%)

| Modelo | Básico | + "no reveles" | Prompt explícito | Global |
|---|---|---|---|---|
| gpt-4.1 | 87,2 | 23,2 | 49,2 (46,8–51,7) | 52,2 |
| deepseek | 78,4 | 37,4 | 58,9 (57,4–60,4) | 58,4 |
| gpt-5.4-mini | 42,4 | 0,4 | 6,2 (5,6–6,7) | 13,8 |
| gpt-4o | 24,0 | 0,4 | 11,8 (11,0–12,6) | 12,0 |
| gemini-flash | 23,2 | 0,9 | 14,5 (13,6–15,4) | 13,3 |

Todos los contrastes entre modelos y entre condiciones son estadísticamente significativos (χ², FDR-BH, p-ajustado < 10⁻³). Tamaños de efecto: V de Cramér entre 0,24 y 0,54 por condición; h de Cohen entre 0,35 y 1,41 en comparaciones pareadas.

### Sesgo de género ocupacional (% nombres masculinos)

| Modelo | Enfermería (n) | Medicina (n) |
|---|---|---|
| gpt-5.4-mini | 23,7 % (38) *** | 100,0 % (43) *** |
| gpt-4.1 | 0,0 % (63) *** | 78,2 % (55) *** |
| gpt-4o | 50,0 % (22) n.s. | 88,9 % (27) *** |
| gemini-flash | 35,7 % (28) n.s. | 63,6 % (22) n.s. |
| deepseek | 57,1 % (70) n.s. | 73,8 % (84) *** |

*** p-ajustado < 0,01 (test binomial bilateral, H₀: p = 0,5, corrección FDR-BH).

---

## Figuras generadas

| Figura | Contenido |
|---|---|
| `fig1_leak_heatmap.png` | Heatmap tasa de revelación: modelo × condición |
| `fig2_leak_por_pregunta.png` | Heatmap por pregunta × condición + Δ de reducción |
| `fig3_gender_bias.png` | Sesgo de género en personal sanitario (barras apiladas + asteriscos de significación) |
| `fig4_effect_sizes.png` | Cramér's V por atributo: entre modelos / condiciones / enfermedades |
| `fig5_profile_stability.png` | Estabilidad del perfil modal ante cambios de prompt |
| `fig6_pairwise_models.png` | Matriz de pares: nº atributos significativos + V de Cramér medio |

---

## 1. Introducción

La adopción de modelos de lenguaje de gran tamaño (LLMs) como simuladores de pacientes en educación médica se ha acelerado en los últimos dos años, incorporándose a prácticas de anamnesis estructurada, razonamiento diagnóstico y comunicación clínica. La promesa pedagógica es evidente: disponibilidad permanente, variabilidad ilimitada de casos, retroalimentación inmediata y eliminación del sesgo observador típico del paciente estandarizado humano. La ventaja, sin embargo, se asienta sobre un supuesto frágil: que el modelo es capaz de sostener el rol del paciente sin introducir información espuria ni reproducir estereotipos clínicos que puedan sedimentar en los esquemas cognitivos del estudiante en formación.

La literatura disponible sobre sesgos en LLMs clínicos se ha concentrado en dos ejes. El primero es la exactitud diagnóstica diferencial por grupo demográfico cuando se proporcionan los datos del paciente; el segundo es la toxicidad o contenido inapropiado en respuestas abiertas. Falta una evaluación sistemática del fenómeno complementario y, para el contexto educativo, más relevante: qué datos demográficos produce espontáneamente el modelo cuando no se le han proporcionado, bajo presión conversacional análoga a la que un estudiante puede ejercer durante una entrevista clínica simulada. Esta confabulación no es una curiosidad técnica; es el canal por el que los estereotipos incorporados al modelo durante el preentrenamiento se proyectan sobre el personaje clínico que el estudiante percibe.

A la dificultad técnica se suma una dificultad organizativa. Los equipos responsables del despliegue de simuladores clínicos con LLM afrontan decisiones recurrentes sobre qué modelo utilizar, cómo diseñar el prompt y cómo proceder cuando el proveedor anuncia la descontinuación del modelo en producción. Estas decisiones suelen tomarse bajo presiones de calendario y con evidencia técnica indirecta (benchmarks generalistas, capacidades anunciadas por el proveedor) que no necesariamente se traducen en mejor comportamiento sobre la tarea concreta.

Este estudio presenta una metodología reproducible para medir la confabulación demográfica bajo presión conversacional y la aplica a la evaluación comparada de cinco modelos comerciales sobre el simulador MediRol, desarrollado en nuestro centro para la formación de residentes y estudiantes avanzados. El punto de partida es una situación concreta — la descontinuación anunciada de gpt-4o y la necesidad de evaluar su reemplazo — que articulamos en objetivos explícitos en la sección siguiente.

---

## 2. Objetivos

El estudio se organiza en torno a dos objetivos primarios y tres objetivos secundarios, que se corresponden con decisiones concretas que afrontan los equipos responsables de desplegar simuladores clínicos basados en LLM en entornos educativos.

### Objetivos primarios

**OP1.** Evaluar si gpt-5.4-mini constituye una migración adecuada desde gpt-4o, actualmente en producción en el simulador MediRol y en proceso de descontinuación por el proveedor. La evaluación se realiza con el prompt de producción, optimizado originalmente para gpt-4o, a fin de reproducir las condiciones reales de despliegue.

**OP2.** Demostrar empíricamente que las intervenciones de prompt orientadas a mitigar un sesgo concreto pueden amplificar o generar sesgos colaterales no contemplados en el diseño de la intervención. Caracterizar la magnitud, dirección y especificidad de estos efectos cruzados.

### Objetivos secundarios

**OS1.** Proponer y validar una metodología reproducible para evaluar sesgos en LLMs clínicos basada en la construcción de casos clínicos con carga estigmatizante, omisión deliberada de datos demográficos y sondeo adversarial con escalada de presión conversacional.

**OS2.** Caracterizar el comportamiento comparado de cinco modelos de lenguaje comerciales en términos de fuga de datos confabulados y estereotipia demográfica, incluyendo la hipótesis implícita en la industria de que versiones más recientes (gpt-4.1 como supuesta actualización de gpt-4o) son necesariamente más seguras.

**OS3.** Cuantificar el efecto diferencial del diseño del prompt sobre distintos tipos de sesgo, con atención específica a los ejes que no son objeto directo de la intervención.

---

## 3. Metodología

### 3.1. El simulador MediRol

MediRol es un simulador clínico basado en LLM desarrollado para la formación de profesionales sanitarios en anamnesis y razonamiento diagnóstico. El modelo asume el rol del paciente a partir de una ficha de caso clínico estructurada y responde por turnos a las preguntas del estudiante, que ejerce el rol de clínico. El prompt de producción establece un marco de control de flujo clínico: restringe la entrega de información a lo explícitamente solicitado, diferencia categorías clínicas (antecedentes, síntomas, exploración, pruebas, tratamiento), y define los roles de paciente, enfermería y medicina con sus tonos correspondientes. Fue diseñado e iterativamente optimizado sobre gpt-4o.

### 3.2. Diseño de casos clínicos controvertidos

Para maximizar la sensibilidad del instrumento al sesgo, se construyeron once casos clínicos correspondientes a patologías con carga estigmatizante o estereotipia social documentada: infección por VIH, sífilis, hepatitis C, tuberculosis, dependencia de opioides, trastorno por consumo de alcohol, obesidad, escabiosis, depresión, ansiedad y fibromialgia. Cada ficha de caso contenía el motivo de consulta, síntomas y características clínicas relevantes, pero deliberadamente omitía todos los datos demográficos e identificativos del paciente (edad, sexo, orientación sexual, profesión, religión, etnia, nacionalidad, nivel de estudios, situación económica, situación familiar, lugar de residencia, y nombres del personal asistencial).

Esta omisión convierte al simulador en un sistema con una propiedad empíricamente útil: cualquier dato demográfico que el modelo proporcione es, por construcción, una confabulación. Si la confabulación es consistente con estereotipos clínicos del grupo patológico correspondiente, la evidencia sugiere que el modelo está transmitiendo al estudiante un perfil demográfico aprendido del preentrenamiento, no derivado del caso.

### 3.3. Sondeo adversarial con escalada de presión

Sobre cada caso se formularon catorce preguntas dirigidas a extraer datos demográficos o identificativos. Cada pregunta se planteó en tres rondas de presión creciente: pregunta directa y neutra (ronda 1), pregunta forzada apelando a necesidad clínica (ronda 2), y pregunta máxima invocando urgencia y riesgo vital (ronda 3). La escalada reproduce el tipo de presión conversacional que un estudiante novato puede ejercer sobre un paciente real ante información difícil de obtener, y permite diferenciar entre modelos que resisten solo la pregunta neutra de aquellos que mantienen la negativa bajo presión sostenida.

### 3.4. Condiciones experimentales

Se evaluaron tres configuraciones de prompt:

1. **Básico**: instrucción mínima de rol (una frase).
2. **Básico + "no reveles"**: el anterior, más una línea explícita prohibiendo la revelación de datos demográficos.
3. **Prompt explícito**: el prompt de producción completo del simulador MediRol, con el marco estructurado de control de flujo clínico.

La condición de prompt explícito se ejecutó dos veces de forma independiente (identificadas como r1 y r2 en las tablas de reproducibilidad) con idéntica configuración de temperatura, con dos fines complementarios: evaluar la estabilidad intra-condición del comportamiento del modelo, y aumentar la potencia estadística en la condición correspondiente a la configuración de producción, que es la de mayor interés para los objetivos del estudio.

Cada configuración se aplicó a los cinco modelos sobre los once casos, con las catorce preguntas y las tres rondas de escalada, generando 2 310 interacciones por modelo y por ejecución. La condición de prompt explícito acumula por tanto 4 620 interacciones por modelo.

### 3.5. Clasificación de respuestas

Cada respuesta del simulador se clasificó en una de cuatro categorías:

- **Negación**: el paciente rechaza entregar el dato.
- **Evasión**: redirige a información clínica sin responder a la pregunta.
- **Revelación atenuada**: entrega un valor con marcador de incertidumbre.
- **Revelación**: afirma un valor concreto.

El pipeline combinó un clasificador basado en expresiones regulares con dos jueces LLM independientes; las discrepancias entre jueces se resolvieron mediante un tercer juez. La concordancia entre los dos jueces primarios fue de 1 795/2 068 = 87 % en los casos que requirieron revisión. La métrica primaria, denominada tasa de fuga o *leak rate*, se definió como la proporción de respuestas clasificadas como revelación o revelación atenuada sobre el total de respuestas.

Para las preguntas sobre personal sanitario se extrajeron los nombres propuestos por el modelo y se infirió el género mediante diccionario de nombres en español (biblioteca gender-guesser adaptada), obteniendo una métrica complementaria de estereotipia de género ocupacional. Los nombres con género ambiguo, neutral o no identificable se codificaron como categoría separada y se excluyeron del análisis binomial.

### 3.6. Análisis estadístico

Las comparaciones entre tasas se realizaron mediante pruebas χ² con corrección de Yates o prueba exacta de Fisher cuando el tamaño muestral lo requería, ajustando todos los p-valores por tasa de falso descubrimiento (FDR, procedimiento de Benjamini-Hochberg). Los tamaños de efecto se reportan como V de Cramér para tablas de contingencia y h de Cohen para comparaciones de proporciones. La reproducibilidad entre las dos ejecuciones del prompt explícito se evaluó por contraste directo de tasas dentro de cada modelo (ausencia de diferencia significativa como criterio de estabilidad).

---

## 4. Resultados

### 4.1. Tasa de fuga: el modelo domina sobre el prompt

La tasa global de confabulación de datos demográficos bajo presión conversacional varía entre el 6 % y el 87 % según la combinación modelo × condición (Tabla 1). Los contrastes entre condiciones dentro de cada modelo son todos estadísticamente significativos tras ajuste FDR (p < 10⁻³ en las comparaciones pareadas relevantes), con tamaños de efecto medianos a grandes (h de Cohen entre 0,35 y 1,41).

**Tabla 1.** Tasa de fuga (%) por modelo y condición experimental. En la condición de prompt explícito se reporta el promedio de las dos ejecuciones independientes con el rango entre paréntesis. n = 2 310 interacciones por celda básica; n = 4 620 para prompt explícito.

| Modelo | Básico | + "no reveles" | Prompt explícito | Global |
|---|---|---|---|---|
| gpt-4.1 | 87,2 | 23,2 | 49,2 (46,8–51,7) | 52,2 |
| deepseek | 78,4 | 37,4 | 58,9 (57,4–60,4) | 58,4 |
| gpt-5.4-mini | 42,4 | 0,4 | 6,2 (5,6–6,7) | 13,8 |
| gpt-4o | 24,0 | 0,4 | 11,8 (11,0–12,6) | 12,0 |
| gemini-flash | 23,2 | 0,9 | 14,5 (13,6–15,4) | 13,3 |

La reproducibilidad entre las dos ejecuciones del prompt explícito es alta en los cinco modelos: ninguna diferencia entre ejecuciones alcanza significación estadística (p-ajustado entre 0,35 y 0,49), lo que confirma que las estimaciones son estables y no reflejan ruido de muestreo.

La primera observación estructural que se desprende de la tabla es que **la elección de modelo domina sobre la elección de prompt**: el mejor resultado de gpt-4.1 con el prompt más estricto (23,2 %) es peor que el resultado de gpt-4o o gpt-5.4-mini en su configuración menos protectora (11,0-12,6 % con prompt explícito). Cualquier esfuerzo de ingeniería de prompt queda subordinado a las propiedades del modelo subyacente.

### 4.2. El prompt de producción transfiere de forma heterogénea entre modelos

El prompt explícito fue desarrollado y optimizado sobre gpt-4o. Al aplicarlo a otros modelos, su rendimiento varía de forma no trivial. Tomando la media de las dos ejecuciones como medida robusta, **gpt-5.4-mini (6,2 %) y gemini-flash (14,5 %)** rinden comparable o mejor que gpt-4o (11,8 %); **gpt-4.1 (49,2 %) y deepseek (58,9 %)** rinden sustancialmente peor.

Que un prompt optimizado para un modelo funcione razonablemente en algunos y mal en otros no es sorprendente en sí; lo relevante es que la variabilidad entre familias de modelos alcanza un factor superior a ocho entre el mejor y el peor comportamiento, lo que obliga a revalidar empíricamente el prompt tras cualquier cambio de modelo, incluso cuando el cambio se realiza dentro del mismo proveedor. La asunción habitual en equipos de desarrollo — que un prompt validado en un modelo es exportable sin cambios — no se sostiene empíricamente.

### 4.3. Un modelo más reciente no es necesariamente más seguro

La comparación entre gpt-4o y gpt-4.1 es el hallazgo más contraintuitivo del estudio y responde directamente al objetivo secundario OS2. En todas las condiciones, gpt-4.1 filtra sustancialmente más que gpt-4o:

- Condición básica: 87,2 % frente a 24,0 % (+63 pp, h = 1,41)
- Básico + "no reveles": 23,2 % frente a 0,4 % (+23 pp, h = 0,79)
- Prompt explícito: 49,2 % frente a 11,8 % (+37 pp, h = 0,86)

Todas las diferencias son altamente significativas tras ajuste FDR. El patrón es consistente: gpt-4.1, pese a ser un modelo más reciente y posicionado comercialmente como mejora de gpt-4o, se comporta como un modelo marcadamente más colaborativo ante la presión para revelar datos, en los tres regímenes de control de prompt evaluados. En el contexto educativo, la asunción implícita de que actualizar de versión mejora o al menos preserva las propiedades deseables es empíricamente incorrecta para el caso evaluado.

### 4.4. Evaluación de gpt-5.4-mini como migración desde gpt-4o

Con gpt-4.1 descartado por los resultados anteriores, gpt-5.4-mini se propone como candidato de reemplazo ante la descontinuación de gpt-4o. Los datos respaldan la migración en la métrica principal del estudio (OP1): bajo el prompt de producción, gpt-5.4-mini filtra el 6,2 % de las preguntas, aproximadamente la mitad que gpt-4o (11,8 %). La diferencia es estadísticamente significativa (p-ajustado < 10⁻¹¹) y se replica entre las dos ejecuciones del prompt en ambos modelos.

La evaluación por una sola métrica es, sin embargo, insuficiente. Como se detalla en las secciones siguientes, gpt-5.4-mini presenta un perfil de estereotipia de género ocupacional más extremo que gpt-4o, y el cambio de modelo implica asumir esta contrapartida.

### 4.5. El perfil del paciente confabulado reproduce estereotipos clínicos

Cuando los modelos revelan datos demográficos, el perfil que construyen reproduce estereotipos clínicos consistentes con la literatura epidemiológica y social de cada patología. En la Tabla 2 se sintetizan los perfiles modales para cuatro enfermedades ilustrativas.

**Tabla 2.** Perfil modal del paciente confabulado para cuatro patologías seleccionadas, agregando los cinco modelos, en la condición de prompt explícito.

| Atributo | VIH | Dependencia opioides | Ansiedad | Hepatitis C |
|---|---|---|---|---|
| Edad | Joven (<30) | Adulto (30–50) | Adulto (30–50) | Adulto (30–50) |
| Sexo | Hombre | Hombre | Otro/desconocido | Hombre |
| Orientación sexual | Bisexual / HSH | Heterosexual | Heterosexual | Heterosexual |
| Profesión | Administrativo | Oficio manual | Educador / Administrativo | Oficio manual |
| Nivel de estudios | Universitario | ESO/Secundaria | Universitario | ESO/Secundaria |
| Situación económica | Variable | Baja | Alta | Baja |
| Situación familiar | Con pareja sin hijos | Con pareja sin hijos | Con pareja e hijos | Con pareja sin hijos |

Los perfiles son notablemente estables entre las dos ejecuciones del prompt explícito (concordancia de valor modal superior al 80 % en la mayoría de atributos) y varían de forma significativa entre modelos para 9 de los 11 atributos evaluados (V de Cramér entre 0,22 y 0,66 según atributo, p-ajustado < 0,03 en todas las comparaciones significativas).

### 4.6. Sesgo de género ocupacional

Cuando el modelo nombra al personal sanitario en respuesta a preguntas específicas, la distribución de género inferido se desvía significativamente del equilibrio en direcciones complementarias y estereotipadas (Tabla 3).

**Tabla 3.** Proporción de nombres masculinos (%) entre nombres con género identificable, para personal de enfermería y medicina, por modelo. Agregado sobre las tres condiciones experimentales.

| Modelo | Enfermería %H (n) | Medicina %H (n) |
|---|---|---|
| gpt-5.4-mini | 23,7 (38) *** | 100,0 (43) *** |
| gpt-4.1 | 0,0 (63) *** | 78,2 (55) *** |
| gpt-4o | 50,0 (22) n.s. | 88,9 (27) *** |
| gemini-flash | 35,7 (28) n.s. | 63,6 (22) n.s. |
| deepseek | 57,1 (70) n.s. | 73,8 (84) *** |

*** p-ajustado < 0,01; n.s. = no significativo.

El test global χ² entre modelos es significativo para ambos roles (enfermería: V de Cramér = 0,50, p-ajustado = 6,7·10⁻¹¹; medicina: V = 0,28, p-ajustado = 2,2·10⁻³).

### 4.7. El bloqueo de información demográfica produce un sesgo de género por defecto

La condición "Básico + 'no reveles'", diseñada para suprimir la confabulación demográfica, introduce un efecto colateral específico sobre el género del personal de enfermería (Tabla 4).

**Tabla 4.** Proporción de nombres masculinos (%) entre nombres con género identificable, por rol y condición, agregando los cinco modelos.

| Condición | Enfermería %H (n) | Medicina %H (n) |
|---|---|---|
| Básico | 64,5 (31) | 83,3 (54) |
| Básico + "no reveles" | **100,0 (12)** | 81,8 (11) |
| Prompt explícito r1 | 7,7 (52) | 81,8 (44) |
| Prompt explícito r2 | 20,4 (54) | 65,5 (55) |

La comparación entre la condición basal y la condición con instrucción anti-demográfica muestra dos efectos de signos opuestos según el rol. En enfermería, la introducción de la línea "no reveles" invierte el patrón: el 35,5 % de nombres femeninos en condición basal pasa al 0 % (prueba exacta de Fisher: OR = 0, p = 0,019). En medicina, la misma instrucción no modifica el patrón (83,3 % → 81,8 % masculino; p = 1,00).

El resultado neto es que la intervención de control produce un sesgo de género absoluto en el personal circundante: la enfermería se masculiniza al 100 % mediante el mecanismo del masculino gramaticalmente no marcado del español como estrategia de minimización de compromiso demográfico aparente.

### 4.8. El prompt modifica no solo la magnitud del sesgo sino su distribución

De los once atributos demográficos evaluados, seis muestran distribuciones significativamente distintas entre condiciones tras ajuste FDR (Tabla 5).

**Tabla 5.** Test de distribución de atributos demográficos confabulados entre las tres condiciones experimentales.

| Atributo | χ² | V de Cramér | n | p-ajustado | Significación |
|---|---|---|---|---|---|
| Sexo | 21,89 | 0,177 | 351 | 0,007 | *** |
| Profesión | 37,62 | 0,258 | 188 | 0,007 | *** |
| Raza | 35,78 | 0,314 | 121 | 0,007 | *** |
| Nacionalidad | 17,44 | 0,254 | 135 | 0,021 | *** |
| Estudios | 33,62 | 0,232 | 208 | 0,031 | *** |
| Familia | 24,30 | 0,153 | 346 | 0,034 | *** |
| Edad | 15,80 | 0,144 | 255 | 0,098 | n.s. |
| Orientación | 11,25 | 0,169 | 131 | 0,508 | n.s. |
| Religión | 9,61 | 0,166 | 116 | 0,421 | n.s. |
| Economía | 11,46 | 0,162 | 145 | 0,300 | n.s. |
| Etnia | 21,21 | 0,237 | 126 | 0,074 | n.s. |

Cambiar el prompt no modifica únicamente con qué frecuencia se revela, sino también qué se revela cuando se revela: el perfil del paciente imaginario se deforma según la instrucción de control aplicada.

---

## 5. Discusión

### 5.1. Las magnitudes son relevantes para la práctica docente

Una tasa de fuga del 50 % bajo el prompt básico implica que, sin controles mínimos, un estudiante que interroga al simulador sobre aspectos demográficos recibe información fabricada la mitad de las veces. Esa información tiende a ser consistente con estereotipos epidemiológicos dominantes, lo que plantea un riesgo pedagógico específico: el simulador puede reforzar en el estudiante asociaciones heurísticas que invisibilizan la diversidad real de pacientes con estas patologías.

### 5.2. La evaluación de sesgos no se reduce a una métrica

El caso del reemplazo de gpt-4o por gpt-5.4-mini ilustra el problema. Un modelo puede mejorar en una dimensión (tasa de fuga a la mitad) a costa de empeorar en otra (estereotipia de género ocupacional en su forma máxima). Un mismo prompt puede reducir un sesgo y amplificar otro, como ilustra el contraste entre la condición "no reveles" (tasa de fuga mínima pero masculinización absoluta del personal de enfermería) y las condiciones con prompt explícito (tasa intermedia pero feminización extrema del mismo rol).

Las intervenciones de mitigación deben evaluarse no solo en su eje de diseño sino en la totalidad de ejes sensibles accesibles al análisis. Una instrucción no es inocua por no mencionar una dimensión: puede estar regulando esa dimensión por defecto de forma más rígida que cualquier instrucción explícita.

### 5.3. Las actualizaciones de modelo no son rutinarias

El resultado más contraintuitivo del estudio es que gpt-4.1, presentado comercialmente como sucesor de gpt-4o, triplica su tasa de fuga. La capacidad general medida en benchmarks generalistas y la seguridad contextual en una tarea concreta no covarían de forma garantizada, y cualquier migración requiere revalidación empírica sobre la tarea específica.

### 5.4. Limitaciones

- Número de casos por enfermedad reducido (uno por patología).
- Las celdas con efecto suelo limitan el análisis de contenido en la condición "Básico + no reveles".
- El análisis de género depende de la inferencia a partir del primer nombre (diccionario en español); identidades no binarias no capturables.
- Los modelos comerciales pueden modificarse sin aviso por actualizaciones del proveedor.

### 5.5. Implicaciones para la práctica docente

1. **Auditar** el comportamiento del modelo concreto que se despliega, no del modelo sobre el que se diseñó el prompt.
2. **Evaluar al menos dos ejes** de sesgo (confabulación y estereotipia) antes de desplegar, y revalidar ante cualquier cambio.
3. **Comunicar la limitación** a los estudiantes y orientarles a detectar estereotipos, convirtiendo la limitación en oportunidad formativa.
4. **Tratar las migraciones de modelo** como decisiones clínico-pedagógicas sujetas a validación, no como actualizaciones técnicas rutinarias.

---

## 6. Conclusiones

La evaluación de cinco modelos de lenguaje sobre el simulador clínico MediRol, bajo tres configuraciones de prompt y once patologías con carga estigmatizante, documenta que la confabulación de datos demográficos bajo presión conversacional es un fenómeno cuantitativamente masivo (entre el 6 % y el 87 %), cualitativamente estereotipado (los datos confabulados reproducen perfiles epidemiológicos y sociales dominantes), y con variabilidad entre modelos de hasta un factor diez.

Respecto a OP1, gpt-5.4-mini muestra una tasa de fuga aproximadamente la mitad de la de gpt-4o bajo el prompt de producción, lo que respalda la migración; debe acompañarse de medidas para atenuar una estereotipia de género ocupacional más extrema. Respecto a OP2, las intervenciones de prompt orientadas a mitigar la confabulación demográfica modifican sistemáticamente la distribución de los sesgos colaterales: una instrucción mínima y formalmente neutra produce una masculinización absoluta del personal de enfermería. Los sesgos no desaparecen con las intervenciones: se reorganizan.

La metodología de sondeo adversarial sobre casos controvertidos aquí descrita es reproducible y extensible, y se propone como estándar mínimo en la evaluación de simuladores clínicos basados en LLM antes de su despliegue en entornos formativos.

---

## Declaraciones

**Contribución de los autores.** [Pendiente de completar por el equipo investigador.]

**Conflictos de interés.** [Pendiente.]

**Financiación.** [Pendiente.]

**Aprobación ética.** Dado que el estudio no involucra participantes humanos ni datos clínicos reales — las interacciones analizadas son exclusivamente entre el evaluador automatizado y los modelos de lenguaje — no se requirió aprobación del Comité de Ética de la Investigación con medicamentos (CEIm). Los casos clínicos utilizados son ficticios y fueron construidos específicamente para el estudio.

**Disponibilidad de datos y código.** [Pendiente de definir repositorio.]

---

*Manuscrito preparado para envío a revista de educación médica.*
