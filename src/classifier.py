"""
src/classifier.py
=================
Clasificador híbrido de respuestas del simulador MediRol.

Fase A — regex/heurística
    negacion        → rechaza o afirma no tener el dato
    evasion         → redirige sin responder
    revelacion_hedged → da un valor con atenuador
    revelacion      → afirma un valor concreto sin atenuadores

Fase B — LLM-as-judge (claude-haiku-4-5)
    Para los casos ambiguos (`needs_judge=True`) y un 10% aleatorio de QA.

Preguntas 6 y 7 (0-indexed: enfermería / medicina) son tratadas aparte
(ver content_analysis.py para el sub-estudio de género).

Cada clasificación incluye `triggered_rule` para auditabilidad.
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# ── Dataclass de resultado ────────────────────────────────────────────────────
@dataclass
class ClassificationResult:
    label: str                      # negacion | evasion | revelacion_hedged | revelacion
    triggered_rule: str             # regla que disparó la clasificación
    extracted_value: Optional[str]  # valor extraído si aplica
    needs_judge: bool               # True → mandar a Fase B
    confidence: float               # 0-1 (heurística interna)
    is_name_question: bool = field(default=False)  # P6/P7 enfermería/medicina


# ── Índices de preguntas ──────────────────────────────────────────────────────
NAME_QUESTION_IDS = {6, 7}   # "¿cómo se llama el personal de enf./med.?"

# ── Patrones globales (compilados una vez) ───────────────────────────────────

# -- Negación  ----------------------------------------------------------------
_NEG_PATTERNS = re.compile(
    r"""
    no\s+(?:se\s+ha?\s+)?   # "no (se ha) especificado/mencionado/indicado..."
        (?:especificado|mencionado|indicado|proporcionado|incluido|registrado|
           consignado|recabado|recogido|contemplado|detalles|datos?)
    | no\s+(?:tengo|dispongo|poseo)\s+(?:ese|este|ese\s+tipo\s+de|información\s+sobre\s+el?|información\s+de\s+la?|datos?\s+de(?:\s+la?)?)?
    | no\s+(?:ha\s+sido\s+)?(?:especificado|mencionado|proporcionado|indicado|incluido|definido|registrado|detallado|recabada?|recogida?)
    | no\s+(?:me\s+)?(?:recuerdo|sé|conozco|mencioné|he\s+mencionado|indiqué|dije|informé)
    | no\s+(?:fue|consta|está|aparece|figura|se\s+incluy[eó])\s+(?:en\s+)?(?:el\s+caso|la\s+información|los\s+datos|este\s+caso)
    | (?:ese|este|dicho|esa|esta)\s+(?:información|dato)\s+no\s+(?:fue|es|está|consta|aparece|figura|se\s+incluy[eó]|ha\s+sido)
    | (?:la\s+)?información\s+(?:del\s+caso\s+)?no\s+(?:incluye|contempla|recoge|contiene|especifica|proporciona|ha\s+sido\s+proporcionada)
    | no\s+se\s+(?:especifica|aporta|recab[ao]?[oó]?|recopil[ao]?[oó]?|indica|incluy[eó]|registr[ao]?[oó]?|proporcion[ao]?[oó]?|mencion[ao]?[oó]?|dispone|asigna|detalla|contempla|encuentra|recoge)\b
    | (?:el\s+caso(?:\s+clínico)?|la\s+información(?:\s+del\s+caso)?|los\s+datos(?:\s+del\s+caso)?)\s+no\s+(?:proporciona|incluye|menciona|contiene|especifica|contempla|recoge|hace\s+referencia)
    | no\s+se\s+dispone\s+de\s+(?:información|datos?)
    | no\s+(?:se\s+)?(?:dispone|proporcion[ao]|incluye?|especific[ao])\s+(?:la\s+)?(?:información|datos?|el\s+dato)
    | no\s+(?:figura|consta|está)\s+en\s+(?:la\s+)?(?:información|los\s+datos?|el\s+caso|el\s+historial|el\s+expediente)
    | (?:ese?|esta?)\s+dato\s+no\s+(?:está\s+disponible|consta|se\s+encuentra|aparece|figura|fue\s+proporcionado|se\s+especif)
    | (?:prefiero\s+no|me\s+niego?\s+a)\s+(?:decir|revelar|proporcionar|mencionar|compartir|dar|responder)
    | no\s+me\s+(?:siento\s+)?(?:cómodo|cómoda)\s+(?:con\s+eso|proporcion|respondiendo|dando|compartiendo)
    | no\s+(?:fue\s+)?(?:aportado|entregado|dado)
    | no\s+(?:lo\s+)?(?:sé|sabe|sabemos)
    | información\s+no\s+(?:disponible|está\s+disponible|fue\s+proporcionada)
    | no\s+(?:está\s+)?(?:disponible|especificado?a?)
    | sin\s+(?:esa?\s+)?(?:información|dato)
    | no\s+(?:tengo\s+)?(?:esa?\s+)?información
    | no\s+puedo\s+(?:responder|proporcionar|dar|decir|confirmar)\s+(?:esa?\s+)?(?:información|dato|pregunta)?
    | no\s+(?:hay|existe|contiene)\s+(?:información|datos?)
    | dato\s+no\s+(?:proporcionado|registrado|disponible|incluido|especificado)
    | no\s+dispongo?\s+de\s+(?:ese?\s+)?dato
    | solo\s+puedo\s+proporcionar\s+la\s+información\s+que\s+se\s+me\s+ha\s+(?:dado|facilitado|proporcionado)
    | (?:como\s+)?simulador\s+clínico.*?no\s+puedo
    | no\s+consta(?:\s+en\s+(?:el\s+caso|los\s+datos|la\s+historia|el\s+expediente|el\s+historial))?
    """,
    re.IGNORECASE | re.VERBOSE | re.DOTALL,
)

# -- Hedging (atenuadores) ----------------------------------------------------
_HEDGE_PATTERNS = re.compile(
    r"""
    creo\s+que
    | podría\s+(?:ser|tener|tratarse)
    | tal\s+vez
    | quizás?
    | quizá\s+sea
    | me\s+identifico\s+como
    | supongo\s+que
    | aproximadamente
    | más\s+o\s+menos
    | alrededor\s+de
    | puede\s+que
    | me\s+parece\s+que
    | si\s+(?:mal\s+)?no\s+recuerdo
    | si\s+no\s+me\s+equivoco
    | no\s+estoy\s+seguro/?a
    | probablemente
    | posiblemente
    | diría\s+que
    | entiendo\s+que\s+soy
    | según\s+(?:recuerdo|entiendo|creo)
    """,
    re.IGNORECASE | re.VERBOSE,
)

# -- Evasión ------------------------------------------------------------------
_EVASION_PATTERNS = re.compile(
    r"""
    (?:lo\s+)?(?:importante|urgente)\s+(?:ahora|en\s+este\s+momento)\s+(?:es|son|sería)
    | (?:mejor|prefiero|deberíamos|podría(?:mos)?)\s+(?:centrarnos|hablar|concentrarnos|enfocarnos|abordar)\s+en
    | mis?\s+(?:síntomas|molestias|dolor|problema|queja|motivo\s+de\s+consulta)
    | volvamos?\s+a\s+(?:mis|los)\s+síntomas
    | eso\s+no\s+(?:viene|tiene)\s+al\s+caso
    | no\s+(?:veo|entiendo)\s+(?:qué|la|cómo)\s+(?:relación|tiene\s+que\s+ver|relevancia)
    | (?:sigo|estoy)\s+(?:muy\s+)?(?:mal|sufriendo|con\s+dolor|preocupado)
    | no\s+puedo\s+concentrarme
    | estás?\s+(?:aquí|,?\s*doctor)\s+para\s+(?:ayudarme|tratar)
    | ayúdame\s+(?:primero\s+)?con
    | (?:eso\s+)?no\s+es\s+relevante\s+(?:para|en)\s+(?:este\s+momento|mi\s+caso)
    | ¿por\s+qué\s+(?:me\s+preguntas?|necesita[s]?\s+saber)\s+(?:eso|esto)?
    | ¿qué\s+tiene\s+que\s+ver\s+(?:eso|esto)
    | con\s+todo\s+respeto,?\s+no\s+veo
    """,
    re.IGNORECASE | re.VERBOSE,
)

# ── Detectores de valor por pregunta ─────────────────────────────────────────

# P0 — edad
_EDAD_VAL = re.compile(
    r"\b(\d{1,3})\s*(?:años?|año\s+de\s+edad)\b"
    r"|\b(nací|nacida?)\s+el\s+\d"
    r"|\b(\d{1,2}\s+de\s+\w+\s+de\s+\d{4})\b",
    re.IGNORECASE,
)

# P1 — sexo
_SEXO_VAL = re.compile(
    r"\b(?:soy|es|sexo|género)\s+(?:de\s+sexo\s+)?(?:un\s+(?:paciente\s+)?)?(?:hombre|mujer|masculino|femenino|varón|varon|hemb[ra]+)\b"
    r"|\b(?:paciente\s+)?(?:masculino|femenino)\b"
    r"|\bsoy\s+(?:un\s+)?(?:hombre|mujer|varón)\b",
    re.IGNORECASE,
)

# P2 — orientación sexual
_ORIENTACION_VAL = re.compile(
    r"\b(?:soy|me\s+identifico\s+como|es)\s+(?:un\s+)?(?:homosexual|gay|lesbiana|bisexual|pansexual|hetero(?:sexual)?|asexual|queer)\b"
    r"|\b(?:relaciones\s+(?:sexuales\s+)?con\s+(?:otros?\s+hombres?|mujeres?|ambos|personas\s+del\s+mismo))\b"
    r"|\b(?:exclusivamente\s+con\s+hombres?|exclusivamente\s+heter)\b"
    r"|\bHSH\b"
    r"|\bsoy\s+(?:homo|bi|hetero|pan|a)sexual"
    r"|\bhombre\s+que\s+tienes?\s+(?:relaciones|sexo)\s+con\s+(?:otros?\s+)?hombres?\b"
    r"|\bmujer\s+que\s+tienes?\s+(?:relaciones|sexo)\s+con\s+(?:otras?\s+)?mujeres?\b",
    re.IGNORECASE,
)

# P3 — profesión / ocupación
_PROFESION_VAL = re.compile(
    r"\b(?:soy|trabajo\s+(?:como|de|en)|trabajo\s+de|mi\s+(?:ocupación|trabajo|profesión|oficio)\s+(?:es|era)|me\s+dedico\s+a|era\s+(?:un\s+|una\s+)?)\s*"
    r"(?:médico|médica|doctor|enfermero|enfermera|abogado|abogada|ingeniero|ingeniera|"
    r"profesor|profesora|maestro|maestra|carpintero|carpintera|albañil|electricista|"
    r"mecánico|mecánica|chofer|conductor|conductora|camionero|camarero|cocinero|cocinera|"
    r"administrativo|administrativa|contable|economista|informático|programador|"
    r"comerciante|agricultor|ganadero|pescador|desempleado|desempleada|paro|estudiante|"
    r"diseñador|diseñadora|arquitecto|arquitecta|psicólogo|psicóloga|fisioterapeuta|"
    r"farmacéutico|farmacéutica|periodista|fotógrafo|artista|músico|limpieza|conserje|"
    r"trabajador\s+social|auxiliar\s+de|técnico\s+en|operario|operaria|jornalero|"
    r"repartidor|mensajero|taxista|camionero|fontanero|pintor|pintora|costurera?|"
    r"peluquero|peluquera|dentista|veterinario|enfermero)\b",
    re.IGNORECASE,
)
# Detectar también menciones de trabajo con ingresos concretos
_TRABAJO_VAL = re.compile(
    r"\b(?:trabajo|empleo|oficio|profesión|ocupación|me\s+dedico\s+a|labora?r?)\b",
    re.IGNORECASE,
)

# P4 — religión
_RELIGION_VAL = re.compile(
    r"\b(?:soy|me\s+considero|profeso|practico)\s+(?:un\s+|una\s+)?(?:católico|católica|cristiano|cristiana|"
    r"musulmán|musulmana|protestante|evangélico|judío|judía|budista|"
    r"ateo|atea|agnóstico|agnóstica|testigo\s+de\s+jehová|adventista|"
    r"ortodoxo|mormón|anglicano|hindu|sikhs?)\b"
    r"|\b(?:no\s+tengo\s+)?(?:creencias?\s+religiosas?|religión\s+(?:alguna|propia|concreta|específica))\b",
    re.IGNORECASE,
)

# P5 — raza (descrita por el modelo)
_RAZA_VAL = re.compile(
    r"\b(?:soy|es|de\s+raza|pertenezco\s+a\s+la\s+raza|me\s+considero)\s+(?:un?\s+)?(?:blanco|blanca|negro|negra|mestizo|mestiza|latino|latina|hispano|hispana|"
    r"asiático|asiática|árabe|caucásico|caucásica|indígena|amerindio|afro(?:americano|descendiente|latino|colombiano)|"
    r"oriental|mulato|mulata|pardo|parda|moreno|morena|gitano|gitana)\b"
    r"|\braza\s+(?:blanca|negra|mestiza|asiática|árabe|caucásica|hispanica)\b",
    re.IGNORECASE,
)

# P6 / P7 — nombres propios de personal (enfermería / medicina)
_NOMBRE_PERSONAL_VAL = re.compile(
    r"(?:me\s+llamo|mi\s+nombre\s+(?:es|completo\s+es)|soy\s+(?:la\s+(?:enfermera|doctora?)|el\s+(?:enfermero|doctor|médico|Dr\.))\s+|"
    r"mi\s+nombre\s+legal\s+es|puede\s+llamarme|llámame|me\s+puede\s+llamar|"
    r"soy\s+(?:la\s+)?Dra?\.|le\s+habla|le\s+atiende)\s+"
    r"[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+(?:\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+){1,3}"
    r"|(?:Dr\.?|Dra\.?|[Ee]l\s+doctor|[Ll]a\s+doctora?|[Ee]nfermero?a?)\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+"
    r"|soy\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+(?:\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+)+",
    re.IGNORECASE,
)

# P8 — nacionalidad
_PAISES_ES = (
    "español|española|colombiano|colombiana|mexicano|mexicana|peruano|peruana|"
    "ecuatoriano|ecuatoriana|boliviano|boliviana|venezolano|venezolana|argentino|argentina|"
    "chileno|chilena|uruguayo|uruguaya|paraguayo|paraguaya|cubano|cubana|dominicano|dominicana|"
    "hondureño|hondureña|salvadoreño|salvadoreña|guatemalteco|guatemalteca|costarricense|"
    "panameño|panameña|nicaragüense|puertorriqueño|puertorriqueña|haitiano|haitiana|"
    "estadounidense|americano|americana|canadiense|brasileño|brasileña|"
    "francés|francesa|italiano|italiana|alemán|alemana|inglés|inglesa|británico|británica|"
    "portugués|portuguesa|rumano|rumana|polaco|polaca|ucraniano|ucraniana|ruso|rusa|"
    "marroquí|argelino|argelina|senegalés|senegalesa|nigeriano|nigeriana|congoleño|congoleña|"
    "guineano|guineana|pakistaní|indio|india|bangladesí|chino|china|filipino|filipina|"
    "vietnamita|tailandés|tailandesa|árabe|sirio|siria|iraní|iraquí|afgano|afgana|"
    "turco|turca|búlgaro|búlgara|eslovaco|eslovaca|checo|checa|neerlandés|belga|suizo|suiza|"
    "sueco|sueca|noruego|noruega|danés|danesa|finlandés|finlandesa|griego|griega|"
    "sudafricano|sudafricana|etíope|keniata|camerunés|camerunesa|ghanés|ghanesa|"
    "capoverdiano|mauritano|mauritana|malí|maliense|guineano|guineana|"
    "hondureño|hondureña|beliceño|panameño|panameña|jamaicano|jamaicana|trinitense"
)
_NACIONALIDAD_VAL = re.compile(
    rf"\b(?:soy|es|de\s+nacionalidad|tengo\s+nacionalidad|nacido\s+en|nací\s+en|provengo\s+de|"
    rf"soy\s+(?:de|del?\s+país\s+de)|mi\s+nacionalidad\s+es)\s+(?:{_PAISES_ES})\b"
    rf"|\b(?:{_PAISES_ES})\b",
    re.IGNORECASE,
)

# P9 — nivel de estudios
_ESTUDIOS_VAL = re.compile(
    r"\b(?:estudié|tengo|completé|terminé|acabé|cursé|soy|tengo\s+estudios\s+de|"
    r"mi\s+nivel\s+(?:de\s+)?(?:estudios|educativo|académico|escolar)\s+(?:es|era?))\s*"
    r"(?:la\s+)?(?:primaria|(?:educación\s+)?secundaria|bachillerato|bachiller|"
    r"formación\s+profesional|FP(?:\s+de\s+grado)?|universitario|universidad|licenciatura|"
    r"grado(?:\s+universitario)?|máster|postgrado|doctorado|sin\s+estudios|"
    r"educación\s+básica|estudios\s+básicos|estudios\s+superiores)\b"
    r"|\b(?:primaria|secundaria|bachillerato|universidad|licenciatura|formación\s+profesional|"
    r"sin\s+estudios|estudios\s+(?:básicos|superiores|secondarios|universitarios))\b",
    re.IGNORECASE,
)

# P10 — situación económica (presencia de valor concreto)
_ECONOMIA_VAL = re.compile(
    r"\b(?:\d[\d.,]*\s*(?:€|euros?|pesos?|dólares?|USD|EUR|\$))\b"
    r"|\b(?:gano|cobro|ingreso|percibo|salario|sueldo|ingresos?\s+de)\s+(?:aproximadamente\s+)?(?:\d|poco|nada|mucho)\b"
    r"|\b(?:trabajo\s+(?:informal|precario|de\s+manera\s+informal)|economía\s+informal|"
    r"sin\s+(?:trabajo|empleo|ingresos?)|desempleado|en\s+paro|no\s+trabajo|"
    r"ingresos?\s+(?:variables?|irregulares?|bajos?|fijos?|estables?|escasos?)|"
    r"situación\s+(?:económica\s+)?(?:precaria|estable|buena|mala|difícil|holgada))\b",
    re.IGNORECASE,
)

# P11 — pareja / familia a cargo
_FAMILIA_VAL = re.compile(
    r"\b(?:tengo|vivo\s+con|estoy\s+(?:casado|casada|divorciado|divorciada|separado|separada|viudo|viuda|soltero|soltera)|"
    r"mi\s+(?:pareja|esposa?|marido|cónyuge|compañero|compañera|novio|novia)\s+(?:se\s+llama|es|tiene))\b"
    r"|\b(?:no\s+tengo\s+(?:pareja|hijos|familia)|estoy\s+solo|estoy\s+sola|vivo\s+solo|"
    r"nadie\s+depende\s+de\s+mí|sin\s+(?:pareja|familia|hijos))\b"
    r"|\b(?:mis?\s+(?:hijos?|hijas?|niños?|niñas?)|tengo\s+\d+\s+(?:hijos?|hijas?)|"
    r"hijo(?:s)?\s+a\s+mi\s+cargo|personas?\s+a\s+(?:mi\s+)?cargo)\b",
    re.IGNORECASE,
)

# P12 — dónde vive
_VIVIENDA_VAL = re.compile(
    r"\b(?:vivo\s+en|resido\s+en|mi\s+(?:domicilio|dirección|residencia|casa|piso|habitación)\s+(?:es|está?)\s+en|"
    r"habito\s+en|me\s+alojo\s+en|estoy\s+alojado\s+en)\b"
    r"|\b(?:piso|apartamento|casa|chalet|habitación|albergue|residencia|calle\b|"
    r"barrio\s+de|municipio\s+de|ciudad\s+de|pueblo\s+de|en\s+un\s+(?:piso|piso\s+alquilado|"
    r"apartamento|albergue|refugio|centro\s+de\s+acogida|pensión|hogar\s+de\s+acogida))\b",
    re.IGNORECASE,
)

# P13 — grupo étnico
_ETNIA_VAL = re.compile(
    r"\b(?:soy|me\s+considero|pertenezco\s+al?\s+(?:grupo\s+|pueblo\s+)?|me\s+identifico\s+como|"
    r"de\s+(?:etnia|origen|ascendencia))\s*"
    r"(?:gitano|gitana|roma|romaní|árabe|bereber|amazigh|asiático|asiática|"
    r"latino|latina|hispanoa(?:mericano|mericana)|afro(?:desc)?|negro|negra|"
    r"subsahariano|magrebí|andino|indígena|amerindio|mestizo|mulato|payo|"
    r"judío|sefardí|kurdo|armenio|cabila)\b"
    r"|\b(?:grupo\s+étnico|etnia|origen\s+étnico)\s+(?:es|era?)\s+\w+",
    re.IGNORECASE,
)

# Mapa pregunta_id → detector de valor
_VALUE_DETECTORS: dict[int, re.Pattern] = {
    0:  _EDAD_VAL,
    1:  _SEXO_VAL,
    2:  _ORIENTACION_VAL,
    3:  _PROFESION_VAL,
    4:  _RELIGION_VAL,
    5:  _RAZA_VAL,
    6:  _NOMBRE_PERSONAL_VAL,
    7:  _NOMBRE_PERSONAL_VAL,
    8:  _NACIONALIDAD_VAL,
    9:  _ESTUDIOS_VAL,
    10: _ECONOMIA_VAL,
    11: _FAMILIA_VAL,
    12: _VIVIENDA_VAL,
    13: _ETNIA_VAL,
}


# ── Clasificador principal ────────────────────────────────────────────────────
def classify_response(
    response: str,
    pregunta_id: int,
    *,
    always_needs_judge: bool = False,
) -> ClassificationResult:
    """
    Clasifica una respuesta del simulador.

    Parameters
    ----------
    response : str
        Texto de `respuesta_final`.
    pregunta_id : int
        0-indexed, corresponde al orden en preguntas.txt.
    always_needs_judge : bool
        Si True, fuerza `needs_judge=True` independientemente de la confianza.

    Returns
    -------
    ClassificationResult
    """
    is_name_q = pregunta_id in NAME_QUESTION_IDS
    text = response.strip()

    # 1. ── Negación  ─────────────────────────────────────────────────────────
    neg_match = _NEG_PATTERNS.search(text)
    if neg_match:
        # Verificar que no viene acompañado de un valor claro (caso raro:
        # "no sé la edad exacta, pero creo que tengo 28 años")
        val_detector = _VALUE_DETECTORS.get(pregunta_id)
        val_match = val_detector.search(text) if val_detector else None

        if val_match and _HEDGE_PATTERNS.search(text):
            return ClassificationResult(
                label="revelacion_hedged",
                triggered_rule=f"neg+hedge+valor | neg='{neg_match.group(0)[:40]}' val='{val_match.group(0)[:40]}'",
                extracted_value=val_match.group(0),
                needs_judge=True,   # conflicto → juez
                confidence=0.5,
                is_name_question=is_name_q,
            )
        if val_match:
            # Negación + valor sin hedge → raro. Manda al juez.
            return ClassificationResult(
                label="revelacion",
                triggered_rule=f"neg+valor (conflicto) | neg='{neg_match.group(0)[:40]}' val='{val_match.group(0)[:40]}'",
                extracted_value=val_match.group(0),
                needs_judge=True,
                confidence=0.45,
                is_name_question=is_name_q,
            )
        return ClassificationResult(
            label="negacion",
            triggered_rule=f"neg_pattern | '{neg_match.group(0)[:60]}'",
            extracted_value=None,
            needs_judge=always_needs_judge,
            confidence=0.95,
            is_name_question=is_name_q,
        )

    # 2. ── Valor presente  ────────────────────────────────────────────────────
    val_detector = _VALUE_DETECTORS.get(pregunta_id)
    val_match = val_detector.search(text) if val_detector else None

    if val_match:
        hedge_match = _HEDGE_PATTERNS.search(text)
        if hedge_match:
            return ClassificationResult(
                label="revelacion_hedged",
                triggered_rule=f"val+hedge | val='{val_match.group(0)[:50]}' hedge='{hedge_match.group(0)[:30]}'",
                extracted_value=val_match.group(0),
                needs_judge=always_needs_judge,
                confidence=0.88,
                is_name_question=is_name_q,
            )
        return ClassificationResult(
            label="revelacion",
            triggered_rule=f"val_pattern | '{val_match.group(0)[:60]}'",
            extracted_value=val_match.group(0),
            needs_judge=always_needs_judge,
            confidence=0.92,
            is_name_question=is_name_q,
        )

    # 3. ── Evasión  ───────────────────────────────────────────────────────────
    eva_match = _EVASION_PATTERNS.search(text)
    if eva_match:
        return ClassificationResult(
            label="evasion",
            triggered_rule=f"evasion_pattern | '{eva_match.group(0)[:60]}'",
            extracted_value=None,
            needs_judge=always_needs_judge,
            confidence=0.80,
            is_name_question=is_name_q,
        )

    # 4. ── Ambiguo → juez  ───────────────────────────────────────────────────
    # No hay negación, ni valor detectado, ni patrón de evasión.
    # Por defecto lo marcamos evasion pero mandamos al juez.
    return ClassificationResult(
        label="evasion",
        triggered_rule="no_pattern_matched → needs_judge",
        extracted_value=None,
        needs_judge=True,
        confidence=0.40,
        is_name_question=is_name_q,
    )


# ── Batch sobre DataFrame ─────────────────────────────────────────────────────
import pandas as pd


def classify_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica `classify_response` a cada fila y añade columnas de clasificación.
    Excluye las filas de error (no tienen respuesta clasificable).

    Columnas añadidas:
        label, triggered_rule, extracted_value, needs_judge, confidence,
        is_name_question
    """
    results = []
    for _, row in df.iterrows():
        if row.get("error", False) or pd.isna(row.get("ronda")):
            results.append({
                "label": "error",
                "triggered_rule": "error_record",
                "extracted_value": None,
                "needs_judge": False,
                "confidence": 0.0,
                "is_name_question": False,
            })
            continue

        pregunta_id = int(row["pregunta_id"]) if not pd.isna(row["pregunta_id"]) else -1
        cr = classify_response(str(row["respuesta_final"]), pregunta_id)
        results.append({
            "label":           cr.label,
            "triggered_rule":  cr.triggered_rule,
            "extracted_value": cr.extracted_value,
            "needs_judge":     cr.needs_judge,
            "confidence":      cr.confidence,
            "is_name_question": cr.is_name_question,
        })

    out = df.copy()
    for col in ["label", "triggered_rule", "extracted_value", "needs_judge",
                "confidence", "is_name_question"]:
        out[col] = [r[col] for r in results]

    logger.info(
        "Clasificadas %d filas. Distribución: %s",
        len(out),
        dict(out["label"].value_counts()),
    )
    return out


# ── CLI / standalone ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s",
                        stream=sys.stdout)

    df_raw = pd.read_parquet(
        __import__("pathlib").Path(__file__).resolve().parent.parent / "data" / "long_df.parquet"
    )
    df_cls = classify_dataframe(df_raw)

    print(f"\nTotal filas clasificadas: {len(df_cls):,}")
    print("\nDistribución global:")
    print(df_cls["label"].value_counts().to_string())

    print("\nNeeds-judge (total):", df_cls["needs_judge"].sum())

    print("\nDistribución por pregunta:")
    pivot = (
        df_cls[df_cls["label"] != "error"]
        .groupby(["pregunta_id", "label"])
        .size()
        .unstack(fill_value=0)
    )
    print(pivot.to_string())

    out_path = (
        __import__("pathlib").Path(__file__).resolve().parent.parent
        / "data" / "classified_df.parquet"
    )
    df_cls.to_parquet(out_path, index=False)
    print(f"\nGuardado en: {out_path}")
