"""
explicador.py
=============
Agente LangChain + Groq que explica las predicciones del modelo
y responde preguntas de seguimiento en lenguaje natural.

Requiere GROQ_API_KEY en el archivo .env del proyecto.
"""

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

load_dotenv()

_GROQ_MODEL = "llama-3.3-70b-versatile"

# ---------------------------------------------------------------------------
# Valores de referencia del dataset (medianas aproximadas por tier)
# Útiles para que el agente pueda contextualizar sin inventar datos
# ---------------------------------------------------------------------------
_REFERENCIA_TIERS = """
╔═════════════════════════════════════════════════════════════════╗
║           VALORES TÍPICOS POR TIER (dataset de 182 artistas)    ║
╠══════════════════╦══════════════╦══════════════╦════════════════╣
║ Métrica          ║     BAJO     ║    MEDIO     ║      ALTO      ║
╠══════════════════╬══════════════╬══════════════╬════════════════╣
║ Last.fm oyentes  ║  1K a 15K    ║  15K a 150K  ║  150K a 2M+    ║
║ Last.fm scrobbles║  10K a 200K  ║  200K a 3M   ║  3M a 50M+     ║
║ YT suscriptores  ║  500 a 15K   ║  10K a 200K  ║  200K a 5M+    ║
║ YT vistas tot.   ║  50K a 500K  ║  500K a 10M  ║  10M a 500M+   ║
║ Conciertos sl.fm ║  0 a 5 (21%  ║  5 a 30 (75% ║  20 a 100+     ║
║                  ║  con datos)  ║  con datos)  ║  (95% con dat.)║
║ Años activo      ║  1 a 4       ║  3 a 7       ║  5 a 12        ║
╠══════════════════╬══════════════╬══════════════╬════════════════╣
║ Artistas tipo    ║ Tarchi,      ║ BEJO,        ║ Bad Gyal,      ║
║                  ║ Gatti,       ║ Choclock,    ║ Quevedo,       ║
║                  ║ Xico Palma   ║ Metrika,     ║ Morad,         ║
║                  ║              ║ Elio Toffana,║ La Zowi,       ║
║                  ║              ║ Enry-k       ║ Rels B,        ║
║                  ║              ║              ║ Yung Beef,     ║
║                  ║              ║              ║ SAIKO, Maka    ║
╚══════════════════╩══════════════╩══════════════╩════════════════╝
Nota: Quevedo y Bad Gyal son outliers extremos de YouTube (10-50x la mediana de alto).
"""


def _system_prompt(resultado: dict, nombre: str, info_adicional: str) -> str:
    """Construye el system prompt con todos los datos del artista y contexto del dataset."""
    nivel = resultado["nivel"]
    proba = resultado["probabilidades"]
    f     = resultado["features"]

    tiene_datos_sl = int(f.get("sl_tiene_datos", 0))
    perfil_sl = "Sí aparece en setlist.fm" if tiene_datos_sl else "No aparece en setlist.fm"

    scrobbles_por_oyente = f.get("lfm_scrobbles_por_oyente", 0)
    if scrobbles_por_oyente > 30:
        engagement_lfm = "muy alto (base de fans muy fiel)"
    elif scrobbles_por_oyente > 10:
        engagement_lfm = "medio-alto"
    elif scrobbles_por_oyente > 3:
        engagement_lfm = "normal"
    else:
        engagement_lfm = "bajo (oyentes pasivos o pocos datos)"

    return f"""Eres un analista experto en la escena del rap y música urbana española.
Tu función es explicar predicciones de un modelo de Machine Learning que estima
el tier de sala que un artista puede llenar en España.

═══ SISTEMA DE TIERS ═══
• Bajo  (1): < 200 personas — underground, salas pequeñas (ej: Sala Víbora, Bar Berlin)
• Medio (2): 200 a 2.000 personas — salas medianas (ej: Planta Baja, Copera, La Riviera, Razzmatazz)
• Alto  (3): > 2.000 personas — palacios de deportes, WiZink, Movistar Arena, festivales grandes

═══ MODELO ═══
• Dataset: 182 artistas de rap/urbano español (bajo=85 · medio=65 · alto=32)
• Accuracy CV5: 68.2% | F1 macro: 66.6%  (baseline dummy: 39.5%)
• Features con más peso (SHAP): nº conciertos en setlist.fm · oyentes Last.fm ·
  vistas y suscriptores YouTube · si aparece o no en setlist.fm (sl_tiene_datos) ·
  scrobbles Last.fm · países donde ha actuado
• Clase más difícil de predecir: MEDIO (artistas frontera bajo/alto)

{_REFERENCIA_TIERS}

═══ PREDICCIÓN: {nombre.upper()} ═══
• Tier predicho  : {nivel.upper()}
• Probabilidades : Bajo {proba['bajo']:.1%} · Medio {proba['medio']:.1%} · Alto {proba['alto']:.1%}
• Confianza      : {"alta (≥70%)" if proba[nivel] >= 0.70 else "media (55–70%)" if proba[nivel] >= 0.55 else "baja (<55%) — caso frontera"}

═══ MÉTRICAS DEL ARTISTA ═══
• Spotify  : {f['sp_num_albums']:.0f} álbumes · {f['sp_num_singles']:.0f} singles · {f['sp_anos_activo']:.1f} años activo
             Ritmo: {f['sp_releases_por_ano']:.1f} lanzamientos/año
• Last.fm  : {f['lfm_oyentes']:,.0f} oyentes únicos · {f['lfm_scrobbles']:,.0f} scrobbles
             Engagement: {scrobbles_por_oyente:.1f} scrobbles/oyente → {engagement_lfm}
• YouTube  : {f['yt_suscriptores']:,.0f} suscriptores · {f['yt_vistas_totales']:,.0f} vistas totales
• Directo  : {f['sl_num_conciertos']:.0f} conciertos documentados · {perfil_sl}
             Países: {f.get('sl_num_paises', 0):.0f}

═══ INFO ADICIONAL (texto libre del usuario) ═══
{info_adicional.strip() if info_adicional and info_adicional.strip() else "No proporcionada."}

═══ NORMAS DE RESPUESTA ═══
- Responde SIEMPRE en español.
- Usa los datos reales del artista y compáralos con los rangos de la tabla de referencia.
- Para comparar con artistas concretos (ej: "¿cómo estoy respecto a Yung Beef?"),
  usa la tabla de referencia: Yung Beef es ALTO, con valores típicos de esa franja.
  No inventes cifras exactas de otros artistas si no las tienes.
- Sé concreto: menciona los números del artista y en qué rango de tier caen.
- El modelo tiene 68.2% de accuracy — reconoce su incertidumbre con honestidad.
- Máximo 250 palabras por respuesta salvo que el usuario pida más detalle.
- No inventes las respuestas, si no la sabes o no estás seguro, dilo.
- Básate únicamente en las métricas entrenadas de artistas urbanos españoles."""


def generar_explicacion(resultado: dict, nombre: str, info_adicional: str) -> str:
    """
    Genera la explicación inicial de la predicción.
    Se llama una sola vez al entrar en la página de Análisis IA.
    """
    llm = ChatGroq(model=_GROQ_MODEL, temperature=0.3)
    nivel  = resultado["nivel"]
    proba  = resultado["probabilidades"]
    f      = resultado["features"]

    segundo = sorted(proba.items(), key=lambda x: x[1], reverse=True)[1]
    es_frontera = proba[nivel] < 0.55

    contexto_frontera = (
        f" El modelo duda: también considera {segundo[0].upper()} con un {segundo[1]:.1%}."
        if es_frontera else ""
    )

    pregunta = f"""El modelo clasifica a {nombre} como tier "{nivel.upper()}" \
con una confianza del {proba[nivel]:.1%}.{contexto_frontera}

Proporciona un análisis estructurado con exactamente estas tres secciones:

**1. ¿Por qué {nivel.upper()}?**
Explica qué métricas concretas del artista justifican esta clasificación, \
comparándolas con los rangos típicos del tier {nivel} de la tabla de referencia. \
Menciona los números reales.

**2. Confianza de la predicción**
Interpreta las probabilidades ({proba['bajo']:.1%} bajo · {proba['medio']:.1%} medio · {proba['alto']:.1%} alto). \
¿Es un caso claro o un artista frontera?

**3. ¿Qué necesita para subir de tier?**
Basándote en las features más importantes del modelo (conciertos, oyentes Last.fm, YouTube), \
indica qué métricas necesitaría mejorar y en qué magnitud para llegar al siguiente nivel. \
Usa los rangos de la tabla de referencia como objetivo concreto."""

    messages = [
        SystemMessage(content=_system_prompt(resultado, nombre, info_adicional)),
        HumanMessage(content=pregunta),
    ]
    return llm.invoke(messages).content


def chat(
    historial: list[dict],
    pregunta: str,
    resultado: dict,
    nombre: str,
    info_adicional: str,
) -> str:
    """
    Responde una pregunta de seguimiento manteniendo el historial de conversación.

    historial : lista de {"role": "user"|"assistant", "content": str}
    pregunta  : nuevo mensaje del usuario
    """
    llm = ChatGroq(model=_GROQ_MODEL, temperature=0.4)

    messages = [SystemMessage(content=_system_prompt(resultado, nombre, info_adicional))]
    for msg in historial:
        cls = HumanMessage if msg["role"] == "user" else AIMessage
        messages.append(cls(content=msg["content"]))
    messages.append(HumanMessage(content=pregunta))

    return llm.invoke(messages).content
