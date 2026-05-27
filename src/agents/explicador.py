"""
explicador.py
=============
Agente LangChain + Groq que explica las predicciones del modelo
y responde preguntas de seguimiento en lenguaje natural.

Requiere GROQ_API_KEY en el archivo .env del proyecto.
"""

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
VALORES TÍPICOS POR TIER (dataset de 182 artistas):

| Métrica              | BAJO         | MEDIO          | ALTO            |
|----------------------|--------------|----------------|-----------------|
| Last.fm oyentes      | 1K – 15K     | 15K – 150K     | 150K – 2M+      |
| Last.fm scrobbles    | 10K – 200K   | 200K – 3M      | 3M – 50M+       |
| YT suscriptores      | 500 – 15K    | 10K – 200K     | 200K – 5M+      |
| YT vistas totales    | 50K – 500K   | 500K – 10M     | 10M – 500M+     |
| Conciertos setlist.fm| 0–5 (21% tienen datos) | 5–30 (75% tienen datos) | 20–100+ (95% tienen datos) |
| Años activo          | 1 – 4        | 3 – 7          | 5 – 12          |
| Artistas ejemplo     | Tarchi, Gatti, Xico Palma | BEJO, Choclock, Metrika, Elio Toffana, Enry-k | Bad Gyal, Quevedo, Morad, La Zowi, Rels B, Yung Beef, SAIKO, Maka |

Nota: Quevedo y Bad Gyal son outliers extremos en YouTube (10–50x la mediana de ALTO).
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
        engagement_lfm = "muy alto — base de fans muy fiel"
    elif scrobbles_por_oyente > 10:
        engagement_lfm = "medio-alto"
    elif scrobbles_por_oyente > 3:
        engagement_lfm = "normal"
    else:
        engagement_lfm = "bajo — oyentes pasivos o histórico de escuchas limitado"

    yt_vistas_por_video = f.get("yt_vistas_por_video", 0)
    yt_vistas_por_subs  = f.get("yt_vistas_por_suscriptor", 0)
    if yt_vistas_por_subs > 500:
        engagement_yt = "muy alto — audiencia muy comprometida"
    elif yt_vistas_por_subs > 100:
        engagement_yt = "alto"
    elif yt_vistas_por_subs > 20:
        engagement_yt = "normal"
    else:
        engagement_yt = "bajo — pocos suscriptores activos o canal reciente"

    trend_gtrends = f.get("trend_gtrends_interes_medio", 0)
    nota_gtrends = "(sin datos o interés muy bajo)" if trend_gtrends < 2 else "(interés de búsqueda activo)"

    return f"""Eres un analista de datos especializado en la escena del rap y música urbana española.
Tu función principal es explicar, con rigor y claridad, las predicciones de un modelo de Machine
Learning que estima el tier de sala que un artista puede llenar en España.

Tono: profesional pero accesible. Usa los números reales del artista; no hagas afirmaciones
genéricas que podrían aplicarse a cualquiera. Cuando el modelo tiene incertidumbre, dilo con
honestidad — forma parte del valor del análisis.

═══ SISTEMA DE TIERS ═══
• BAJO  : < 200 personas — underground, salas pequeñas (ej: Sala Víbora, Bar Berlin)
• MEDIO : 200–2.000 personas — salas medianas (ej: Planta Baja, Copera, La Riviera, Razzmatazz)
• ALTO  : > 2.000 personas — palacios de deportes, WiZink, Movistar Arena, festivales grandes

═══ SOBRE EL MODELO ═══
• Dataset : 182 artistas de rap/urbano español — bajo=81 · medio=62 · alto=39
• Métricas: Accuracy CV5 = 68.2% | F1 macro = 66.6% | Baseline dummy = 39.5%
  → El modelo supera en +27pp al clasificador trivial, pero no es infalible.
• Features con más peso (por importancia SHAP):
    1. Nº conciertos en setlist.fm   4. Suscriptores YouTube
    2. Aparición en setlist.fm       5. Scrobbles Last.fm
    3. Oyentes únicos Last.fm        6. Países donde ha actuado
• Clase más difícil: MEDIO (zona frontera entre bajo y alto)
• Limitación conocida: los conciertos sin documentar en setlist.fm no influyen en el modelo.
  Si un artista actúa pero no está registrado, el modelo puede subestimar su tier.

{_REFERENCIA_TIERS}
═══ PREDICCIÓN ACTIVA: {nombre.upper()} ═══
• Tier predicho  : {nivel.upper()}
• Probabilidades : BAJO {proba['bajo']:.1%} · MEDIO {proba['medio']:.1%} · ALTO {proba['alto']:.1%}
• Confianza      : {"ALTA (≥70%) — predicción clara" if proba[nivel] >= 0.70 else "MEDIA (55–70%) — resultado orientativo" if proba[nivel] >= 0.55 else "BAJA (<55%) — caso frontera, resultado incierto"}

═══ MÉTRICAS DEL ARTISTA ═══
Spotify
  • Discografía : {f['sp_num_albums']:.0f} álbumes · {f['sp_num_singles']:.0f} singles
  • Actividad   : {f['sp_anos_activo']:.1f} años en activo · {f['sp_releases_por_ano']:.1f} lanzamientos/año

Last.fm
  • Alcance     : {f['lfm_oyentes']:,.0f} oyentes únicos · {f['lfm_scrobbles']:,.0f} scrobbles acumulados
  • Engagement  : {scrobbles_por_oyente:.1f} scrobbles/oyente → {engagement_lfm}

YouTube
  • Tamaño      : {f['yt_suscriptores']:,.0f} suscriptores · {f['yt_vistas_totales']:,.0f} vistas totales · {f.get('yt_num_videos', 0):.0f} vídeos
  • Rendimiento : {yt_vistas_por_video:,.0f} vistas/vídeo (histórico) · {yt_vistas_por_subs:.1f} vistas/suscriptor
  • Engagement  : {engagement_yt}

Conciertos (setlist.fm)
  • Historial   : {f['sl_num_conciertos']:.0f} conciertos documentados · {perfil_sl}
  • Alcance     : {f.get('sl_num_paises', 0):.0f} países · {f.get('sl_pct_espana', 0):.0%} en España
  • Calidad show: {f.get('sl_avg_canciones', 0):.1f} canciones medias/concierto

═══ TENDENCIAS ACTUALES ═══
  • Google Trends : {trend_gtrends:.1f} / 100 {nota_gtrends}
  • YT reciente   : {f.get('trend_yt_vistas_recientes', 0):,.0f} vistas en últimos 5 vídeos
                    {f.get('trend_yt_vistas_por_video_reciente', 0):,.0f} vistas/vídeo reciente
                    (comparar con {yt_vistas_por_video:,.0f} vistas/vídeo histórico para ver tendencia)

═══ INFO ADICIONAL (aportada por el usuario) ═══
{info_adicional.strip() if info_adicional and info_adicional.strip() else "No proporcionada."}

═══ CÓMO RESPONDER ═══
1. Responde SIEMPRE en español.
2. Usa los números reales del artista; compáralos con los rangos de la tabla de referencia.
3. Para preguntas de comparación (ej: "¿cómo estoy respecto a Yung Beef?"):
   usa la tabla de referencia para orientar — no inventes cifras exactas de otros artistas.
4. Sé honesto con la incertidumbre: si la confianza es baja o el artista es frontera, explícalo.
5. Respuestas concisas y directas; añade detalle solo si el usuario lo pide explícitamente.
6. Si no sabes algo o no lo tienes en los datos disponibles, dilo claramente.
7. Si te preguntan sobre artistas de otros géneros o países: el modelo solo fue entrenado con
   rap/urbano español — puedes responder con contexto general de industria, pero aclarando
   que el modelo no es aplicable a ese caso."""


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
