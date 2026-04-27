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


def _system_prompt(resultado: dict, nombre: str, info_adicional: str) -> str:  # exportada para la traza del dashboard
    nivel = resultado["nivel"]
    proba = resultado["probabilidades"]
    f     = resultado["features"]

    return f"""Eres un analista experto en la escena del rap y música urbana española.
Tu función es explicar predicciones de un modelo de Machine Learning que estima
el tier de sala que un artista puede llenar.

═══ TIERS ═══
• Bajo  (1): < 200 personas — underground, salas pequeñas (ej: Sala Víbora)
• Medio (2): 200–2.000 personas — salas medianas (ej: Planta Baja, La Riviera, Razzmatazz)
• Alto  (3): > 2.000 personas — palacios de deportes, WiZink, festivales

═══ MODELO ═══
• Dataset: 157 artistas de rap/urbano español
• Accuracy CV5: 67.5% | F1 macro: 66.9%
• Features con más peso: nº conciertos documentados, oyentes Last.fm, vistas y suscriptores YouTube

═══ PREDICCIÓN: {nombre.upper()} ═══
• Tier predicho : {nivel.upper()}
• Probabilidades : Bajo {proba['bajo']:.1%} | Medio {proba['medio']:.1%} | Alto {proba['alto']:.1%}

═══ MÉTRICAS ═══
• Spotify : {f['sp_num_albums']:.0f} álbumes · {f['sp_num_singles']:.0f} singles · {f['sp_anos_activo']:.1f} años activo
• Last.fm  : {f['lfm_oyentes']:,.0f} oyentes · {f['lfm_scrobbles']:,.0f} scrobbles
• YouTube  : {f['yt_suscriptores']:,.0f} suscriptores · {f['yt_vistas_totales']:,.0f} vistas totales
• Directo  : {f['sl_num_conciertos']:.0f} conciertos documentados en setlist.fm

═══ INFO ADICIONAL ═══
{info_adicional.strip() if info_adicional and info_adicional.strip() else "No proporcionada."}

NORMAS:
- Responde SIEMPRE en español.
- Usa los datos reales del artista; no inventes cifras.
- Sé concreto y útil. Máximo 200 palabras por respuesta salvo que se pida más detalle.
- Cuando expliques qué podría mejorar, basa la respuesta en las features con más peso en el modelo.
- El modelo tiene un 67.5% de accuracy — reconoce su incertidumbre con honestidad."""


def generar_explicacion(resultado: dict, nombre: str, info_adicional: str) -> str:
    """
    Genera la explicación inicial de la predicción.
    Se llama una sola vez al entrar en la página de Análisis IA.
    """
    llm = ChatGroq(model=_GROQ_MODEL, temperature=0.3)
    nivel = resultado["nivel"]
    proba = resultado["probabilidades"]

    pregunta = f"""El modelo clasifica a {nombre} como tier "{nivel.upper()}" \
con una confianza del {proba[nivel]:.1%}.

Proporciona una explicación que incluya:
1. Por qué el modelo lo clasifica como {nivel} (qué métricas pesan más en este caso concreto)
2. Qué tan segura es la predicción (interpreta las probabilidades)
3. Qué podría mejorar el artista para subir de tier
Sé directo y concreto."""

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
    llm = ChatGroq(model=_GROQ_MODEL, temperature=0.3)

    messages = [SystemMessage(content=_system_prompt(resultado, nombre, info_adicional))]
    for msg in historial:
        cls = HumanMessage if msg["role"] == "user" else AIMessage
        messages.append(cls(content=msg["content"]))
    messages.append(HumanMessage(content=pregunta))

    return llm.invoke(messages).content
