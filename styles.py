"""
styles.py — Sistema de diseño compartido para todas las páginas.

Inyecta:
  - Fuentes Space Grotesk + Inter + JetBrains Mono (Google Fonts)
  - Material Symbols Rounded para iconos
  - Variables CSS y reglas para Streamlit (header, sidebar, métricas,
    cards, botones, formularios, chat, expanders, tabs, dataframes…)
  - Helpers (hero, metric_card, nav_card, badge, divider) que devuelven HTML
    para usar con `st.markdown(html, unsafe_allow_html=True)`.

Importar al principio de CADA página:

    from styles import inject_styles, hero, metric_card, nav_card, badge, divider, page_header
    inject_styles()

El tema base se fija en .streamlit/config.toml; este módulo lo extiende.
"""

import streamlit as st


PALETTE = {
    "bg": "#0E0B16",
    "bg_2": "#15101F",
    "surface": "#1A1428",
    "surface_2": "#221934",
    "line": "rgba(255,255,255,0.07)",
    "line_2": "rgba(255,255,255,0.12)",
    "text": "#F2EEFA",
    "text_2": "#B6ADCB",
    "text_3": "#7A7290",
    "violet": "#B026FF",
    "violet_2": "#7C18C2",
    "violet_glow": "rgba(176,38,255,0.35)",
    "lime": "#C6F432",
    "lime_2": "#9ED11A",
    "coral": "#FF5C7A",
    "amber": "#FFB547",
    "mint": "#00E0A4",
    "pink": "#FF3D9A",
    "blue": "#5B8DFF",
}

TIER_COLOR = {"bajo": PALETTE["coral"], "medio": PALETTE["amber"], "alto": PALETTE["mint"]}
TIER_ICON  = {"bajo": "trending_down", "medio": "trending_flat", "alto": "trending_up"}
TIER_DESC  = {
    "bajo":  "Sala pequeña · &lt; 200 personas",
    "medio": "Sala mediana · 200 – 2 000 personas",
    "alto":  "Sala grande · &gt; 2 000 personas · festivales",
}

# ---------------------------------------------------------------------------
# Big CSS blob — keep it here so every page has the same look without
# repeating markup.
# ---------------------------------------------------------------------------
_CSS = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@24,400,0,0&display=block">

<style>
:root{
  --bg:#0E0B16;
  --bg-2:#15101F;
  --surface:#1A1428;
  --surface-2:#221934;
  --line:rgba(255,255,255,0.07);
  --line-2:rgba(255,255,255,0.12);
  --text:#F2EEFA;
  --text-2:#B6ADCB;
  --text-3:#7A7290;
  --violet:#B026FF;
  --violet-glow:rgba(176,38,255,0.35);
  --lime:#C6F432;
  --lime-2:#9ED11A;
  --coral:#FF5C7A;
  --amber:#FFB547;
  --mint:#00E0A4;
  --pink:#FF3D9A;
  --blue:#5B8DFF;
}

/* ---------- Base ---------- */
html, body, [class*="css"]{
  font-family:'Inter', -apple-system, system-ui, sans-serif !important;
  color:var(--text);
}
.stApp{
  background:
    radial-gradient(900px 500px at 100% 0%, rgba(255,61,154,0.07), transparent 60%),
    radial-gradient(800px 600px at 0% 100%, rgba(176,38,255,0.07), transparent 60%),
    var(--bg);
}

/* hide default Streamlit chrome we don't want */
#MainMenu, footer, header[data-testid="stHeader"]{visibility:hidden; height:0;}
.block-container{padding-top:1.5rem !important; padding-bottom:5rem; max-width:1280px;}

/* ---------- Typography ---------- */
h1, h2, h3, h4{font-family:'Space Grotesk', sans-serif !important; letter-spacing:-0.02em; color:var(--text);}
h1{font-size:2.4rem !important; line-height:1.05;}
h2{font-size:1.6rem !important;}
h3{font-size:1.25rem !important;}
.stMarkdown p, .stMarkdown li{color:var(--text-2); line-height:1.6;}
.stMarkdown strong{color:var(--text);}
.stMarkdown code{
  background:rgba(176,38,255,0.12); color:#D26EFF;
  font-family:'JetBrains Mono', monospace; font-size:0.85em;
  padding:1px 6px; border-radius:4px;
}
.stCaption, .st-emotion-cache-1c7y2kd, [data-testid="stCaptionContainer"]{
  color:var(--text-3) !important;
  font-family:'JetBrains Mono', monospace !important;
  font-size:11px !important;
  text-transform:uppercase;
  letter-spacing:0.1em;
}

/* ---------- Sidebar ---------- */
section[data-testid="stSidebar"]{
  background:linear-gradient(180deg, #100B1C 0%, #0B0716 100%) !important;
  border-right:1px solid var(--line);
}
section[data-testid="stSidebar"] .stMarkdown{color:var(--text-2);}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3{font-size:0.95rem !important;}
[data-testid="stSidebarNav"] li a{
  border-radius:10px; padding:8px 10px; color:var(--text-2);
}
[data-testid="stSidebarNav"] li a:hover{background:var(--surface); color:var(--text);}
[data-testid="stSidebarNav"] li a[aria-current="page"]{
  background:linear-gradient(135deg, rgba(176,38,255,0.22), rgba(176,38,255,0.06));
  color:#fff;
  border:1px solid rgba(176,38,255,0.35);
}

/* ---------- Divider ---------- */
hr, [data-testid="stDivider"]{
  border-color:var(--line) !important;
  margin:1.4rem 0 !important;
}

/* ---------- Metric ---------- */
[data-testid="stMetric"]{
  background:linear-gradient(135deg, var(--surface) 0%, var(--surface-2) 100%);
  border:1px solid var(--line);
  border-radius:16px;
  padding:18px 20px;
  position:relative; overflow:hidden;
}
[data-testid="stMetric"]::before{
  content:""; position:absolute; width:180px; height:180px; left:-60px; top:-100px;
  background:radial-gradient(circle, var(--violet-glow) 0%, transparent 60%);
  pointer-events:none;
}
[data-testid="stMetricLabel"]{
  color:var(--text-3) !important;
  font-family:'JetBrains Mono', monospace !important;
  font-size:10.5px !important;
  text-transform:uppercase;
  letter-spacing:0.1em;
}
[data-testid="stMetricValue"]{
  font-family:'Space Grotesk', sans-serif !important;
  font-size:2.1rem !important;
  font-weight:600 !important;
  color:var(--text) !important;
  letter-spacing:-0.03em;
}
[data-testid="stMetricDelta"]{
  font-family:'JetBrains Mono', monospace !important;
  font-size:11px !important;
}

/* ---------- Buttons ---------- */
.stButton > button, .stDownloadButton > button, .stFormSubmitButton > button{
  background:var(--lime) !important;
  color:#1A1426 !important;
  border:none !important;
  border-radius:12px !important;
  font-weight:600 !important;
  font-family:'Inter', sans-serif !important;
  padding:0.65rem 1.2rem !important;
  box-shadow:0 8px 24px -8px rgba(198,244,50,0.5);
  transition:transform 0.1s ease, box-shadow 0.2s ease;
}
.stButton > button:hover, .stDownloadButton > button:hover, .stFormSubmitButton > button:hover{
  background:#D9FF4A !important; color:#1A1426 !important;
  transform:translateY(-1px);
  box-shadow:0 10px 28px -8px rgba(198,244,50,0.7);
}
.stButton > button[kind="secondary"]{
  background:rgba(255,255,255,0.05) !important;
  color:var(--text) !important;
  border:1px solid var(--line-2) !important;
  box-shadow:none;
}
.stButton > button[kind="secondary"]:hover{
  background:rgba(255,255,255,0.09) !important;
  border-color:rgba(176,38,255,0.35) !important;
}

/* ---------- Inputs ---------- */
.stTextInput input, .stNumberInput input, .stTextArea textarea, .stSelectbox > div > div, .stChatInput textarea{
  background:#0F0A1B !important;
  border:1px solid var(--line-2) !important;
  border-radius:10px !important;
  color:var(--text) !important;
  font-family:'Inter', sans-serif !important;
}
.stTextInput input:focus, .stNumberInput input:focus, .stTextArea textarea:focus{
  border-color:var(--violet) !important;
  box-shadow:0 0 0 3px rgba(176,38,255,0.18) !important;
}
.stNumberInput input{font-family:'JetBrains Mono', monospace !important;}
label, .stCheckbox label, .stRadio label{
  color:var(--text-2) !important;
  font-size:13px !important;
  font-weight:500 !important;
}

/* ---------- Alerts (info/success/warning/error) ---------- */
[data-testid="stAlert"]{
  border-radius:14px !important;
  border:1px solid var(--line) !important;
  padding:14px 18px !important;
  background:var(--surface) !important;
}
[data-testid="stAlert"][data-baseweb="notification"][kind="info"],
.stAlert.st-info, [data-testid="stNotification"]{
  border-left:3px solid var(--blue) !important;
}
.stSuccess, [data-testid="stAlert"][data-baseweb*="success"]{
  border-left:3px solid var(--mint) !important;
  background:rgba(0,224,164,0.06) !important;
}
.stWarning{
  border-left:3px solid var(--amber) !important;
  background:rgba(255,181,71,0.06) !important;
}
.stError{
  border-left:3px solid var(--coral) !important;
  background:rgba(255,92,122,0.06) !important;
}
.stInfo{
  border-left:3px solid var(--violet) !important;
  background:rgba(176,38,255,0.06) !important;
}

/* ---------- Container with border ---------- */
[data-testid="stVerticalBlockBorderWrapper"]{
  background:var(--surface);
  border:1px solid var(--line) !important;
  border-radius:18px !important;
  padding:18px 22px !important;
}

/* ---------- Expander ---------- */
[data-testid="stExpander"]{
  background:var(--surface);
  border:1px solid var(--line) !important;
  border-radius:14px !important;
  overflow:hidden;
}
[data-testid="stExpander"] summary{
  font-weight:600 !important;
  color:var(--text) !important;
  padding:12px 18px !important;
}
[data-testid="stExpander"] summary:hover{background:rgba(255,255,255,0.02);}

/* ---------- Status / spinner ---------- */
[data-testid="stStatusWidget"], [data-testid="stStatus"]{
  background:#0F0A1B !important;
  border:1px solid var(--line) !important;
  border-radius:14px !important;
  font-family:'JetBrains Mono', monospace !important;
  font-size:12px;
}
[data-testid="stStatus"] [data-testid="stMarkdownContainer"] p{
  color:var(--text-2) !important;
  font-family:'JetBrains Mono', monospace !important;
  font-size:12px !important;
  line-height:1.7 !important;
}

/* ---------- Tabs ---------- */
.stTabs [data-baseweb="tab-list"]{
  background:#0F0A1B;
  border:1px solid var(--line);
  border-radius:12px;
  padding:4px;
  gap:4px;
}
.stTabs [data-baseweb="tab"]{
  background:transparent !important;
  color:var(--text-3) !important;
  border-radius:9px !important;
  padding:8px 16px !important;
  font-family:'JetBrains Mono', monospace !important;
  font-size:11px !important;
  text-transform:uppercase;
  letter-spacing:0.08em;
  border:none !important;
}
.stTabs [data-baseweb="tab"][aria-selected="true"]{
  background:var(--surface) !important;
  color:var(--text) !important;
  box-shadow:inset 0 0 0 1px var(--line-2);
}
.stTabs [data-baseweb="tab-highlight"]{display:none;}

/* ---------- Progress ---------- */
[data-testid="stProgress"] > div > div{
  background:rgba(255,255,255,0.05) !important;
  border-radius:6px;
}
[data-testid="stProgress"] > div > div > div{
  background:linear-gradient(90deg, var(--violet), var(--pink), var(--lime)) !important;
  border-radius:6px;
}

/* ---------- Dataframes / Tables ---------- */
[data-testid="stDataFrame"], [data-testid="stTable"]{
  border:1px solid var(--line) !important;
  border-radius:14px !important;
  overflow:hidden;
}
.stMarkdown table{
  width:100%; border-collapse:collapse;
  background:var(--surface);
  border-radius:14px; overflow:hidden;
  border:1px solid var(--line);
  font-size:13px;
}
.stMarkdown table th{
  background:#15101F;
  text-align:left; padding:12px 14px;
  font-family:'JetBrains Mono', monospace;
  font-size:10.5px; text-transform:uppercase; letter-spacing:0.08em;
  color:var(--text-3); font-weight:500;
  border-bottom:1px solid var(--line);
}
.stMarkdown table td{
  padding:12px 14px;
  border-bottom:1px solid var(--line);
  color:var(--text-2);
}
.stMarkdown table tr:hover td{background:rgba(255,255,255,0.02);}
.stMarkdown table tr:last-child td{border-bottom:none;}
.stMarkdown table strong{color:var(--text);}

/* ---------- Chat ---------- */
[data-testid="stChatMessage"]{
  background:#15101F;
  border:1px solid var(--line) !important;
  border-radius:14px !important;
  padding:14px 16px !important;
}
[data-testid="stChatMessage"][data-testid*="user"]{
  background:linear-gradient(135deg, rgba(91,141,255,0.16), rgba(91,141,255,0.04)) !important;
  border-color:rgba(91,141,255,0.25) !important;
}
[data-testid="stChatMessage"] [data-testid="chatAvatarIcon-user"]{
  background:rgba(91,141,255,0.2) !important;
  color:var(--blue) !important;
}
[data-testid="stChatMessage"] [data-testid="chatAvatarIcon-assistant"]{
  background:rgba(176,38,255,0.18) !important;
  color:#D26EFF !important;
}
[data-testid="stChatInputContainer"]{
  background:#0F0A1B !important;
  border:1px solid var(--line) !important;
  border-radius:14px !important;
}

/* ---------- Image ---------- */
[data-testid="stImage"] img{
  border-radius:14px;
  border:1px solid var(--line);
}

/* ---------- Code block ---------- */
.stCodeBlock, pre, code{font-family:'JetBrains Mono', monospace !important;}
.stCodeBlock{
  background:#0A0712 !important;
  border:1px solid var(--line);
  border-radius:12px;
}

/* ---------- Form ---------- */
[data-testid="stForm"]{
  background:var(--surface);
  border:1px solid var(--line) !important;
  border-radius:18px !important;
  padding:24px 28px !important;
}

/* ---------- Toast ---------- */
[data-testid="stToast"]{
  background:var(--surface) !important;
  border:1px solid var(--line-2) !important;
  border-radius:12px !important;
}
</style>
"""


def inject_styles():
    """Inyecta CSS y fuentes. Llamar al inicio de cada página."""
    st.markdown(_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# HTML helpers — devuelven strings que se renderizan con
# st.markdown(html, unsafe_allow_html=True). NO añaden estilos
# adicionales aquí; toda la cosmética vive en _CSS.
# ---------------------------------------------------------------------------

def _ms(icon: str, color: str | None = None, size: int = 18) -> str:
    style = f"font-family:'Material Symbols Rounded';font-size:{size}px;vertical-align:-3px;"
    if color:
        style += f"color:{color};"
    return f'<span style="{style}">{icon}</span>'


def page_header(
    crumb: str,
    title_html: str,
    sub: str = "",
    pill_text: str | None = None,
    pill_color: str = "mint",
) -> str:
    """Cabecera unificada de página con migas, título + sub + pill opcional.

    `title_html` permite resaltar palabras con <em>…</em> (gradiente).
    `pill_color` ∈ {mint, violet, pink, amber}.
    """
    color_map = {
        "mint":   ("rgba(0,224,164,0.12)",  "rgba(0,224,164,0.3)",  "#00E0A4"),
        "violet": ("rgba(176,38,255,0.12)", "rgba(176,38,255,0.35)", "#D26EFF"),
        "pink":   ("rgba(255,61,154,0.12)", "rgba(255,61,154,0.3)", "#FF3D9A"),
        "amber":  ("rgba(255,181,71,0.12)", "rgba(255,181,71,0.3)", "#FFB547"),
    }
    bg, br, fg = color_map.get(pill_color, color_map["mint"])

    pill_html = ""
    if pill_text:
        pill_html = f"""
        <div style="display:inline-flex;align-items:center;gap:6px;
                    padding:5px 12px;border-radius:999px;
                    background:{bg};color:{fg};
                    font-family:'JetBrains Mono',monospace;font-size:10.5px;
                    letter-spacing:0.08em;border:1px solid {br};text-transform:uppercase;">
            <span style="width:6px;height:6px;border-radius:50%;background:{fg};box-shadow:0 0 8px {fg};"></span>
            {pill_text}
        </div>"""

    return f"""
    <div style="display:flex;align-items:flex-start;justify-content:space-between;gap:18px;margin:0 0 18px;">
      <div style="flex:1;">
        <div style="font-family:'JetBrains Mono',monospace;font-size:11px;color:var(--text-3);
                    text-transform:uppercase;letter-spacing:0.12em;margin-bottom:8px;">
            {_ms('home', size=14)} &nbsp;{crumb}
        </div>
        <h1 style="font-family:'Space Grotesk',sans-serif;font-size:2.4rem;margin:6px 0 8px;
                   letter-spacing:-0.025em;line-height:1.08;">
            {title_html}
        </h1>
        <p style="color:var(--text-2);font-size:14.5px;margin:0;max-width:680px;">{sub}</p>
      </div>
      <div>{pill_html}</div>
    </div>
    <style>
      h1 em{{
        font-style:normal;
        background:linear-gradient(135deg,#B026FF 0%, #FF3D9A 50%, #C6F432 110%);
        -webkit-background-clip:text;background-clip:text;color:transparent;
      }}
    </style>
    """


def hero(eyebrow: str, title_html: str, lead: str) -> str:
    """Bloque hero grande con grid de fondo y gradientes."""
    return f"""
    <div style="
      background:
        radial-gradient(800px 300px at 90% 0%, rgba(255,61,154,0.18), transparent 60%),
        radial-gradient(600px 400px at 0% 100%, rgba(176,38,255,0.18), transparent 60%),
        linear-gradient(135deg, #1F1632 0%, #150E24 100%);
      border:1px solid var(--line-2);border-radius:22px;padding:34px 38px;
      position:relative;overflow:hidden;margin:8px 0 28px;">
      <div style="position:absolute;inset:0;
                  background-image:linear-gradient(rgba(255,255,255,0.03) 1px, transparent 1px),
                                   linear-gradient(90deg, rgba(255,255,255,0.03) 1px, transparent 1px);
                  background-size:36px 36px;
                  mask-image:radial-gradient(circle at 30% 50%, black 0%, transparent 70%);"></div>
      <div style="position:relative;">
        <div style="font-family:'JetBrains Mono',monospace;font-size:11px;color:#C6F432;
                    text-transform:uppercase;letter-spacing:0.14em;margin-bottom:14px;">
            {_ms('bolt', color='#C6F432', size=14)} &nbsp;{eyebrow}
        </div>
        <h2 style="font-family:'Space Grotesk',sans-serif;font-size:2.6rem;line-height:1.05;
                   letter-spacing:-0.03em;margin:0 0 12px;">{title_html}</h2>
        <p style="color:var(--text-2);font-size:15px;line-height:1.55;max-width:680px;margin:0;">{lead}</p>
      </div>
    </div>
    """


def nav_card(num: str, icon: str, title: str, body: str, accent: str = "violet") -> str:
    """Tarjeta de navegación con número grande, icono, título y CTA.

    accent ∈ {violet, lime, pink}.
    """
    palettes = {
        "violet": ("rgba(176,38,255,0.18)", "rgba(176,38,255,0.02)", "rgba(176,38,255,0.3)", "#D26EFF"),
        "lime":   ("rgba(198,244,50,0.16)", "rgba(198,244,50,0.02)", "rgba(198,244,50,0.28)", "#C6F432"),
        "pink":   ("rgba(255,61,154,0.18)", "rgba(255,61,154,0.02)", "rgba(255,61,154,0.28)", "#FF3D9A"),
    }
    g1, g2, br, fg = palettes.get(accent, palettes["violet"])

    return f"""
    <div style="background:linear-gradient(160deg, {g1}, {g2} 70%);
                border:1px solid {br};border-radius:18px;padding:22px;
                position:relative;overflow:hidden;height:100%;">
      <span style="position:absolute;top:18px;right:22px;font-family:'JetBrains Mono',monospace;
                   font-size:11px;color:var(--text-3);letter-spacing:0.08em;">{num}</span>
      <div style="width:44px;height:44px;border-radius:12px;display:flex;align-items:center;justify-content:center;
                  background:{g1};color:{fg};font-family:'Material Symbols Rounded';font-size:24px;margin-bottom:14px;">
        {icon}
      </div>
      <h4 style="font-family:'Space Grotesk',sans-serif;font-size:17px;margin:0 0 6px;
                 letter-spacing:-0.01em;color:var(--text);">{title}</h4>
      <p style="color:var(--text-2);font-size:12.5px;line-height:1.5;margin:0 0 14px;">{body}</p>
      <div style="display:flex;align-items:center;gap:6px;color:{fg};font-size:11px;font-weight:600;
                  font-family:'JetBrains Mono',monospace;text-transform:uppercase;letter-spacing:0.08em;">
        Explorar {_ms('arrow_forward', color=fg, size=16)}
      </div>
    </div>
    """


def metric_card(label: str, value: str, delta: str = "", icon: str = "insights", color: str = "violet") -> str:
    """Métrica más rica que st.metric: con icono y color de acento.

    Útil cuando quieres una fila destacada en HTML; para usar dentro de
    st.columns(), basta con `st.markdown(metric_card(...), unsafe_allow_html=True)`.
    """
    color_hex = {
        "violet": "#B026FF", "lime": "#C6F432", "mint": "#00E0A4",
        "coral": "#FF5C7A", "amber": "#FFB547", "pink": "#FF3D9A", "blue": "#5B8DFF",
    }.get(color, "#B026FF")

    return f"""
    <div style="background:linear-gradient(135deg, var(--surface) 0%, var(--surface-2) 100%);
                border:1px solid var(--line);border-radius:16px;padding:18px 20px;
                position:relative;overflow:hidden;height:100%;">
      <div style="position:absolute;width:160px;height:160px;left:-40px;top:-90px;
                  background:radial-gradient(circle, {color_hex}38 0%, transparent 60%);pointer-events:none;"></div>
      <div style="position:relative;">
        <div style="font-family:'JetBrains Mono',monospace;font-size:10.5px;
                    text-transform:uppercase;letter-spacing:0.1em;color:var(--text-3);
                    display:flex;align-items:center;gap:6px;">
          {_ms(icon, color=color_hex, size=14)} {label}
        </div>
        <div style="font-family:'Space Grotesk',sans-serif;font-size:2rem;font-weight:600;
                    letter-spacing:-0.03em;line-height:1;margin-top:8px;color:var(--text);">{value}</div>
        <div style="font-family:'JetBrains Mono',monospace;font-size:10.5px;color:var(--mint);
                    margin-top:6px;">{delta}</div>
      </div>
    </div>
    """


def tier_chip(nivel: str, big: bool = False) -> str:
    """Chip de tier con punto coloreado. nivel ∈ {bajo, medio, alto}."""
    color = TIER_COLOR.get(nivel, PALETTE["violet"])
    size = "16px" if big else "13px"
    pad = "8px 14px" if big else "5px 10px"
    return f"""<span style="display:inline-flex;align-items:center;gap:8px;
       padding:{pad};border-radius:999px;background:{color}1f;color:{color};
       border:1px solid {color}55;font-family:'JetBrains Mono',monospace;
       font-size:{size};font-weight:600;text-transform:uppercase;letter-spacing:0.06em;">
       <span style="width:8px;height:8px;border-radius:50%;background:{color};box-shadow:0 0 8px {color};"></span>
       {nivel}
       </span>"""


def result_disc(percentage: int, color: str = "#00E0A4") -> str:
    """Disco circular con porcentaje. Para destacar la confianza."""
    return (
        f'<div style="width:120px;height:120px;border-radius:50%;'
        f'background:conic-gradient({color} 0% {percentage}%, rgba(255,255,255,0.08) {percentage}% 100%);'
        f'display:flex;align-items:center;justify-content:center;position:relative;">'
        f'<div style="position:absolute;inset:12px;background:#0E0918;border-radius:50%;"></div>'
        f'<b style="position:relative;font-family:\'Space Grotesk\';font-size:30px;color:{color};font-weight:600;">{percentage}%</b>'
        f'</div>'
    )


def divider(margin: str = "1.6rem"):
    """Separador discreto."""
    st.markdown(f'<div style="height:1px;background:var(--line);margin:{margin} 0;"></div>', unsafe_allow_html=True)


def section_label(text: str, icon: str = ""):
    """Etiqueta mono pequeña para encabezar bloques."""
    ic = f"{_ms(icon, color=PALETTE['lime'], size=14)} " if icon else ""
    st.markdown(
        f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:11px;'
        f'text-transform:uppercase;letter-spacing:0.12em;color:var(--text-3);margin:18px 0 8px;">'
        f'{ic}{text}</div>',
        unsafe_allow_html=True,
    )


def brand_block() -> str:
    """Marca para el sidebar (logo + nombre + tagline)."""
    return """
    <div style="display:flex;align-items:center;gap:10px;padding:6px 4px 16px;
                border-bottom:1px solid var(--line);margin-bottom:14px;">
      <div style="width:36px;height:36px;border-radius:11px;
                  background:conic-gradient(from 220deg at 50% 50%, #B026FF, #FF3D9A, #C6F432, #B026FF);
                  position:relative;display:flex;align-items:center;justify-content:center;
                  box-shadow:0 0 24px rgba(176,38,255,0.35);">
        <div style="position:absolute;inset:3px;background:#100B1C;border-radius:9px;"></div>
        <span style="position:relative;font-family:'Material Symbols Rounded';font-size:20px;color:#C6F432;">graphic_eq</span>
      </div>
      <div style="line-height:1.1;">
        <div style="font-family:'Space Grotesk';font-size:14px;font-weight:600;letter-spacing:-0.01em;color:#F2EEFA;">Tier Predictor</div>
        <div style="font-size:10px;color:#7A7290;font-family:'JetBrains Mono';text-transform:uppercase;letter-spacing:0.1em;">rap · urbano · ES</div>
      </div>
    </div>
    """
