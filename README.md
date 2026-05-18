# Predicción Musical TFM

## Descripción del proyecto

Proyecto fin de máster para predecir el **tier de sala** de artistas de rap/urbano español — es decir, el tamaño máximo de recinto que un artista puede llenar en España. El dataset cubre **182 artistas** de la escena urbana española, desde artistas completamente underground hasta nombres con presencia en festivales y palacios de deportes.

El pipeline recoge datos de 6 fuentes (Spotify, Last.fm, YouTube, setlist.fm, Google Trends + YouTube reciente) mediante sus APIs, aplica ingeniería de características, y entrena modelos de clasificación multiclase. Los resultados se analizan en notebooks Jupyter.

El foco está en artistas **poco conocidos o emergentes**, lo que implica datos escasos o inexistentes en las APIs estándar y requiere validación y corrección manual.

---

## Estructura del proyecto

```
app.py               # Página de inicio del dashboard (Landing)
pages/
  1_Resultados.py    # Benchmark de modelos, confusión, feature importance y SHAP global
  2_Prediccion.py    # Formulario de predicción con traza detallada y waterfall SHAP
  3_Analisis_IA.py   # Chat con agente LangChain + Groq
src/
  data_collectors/   # Clientes de API para ingesta de datos raw
    spotify.py           # Búsqueda de artistas en Spotify (usado solo por el resolver)
    spotify_features.py  # Discografía y top tracks usando IDs del registry
    lastfmapi.py         # Oyentes, scrobbles y géneros usando nombres del registry
    youtube.py           # Estadísticas de canal usando channel_id del registry
    setlistfm.py         # Historial de conciertos usando MBID del registry
    tendencias.py        # Google Trends + YouTube reciente
  features/          # Ingeniería de características y preprocesado
  models/            # Entrenamiento, evaluación y serialización de modelos
    train.py         # Entrena XGBoost tuned y serializa modelo + metadata
    predict.py       # Inferencia: 8 inputs → 32 features → tier predicho
    shap_explainer.py# Explicabilidad SHAP: plots globales e individuales
  agents/
    explicador.py    # Agente LangChain: explicación inicial + chat de seguimiento
  utils/
    log_streamlit.py # Logging de sesión y sidebar persistente
    feature_labels.py# Nombres amigables, agrupación, formateo y detección de alertas
config/
  registry.py              # Utilidad cargar()/guardar() para el registry
  artistas_registry.csv    # ★ Fuente de verdad de identidades entre plataformas (en git)
  artistas_labels.csv      # ★ Variable objetivo — tiers curados manualmente (en git)
scripts/
  00_resolver_identidades.py  # Resuelve IDs de plataformas para artistas nuevos
  migrar_registry.py          # Migración one-time desde CSVs existentes al registry
  pipeline.py                 # Orquestador completo: recolección → features → entrenamiento
  generar_labels.py           # Etiquetado semisupervisado desde historial de venues
  generar_shap.py             # Genera plots SHAP globales en reports/figures/
  validar_setlistfm.py        # Valida si el 41% de NaN en sl_num_conciertos es un artefacto
notebooks/           # Análisis exploratorio y modelado
  01_eda.ipynb       # EDA completo con Kruskal-Wallis y correlaciones
  02_modelado.ipynb  # Entrenamiento y comparativa de modelos
  03_hiperparametros.ipynb  # Ajuste de hiperparámetros XGBoost con RandomizedSearchCV
tests/               # Suite de tests — 114 tests, 114 passed
  conftest.py        # Fixtures compartidos (DataFrames en memoria, mock del modelo)
  test_build_features.py   # Tests de las 6 funciones _features_*()
  test_preprocess.py # Tests de _imputar() y cargar_datos()
  test_predict.py    # Tests de construir_features() y predecir()
  test_shap_explainer.py   # Tests de shap_waterfall_fig() y constantes
pytest.ini           # Configuración de pytest
models/              # Artefactos entrenados (gitignored)
  xgb_tuned.joblib   # Modelo XGBoost serializado con joblib
  metadata.json      # Features, parámetros, métricas CV y timestamp del entrenamiento
reports/
  figures/           # PNGs de notebooks + plots SHAP (shap_summary_bar, shap_beeswarm_alto)
data/
  raw/               # CSVs originales de cada fuente (gitignored)
  processed/         # artist_features.csv — matriz de features unificada (gitignored)
```

---

## Problema de Machine Learning

### ¿Qué queremos predecir? — Variable objetivo `y`

> **Dado un artista de rap/urbano español, ¿qué tamaño de sala puede llenar?**

```
y = nivel  →  bajo (1) · medio (2) · alto (3)
```

Es un problema de **clasificación multiclase ordinal**: las 3 clases tienen orden natural (bajo < medio < alto), pero la distancia entre ellas no es uniforme.

| Clase | Valor | Aforo equivalente | Ejemplos de artistas |
|-------|-------|-------------------|----------------------|
| `bajo` | 1 | < 200 personas | Tarchi, Gatti, Xico Palma |
| `medio` | 2 | 200 – 2.000 personas | BEJO, Choclock, Metrika, La Zowi |
| `alto` | 3 | > 2.000 personas (festivales, palacios) | Bad Gyal, Quevedo, Morad, Rels B |

**Distribución**: bajo=81 · medio=62 · alto=39 · **total=182 artistas**

La variable objetivo está en `artistas_labels.csv` como `nivel` (texto) y `target` (entero 1/2/3).
`nombre_buscado` es el identificador del artista — **nunca entra como variable predictora**.

---

### ¿Con qué lo predecimos? — Variables predictoras `X`

Señales de presencia digital, actividad musical y trayectoria recogidas de 6 fuentes:

| Fuente | Features | ¿Qué mide? |
|--------|----------|------------|
| **Spotify** (discografía) | 6 | Madurez de carrera: años activo, volumen de lanzamientos, ritmo de publicación |
| **Spotify** (top tracks) | 3 | Estilo: duración de tracks, % explicit, % colaboraciones |
| **Last.fm** | 6 | Audiencia histórica: oyentes únicos, scrobbles totales, engagement de fans |
| **YouTube** | 7 | Presencia digital: suscriptores, vistas, engagement del canal |
| **setlist.fm** | 6 | Actividad en directo: conciertos, países, % en España, flag de presencia |
| **Tendencias** (Google Trends + YT reciente) | 5 | Buzz actual: interés de búsqueda y vistas recientes |

**Total tras preprocesado** (`src/features/preprocess.py`):

| Modo | Features | Escalado | Modelos |
|------|----------|----------|---------|
| `arbol` | **32** | Sin escalar | Random Forest, XGBoost, LightGBM |
| `lineal` | **24** | RobustScaler | Regresión Logística, SVM, Regresión Ordinal |

En modo `lineal` se eliminan adicionalmente las versiones raw cuando existe versión log (`lfm_oyentes`, `lfm_scrobbles`, `yt_suscriptores`, `yt_vistas_totales`, `trend_yt_vistas_recientes`, `trend_gtrends_interes_medio`), `sp_pct_colabs` (r≈−0.02, solo añade ruido), y `trend_yt_vistas_por_video_reciente` (r=0.996 con `trend_yt_vistas_recientes_log` — multicolinealidad casi perfecta).

Features descartadas en ambos modos por el EDA (señal no significativa):
`sp_num_eps` · `sp_num_total_releases` · `sp_dias_desde_ultimo_release` · `yt_edad_canal_anos` · `trend_yt_videos_recientes`

El preprocesado no genera ningún fichero — devuelve `X`, `y` en memoria listos para entrenar.
Verificado: **NaN en X = 0** en ambos modos sobre los 182 artistas.

---

## Registry de artistas (`config/artistas_registry.csv`)

Fuente de verdad única para identidades de artistas entre plataformas. **Este fichero está en git** — contiene las resoluciones de nombre/ID para cada plataforma, incluidas las correcciones manuales, de forma que nunca se pierden al clonar el repositorio.

| Columna | Descripción |
|---------|-------------|
| `nombre_canonico` | Nombre canónico del artista (clave primaria) |
| `spotify_id` / `spotify_nombre` | Artist ID y nombre tal como lo devuelve Spotify |
| `lastfm_nombre` / `lastfm_mbid` | Nombre y MusicBrainz ID en Last.fm |
| `yt_channel_id` / `yt_nombre_canal` | ID y nombre del canal de YouTube |
| `sl_nombre` / `sl_mbid` | Nombre y MBID en setlist.fm |
| `manual_override` | `True` = no sobreescribir automáticamente |
| `notas` | Motivo de corrección o artistas sin canal |

### Pipeline de recolección

```
artistas.txt
    │
    ▼
scripts/00_resolver_identidades.py   ← llama a las 4 APIs, umbral alto de confianza
    │
    ├── Resuelto → config/artistas_registry.csv  (manual_override=False)
    └── No resuelto → pendientes_revision.csv    (rellenar a mano → manual_override=True)
    │
    ▼
scripts/pipeline.py                  ← usa IDs del registry, nunca busca por nombre
    │
    ├── src/data_collectors/*.py     → data/raw/*.csv
    ├── scripts/generar_labels.py    → config/artistas_labels.csv
    ├── src/features/build_features  → data/processed/artist_features.csv
    └── src/models/train.py          → models/xgb_tuned.joblib
```

**Para añadir artistas nuevos:**
```bash
# 1. Añadir nombres a artistas.txt
# 2. Resolver identidades (llama a las 4 APIs)
python scripts/00_resolver_identidades.py
# 3. Revisar pendientes_revision.csv y completar artistas_registry.csv a mano
# 4. Ejecutar el pipeline
python scripts/pipeline.py
```

Los colectores son **incrementales**: si un artista ya tiene datos en el CSV, se salta. Los datos corregidos manualmente nunca se sobreescriben.

---

## Fuentes de datos

Todos los colectores usan los IDs de `config/artistas_registry.csv` directamente, sin búsquedas por nombre, lo que elimina los fallos de matching entre plataformas.

### Spotify Features (`src/data_collectors/spotify_features.py`)

Usa `spotify_id` del registry para extraer features de discografía y top tracks.

- `data/raw/spotify_discografia.csv` — una fila por artista: `num_albums`, `num_singles`, `num_eps`, `num_total_releases`, `primer_lanzamiento`, `ultimo_lanzamiento`, `anos_activo`, `releases_por_ano`
- `data/raw/spotify_top_tracks.csv` — una fila por canción (top 10/artista): `track_name`, `duration_ms`, `explicit`, `num_artistas`

> `popularity` y `audio_features` (`danceability`, `energy`, `valence`...) eliminados — Spotify los bloqueó en noviembre 2024.

### Last.fm (`src/data_collectors/lastfmapi.py`)

Fuente principal para **oyentes, scrobbles y géneros**, ya que Spotify no los proporciona.
Usa `lastfm_nombre` del registry directamente (llamada `artist.getInfo` sin búsqueda previa).

- CSV de salida: `data/raw/lastfm_artistas.csv`
- Columnas: `nombre_buscado`, `nombre_lastfm`, `oyentes`, `scrobbles`, `generos`, `lastfm_url`, `mbid`

**Cómo interpretar los datos de Last.fm:**
- `oyentes` = usuarios únicos históricos, no mensuales. Los números son menores que en Spotify porque Last.fm tiene menos usuarios.
- La correlación relativa entre artistas es válida: los artistas más populares en Spotify también lideran en Last.fm.
- `scrobbles` = reproducciones acumuladas totales. Buen indicador de **engagement del fan** — pocos oyentes con muchos scrobbles = base muy fiel.

### YouTube (`src/data_collectors/youtube.py`)

Presencia digital en YouTube: tamaño de audiencia, actividad y engagement.
Usa `yt_channel_id` del registry directamente (una sola llamada `channels.list`, ~1 unidad de quota por artista frente a ~100 unidades de la búsqueda anterior).

- CSV de salida: `data/raw/youtube_artistas.csv`
- Columnas: `nombre_buscado`, `channel_id`, `suscriptores`, `vistas_totales`, `num_videos`, `fecha_creacion`, `pais_canal`, `descripcion_canal`

**Correcciones manuales**: varios artistas tenían canales incorrectos (BEJO, Rels B, Morad, Dano, Cecilio G, Choclock, Yung Beef, entre otros). Las correcciones están en `config/artistas_registry.csv` con `manual_override=True`. Dos artistas (Xico Palma, GREKKY) no tienen canal — su `yt_channel_id` es nulo en el registry y sus features se imputan como 0.

### Tendencias (`src/data_collectors/tendencias.py`)

Combina dos fuentes para medir el **buzz actual** de cada artista, con especial atención a artistas emergentes donde otras señales son escasas:

- **Google Trends** (`pytrends`): interés de búsqueda semanal normalizado (0-100) para España en los últimos 12 meses.
- **YouTube vídeos recientes** (YouTube Data API v3): vistas acumuladas de los últimos 5 vídeos del artista.

- CSV de salida: `data/raw/tendencias.csv`
- Columnas: `nombre_buscado`, `gtrends_interes_medio`, `gtrends_pico_maximo`, `yt_vistas_recientes`, `yt_videos_recientes`, `yt_vistas_por_video_reciente`

#### Por qué se descartó TikTok

TikTok es hoy la principal plataforma de viralización para artistas emergentes del rap/urbano español. Sin embargo, no existe una API pública accesible: la TikTok Research API requiere aprobación institucional, y las librerías no oficiales van contra los Términos de Servicio. Se optó por YouTube reciente como proxy.

### setlist.fm (`src/data_collectors/setlistfm.py`)

Historial de conciertos (setlists) por artista. Fuente más directa para medir actividad en directo.

- CSV de salida: `data/raw/setlistfm_conciertos.csv`
- Columnas: `nombre`, `setlistfm_mbid`, `setlist_id`, `fecha`, `venue_id`, `venue_nombre`, `ciudad`, `pais`, `num_canciones`, `tiene_encore`, `canciones`, `url_setlist`
- Lógica incremental: omite artistas ya procesados en ejecuciones anteriores.

---

## Etiquetas del modelo (`config/artistas_labels.csv`)

Fichero central con la variable objetivo. Los tiers originales (0-4) se fusionaron en 3 niveles para garantizar suficientes ejemplos por clase:

| `tier_sala` | `nivel` | Aforo equivalente | Descripción |
|-------------|---------|-------------------|-------------|
| 1 | `bajo` | < 200 personas | Sala Víbora · artistas sin sala propia |
| 2 | `medio` | 200 – 2.000 personas | Planta Baja, Copera (Granada) · El Sol, Razzmatazz, La Riviera |
| 3 | `alto` | > 2.000 personas | Palacio de Deportes · WiZink, Movistar Arena, estadios, festivales |

### Proceso de etiquetado

El etiquetado es **semisupervisado**: ~50% de las etiquetas se infirieron automáticamente desde el historial de venues de setlist.fm (`scripts/generar_labels.py`), y el resto se asignaron manualmente basándose en conocimiento directo de la escena musical española.

---

## Feature Engineering (`src/features/build_features.py`)

Combina todas las fuentes raw en la matriz `data/processed/artist_features.csv`.
**182 artistas × 36 features** (+ 4 columnas de metadata/target).

### Features por fuente

#### Spotify — Discografía
| Feature | Descripción |
|---------|-------------|
| `sp_num_albums` | Número de álbumes |
| `sp_num_singles` | Número de singles |
| `sp_num_eps` | Número de EPs |
| `sp_num_total_releases` | Total de lanzamientos |
| `sp_anos_activo` | Años desde el primer lanzamiento |
| `sp_releases_por_ano` | Cadencia media de lanzamientos por año |
| `sp_dias_desde_ultimo_release` | Días desde el último lanzamiento |
| `sp_ratio_albums_singles` | `albums / (singles+1)` — carrera consolidada vs streaming-first |

#### Spotify — Top Tracks
| Feature | Descripción |
|---------|-------------|
| `sp_avg_duration_ms` | Duración media de los tracks |
| `sp_pct_explicit` | % de tracks con contenido explícito |
| `sp_pct_colabs` | % de tracks en colaboración con otros artistas |

#### Last.fm
| Feature | Descripción |
|---------|-------------|
| `lfm_oyentes` | Oyentes históricos únicos |
| `lfm_oyentes_log` | Versión log (corrige asimetría) |
| `lfm_scrobbles` | Reproducciones totales acumuladas |
| `lfm_scrobbles_log` | Versión log |
| `lfm_scrobbles_por_oyente` | Engagement: fidelidad media del fan |
| `lfm_num_generos` | Número de géneros asociados |

#### YouTube
| Feature | Descripción |
|---------|-------------|
| `yt_suscriptores` | Suscriptores del canal |
| `yt_suscriptores_log` | Versión log |
| `yt_vistas_totales` | Vistas totales acumuladas |
| `yt_vistas_log` | Versión log |
| `yt_num_videos` | Número de vídeos publicados |
| `yt_vistas_por_video` | Media de vistas por vídeo |
| `yt_vistas_por_suscriptor` | Ratio vistas/suscriptores (engagement) |
| `yt_edad_canal_anos` | Años desde la creación del canal |

#### setlist.fm
| Feature | Descripción |
|---------|-------------|
| `sl_num_conciertos` | Total de conciertos registrados |
| `sl_avg_canciones` | Media de canciones por setlist |
| `sl_pct_encore` | % de conciertos con encore |
| `sl_num_paises` | Países distintos donde ha actuado |
| `sl_pct_espana` | % de conciertos en España |
| `sl_tiene_datos` | **Flag binario**: 1 si el artista aparece en setlist.fm, 0 si no |

#### Tendencias (Google Trends + YouTube reciente)
| Feature | Descripción |
|---------|-------------|
| `trend_gtrends_interes_medio` | Interés medio de búsqueda en España (últimos 12 meses) |
| `trend_gtrends_log` | Versión log |
| `trend_yt_vistas_recientes` | Vistas acumuladas de los últimos 5 vídeos |
| `trend_yt_vistas_recientes_log` | Versión log |
| `trend_yt_videos_recientes` | Número de vídeos recientes con datos |
| `trend_yt_vistas_por_video_reciente` | Media de vistas por vídeo reciente |

### Imputación de NaN

| Columnas | NaN aprox. | Causa | Imputación |
|----------|------------|-------|------------|
| `sl_tiene_datos` | — | Flag creado antes de imputar | **1** si `sl_num_conciertos` no era NaN, **0** si lo era |
| `sl_num_conciertos`, `sl_num_paises` | 41% | Sin registros en setlist.fm | **0** (sin conciertos = 0 conciertos) |
| `sl_avg_canciones`, `sl_pct_encore`, `sl_pct_espana` | 41% | Sin registros en setlist.fm | **Mediana global** (0 implicaría setlist vacío) |
| `yt_vistas_totales`, `yt_vistas_log`, `yt_num_videos`, `yt_vistas_por_*` | ~5% | Canales corregidos sin datos de vistas | **Mediana por nivel** |
| `trend_gtrends_*` | ~5% | Fallos de rate-limit | **Mediana por nivel** |
| `trend_yt_vistas_recientes*`, `trend_yt_videos_recientes` | ~5% | Sin canal YouTube | **0** |
| `sp_anos_activo`, `sp_releases_por_ano` | <4% | Sin lanzamientos en Spotify | **0** |
| `yt_suscriptores`, `lfm_*` | <1% | Sin canal o sin perfil | **0** |

---

## Análisis exploratorio (`notebooks/01_eda.ipynb`)

El EDA analiza las 36 features y su relación con el target usando **Kruskal-Wallis** (H, discriminación entre clases) y **correlación de Spearman** (r, fuerza y dirección):

**27 de 36 features son significativas** (Kruskal-Wallis p < 0.05).

| Feature | H (Kruskal-Wallis) | r (Spearman) | Conclusión |
|---------|-------------------|--------------|------------|
| `lfm_oyentes` | 61.6 | +0.63 | Feature más discriminativa — audiencia histórica |
| `lfm_scrobbles` | 59.2 | +0.62 | Alta señal — engagement acumulado |
| `yt_suscriptores` | 53.3 | +0.59 | Muy alta señal — canal digital |
| `yt_vistas_totales` | 51.5 | +0.57 | Muy alta señal |
| `yt_vistas_por_video` | 46.6 | +0.55 | Calidad del contenido |
| `sl_num_conciertos` | — | +0.47 | Actividad en directo — feature #1 en importancia de modelos |
| `sp_num_eps` | ~0 | ~0 | Sin varianza útil — eliminada |
| `yt_edad_canal_anos` | ~1 | +0.10 | No significativa — eliminada |
| `trend_yt_videos_recientes` | — | — | Varianza casi nula (>95% tienen 5 vídeos) — eliminada |

**Pares altamente correlacionados** (|r| > 0.85): raw↔log (r≈1.0); `trend_yt_vistas_por_video_reciente`↔`trend_yt_vistas_recientes_log` (r=0.996); `lfm_scrobbles`↔`lfm_oyentes` (r=0.913); `sl_pct_espana`↔`sl_num_paises` (r=−0.961). Las versiones raw se eliminan en modo lineal para evitar multicolinealidad.

**Outliers extremos** (Quevedo, Bad Gyal, Morad): valores de YouTube 10-50× superiores a la mediana de `alto`. Se mantienen como datos válidos. Se usa RobustScaler en modo lineal para mitigar su efecto.

---

## Modelado (`notebooks/02_modelado.ipynb`)

### Estrategia de validación

**StratifiedKFold k=5**: con n=182 artistas un hold-out del 20% dejaría solo ~36 artistas de test. Con CV k=5 cada artista aparece exactamente **una vez en test**, sin data leakage. Métricas reportadas: media ± desviación estándar de los 5 folds.

### Resultados (182 artistas, mayo 2026)

| Modelo | Features | Accuracy (CV5) | F1 macro (CV5) | Estabilidad |
|--------|----------|---------------|----------------|-------------|
| Dummy most_frequent | — | 44.5% ± 0.9% | 20.5% ± 0.3% | — |
| Dummy stratified | — | 38.4% ± 8.3% | 31.4% ± 6.0% | — |
| Regresión Logística | 24 (lineal) | 59.4% ± 7.5% | 56.3% ± 8.1% | ⚠️ variable |
| Regresión Ordinal (mord) | 24 (lineal) | 60.4% ± 4.9% | 56.4% ± 4.5% | ✅ estable |
| Random Forest | 32 (arbol) | 61.6% ± 3.2% | 58.5% ± 3.5% | ✅ estable |
| LightGBM | 32 (arbol) | 61.0% ± 5.7% | 57.6% ± 5.0% | ⚠️ variable |
| XGBoost | 32 (arbol) | 63.2% ± 3.9% | 60.3% ± 3.7% | ✅ estable |
| **SVM (kernel RBF)** | **24 (lineal)** | **64.8% ± 7.6%** | **62.3% ± 8.0%** | ⚠️ variable |

**SVM tiene el mayor F1 macro** (+41.8 puntos sobre el baseline). **Random Forest sigue siendo el más estable** (±3.2%). XGBoost equilibra rendimiento y estabilidad — candidato para tuning.

La **Regresión Ordinal** (mord.LogisticIT) confirma la estructura ordinal del target: bajo < medio < alto aporta señal real sin necesitar `class_weight`.

### Análisis de errores (SVM OOF)

- **64/182 mal clasificados** — accuracy OOF: 64.8%
- `bajo`: F1=0.75 — bien clasificada (señal digital baja es discriminativa)
- `medio`: F1=0.55 — la más difícil; errores simétricos ~12→bajo, ~15→alto
- `alto`: F1=0.62 — 16 artistas clasificados como medio
- Errores de 2 saltos (alto→bajo, bajo→alto): solo 7 casos — el modelo respeta la ordinalidad implícitamente

### Feature importance (RF y XGBoost — consenso)

1. `sl_num_conciertos` — #1 en ambos modelos (⚠️ 41% NaN imputados como 0: posible señal espuria)
2. `lfm_oyentes` — señal más limpia, Spearman r=+0.63
3. `yt_vistas_totales` / `yt_suscriptores`
4. `sl_num_paises` / `lfm_scrobbles`

Las features de tendencias (Google Trends, YouTube reciente) tienen importancia secundaria — complementan pero no dominan frente a métricas acumuladas históricas.

---

## Ajuste de hiperparámetros (`notebooks/03_hiperparametros.ipynb`)

Optimización de XGBoost con `RandomizedSearchCV`: 100 combinaciones aleatorias × CV5 estratificado = 500 fits. Métrica de optimización: F1 macro.

### Resultado

| Modelo | Accuracy (CV5) | F1 macro (CV5) | Estabilidad |
|--------|---------------|----------------|-------------|
| XGBoost base | 63.2% ± 3.9% | 60.3% ± 3.7% | ✅ estable |
| **XGBoost tuned** | **68.2% ± 5.0%** | **66.6% ± 6.3%** | ✅ estable |
| Delta | **+5.0 puntos** | **+6.3 puntos** | — |

El delta de +6.3% en F1 macro supera el umbral de ±3 puntos a partir del cual la diferencia es interpretable con este tamaño de dataset. El XGBoost tuneado (66.6% F1) supera también al SVM base (62.3% F1), consolidándose como el mejor modelo.

### Hiperparámetros óptimos

| Parámetro | Base | Tuned | Efecto |
|-----------|------|-------|--------|
| `learning_rate` | 0.05 | **0.01** | Aprende más despacio → generaliza mejor |
| `min_child_weight` | 1 | **10** | Exige 10 muestras mínimo por hoja → evita splits espurios |
| `reg_alpha` (L1) | 0 | **1.0** | Penalización L1 fuerte → pesos sparse |
| `reg_lambda` (L2) | 1 | **2.0** | Penalización L2 fuerte → pesos más pequeños |
| `gamma` | 0 | **0.5** | Exige ganancia mínima para hacer un split |
| `n_estimators` | 300 | **400** | Más árboles para compensar LR más bajo |
| `subsample` | 0.8 | **0.9** | Más filas por árbol → aprovecha mejor n=182 |
| `colsample_bytree` | 0.8 | **0.5** | Menos features por árbol → más diversidad |
| `max_depth` | 4 | **2** | Árboles más superficiales → menos overfitting |

El patrón es claro: todos los cambios apuntan a **más regularización**. Con n=182 el principal riesgo sigue siendo el overfitting, y la búsqueda lo detectó automáticamente.

---

## Modelo serializado (`src/models/train.py` · `src/models/predict.py`)

### Entrenamiento — `train.py`

Entrena el modelo tuned sobre todos los datos y lo serializa en `models/`.

```bash
python -m src.models.train
```

**Salida** (14 mayo 2026, `acc=0.682 ± 0.050 | f1_macro=0.666 ± 0.063`):

- `models/xgb_tuned.joblib` — modelo listo para inferencia
- `models/metadata.json` — 32 features esperadas, codificación del target, parámetros y métricas CV

El `metadata.json` es el contrato entre el modelo y el dashboard: garantiza que la inferencia usa exactamente las mismas features que el entrenamiento.

### Inferencia — `predict.py`

Módulo de inferencia que expone `predecir()` para el dashboard y el agente IA.
El usuario solo proporciona **8 campos accesibles**; el resto se calcula o imputa automáticamente.

```bash
python -m src.models.predict   # prueba rápida con artista ficticio
```

```python
from src.models.predict import predecir

resultado = predecir(
    sp_num_albums=1, sp_num_singles=6, sp_anos_activo=2,
    lfm_oyentes=9_000, lfm_scrobbles=300_000,
    yt_suscriptores=6_000, yt_vistas_totales=800_000,
    sl_num_conciertos=0,
    sl_tiene_datos=0,   # 0 = sin perfil en setlist.fm
)
# → {"nivel": "bajo", "probabilidades": {"bajo": 0.893, "medio": 0.069, "alto": 0.038}, "features": {...}}
```

**Campos que pide al usuario:**

| Campo | Feature(s) del modelo |
|-------|----------------------|
| Nº álbumes / singles | `sp_num_albums`, `sp_num_singles` |
| Años activo en Spotify | `sp_anos_activo` |
| Oyentes Last.fm | `lfm_oyentes`, `lfm_oyentes_log` |
| Scrobbles Last.fm | `lfm_scrobbles`, `lfm_scrobbles_log`, `lfm_scrobbles_por_oyente` |
| Suscriptores YouTube | `yt_suscriptores`, `yt_suscriptores_log` |
| Vistas totales YouTube | `yt_vistas_totales`, `yt_vistas_log`, `yt_vistas_por_suscriptor` |
| Nº conciertos + aparece en setlist.fm | `sl_num_conciertos`, `sl_num_paises`, `sl_pct_espana`, `sl_tiene_datos` |

**Campos calculados automáticamente:** `sp_releases_por_ano`, `sp_ratio_albums_singles`, ratios y logs derivados.

**Campos imputados con valores neutros:** `sp_avg_duration_ms` (210.000 ms), `sp_pct_explicit` (0.6), `sp_pct_colabs` (0.3), tendencias → 0, `sl_tiene_datos` → 0 si no se indica (sobrescrito automáticamente a 1 si `sl_num_conciertos > 0`).

El campo de texto libre "info del artista" (salas, ciudades, etc.) no entra en el modelo — lo usa el agente LangChain para contextualizar la explicación.

### Explicabilidad SHAP (`src/models/shap_explainer.py`)

El módulo SHAP añade dos niveles de explicabilidad al modelo:

**Global** (`calcular_shap_global()`): genera dos plots sobre el dataset de entrenamiento completo y los guarda en `reports/figures/`:
- `shap_summary_bar.png` — bar plot de la media `|SHAP|` por feature (todas las clases). Más robusto que la importancia MDI de RF ante features correlacionadas.
- `shap_beeswarm_alto.png` — beeswarm para la clase ALTO: cada punto es un artista, el eje X muestra cuánto empuja esa feature la predicción hacia alto/bajo.

```bash
python -m scripts.generar_shap   # requiere data/processed/artist_features.csv + models/xgb_tuned.joblib
```

**Individual** (`shap_waterfall_fig()`): genera un waterfall plot para cada predicción en tiempo real desde el dashboard. Muestra las features que más contribuyen al resultado específico de ese artista, respecto al valor base del modelo.

Ambos plots se renderizan automáticamente en el dashboard:
- Página **📊 Resultados** → pestaña *SHAP (Global)*
- Página **🎤 Predicción** → sección *¿Por qué esta predicción?* (tras el resultado)

---

## Dashboard Streamlit (`app.py` · `pages/`)

### Arranque

```bash
# Activar entorno virtual primero
.venv\Scripts\Activate.ps1   # Windows PowerShell

streamlit run app.py
```

El dashboard necesita dos credenciales en `.env`:

```
GROQ_API_KEY=tu_clave_groq        # para el agente IA (página 3)
# Las claves de Spotify/setlist.fm solo son necesarias para recolectar datos nuevos
```

### Páginas

| Página | Archivo | Descripción |
|--------|---------|-------------|
| 🏠 Landing | `app.py` | Métricas del proyecto, descripción del tier system, navegación |
| 📊 Resultados | `pages/1_Resultados.py` | Benchmark, XGBoost base vs tuned, confusión, feature importance, **SHAP global** |
| 🎤 Predicción | `pages/2_Prediccion.py` | Formulario de 9 campos, traza del pipeline, resultado con probabilidades, **alertas de inconsistencias**, variables con nombres amigables agrupadas por fuente, **waterfall SHAP** |
| 🤖 Análisis IA | `pages/3_Analisis_IA.py` | Explicación estructurada en 3 secciones + chat de seguimiento con historial |

### Flujo de uso

1. **🎤 Predicción** — rellena los 8 campos del artista y pulsa *Predecir tier de sala*. El resultado se guarda en `st.session_state`.
2. **🤖 Análisis IA** — el agente genera automáticamente una explicación y permite hacer preguntas de seguimiento.
3. **🔄 Nueva predicción** — botón en la barra lateral (o en la página) que limpia el estado y permite analizar otro artista sin recargar la app.

### Traza en tiempo real

Cada operación muestra un panel `st.status()` expandido con todos los pasos: inputs recibidos → features calculadas → modelo cargado → probabilidades → respuesta del LLM. Los inputs se muestran con nombres legibles (no técnicos). El log de sesión (barra lateral) acumula todos los eventos con timestamp e icono de nivel (`ℹ️ OK ✅ API 🌐 ML 🤖 DATA 📊`).

### Alertas e inconsistencias (`src/utils/feature_labels.py`)

Tras cada predicción, el dashboard detecta y muestra automáticamente avisos cuando:

- **Baja confianza** (<45%): el modelo duda entre dos tiers — se informa del segundo más probable
- **Presencia digital superior al tier predicho**: métricas de Last.fm o YouTube en rango MEDIO/ALTO pero tier predicho BAJO (generalmente causado por ausencia de datos de conciertos)
- **Sin conciertos documentados + alta presencia digital**: los conciertos son la feature más determinante del modelo; se sugiere añadir datos de setlist.fm
- **`yt_num_videos` no informado con vistas altas**: la métrica "vistas por vídeo" queda a 0, enmascarando la relevancia real del canal
- **Ritmo de lanzamientos anómalo** (>20/año): puede indicar datos inconsistentes

El mismo módulo centraliza los nombres amigables de las 32 features y su agrupación por fuente (Spotify / Last.fm / YouTube / Conciertos / Tendencias) para el expander de variables.

### Agente IA (`src/agents/explicador.py`)

El system prompt incluye:
- Tabla de valores típicos por tier (rangos de oyentes, vistas, conciertos) extraída del dataset
- Artistas de referencia por tier (Tarchi · BEJO/La Zowi · Bad Gyal/Quevedo/Yung Beef)
- Interpretación del ratio scrobbles/oyente como indicador de fidelidad del fan
- Detección automática de casos frontera (confianza < 55%)

La explicación inicial fuerza **3 secciones fijas**: ¿por qué ese tier? · confianza de la predicción · qué necesita mejorar. El chat de seguimiento usa temperatura 0.4 (más conversacional).

### Compartir datos entre páginas

```python
# Página 2 guarda:
st.session_state["prediccion"] = {
    "resultado":       {"nivel": "bajo", "probabilidades": {...}, "features": {...}},
    "nombre":          "Nombre del artista",
    "info_conciertos": "Texto libre adicional",
}

# Página 3 lo lee y genera la explicación una sola vez por predicción
# (se detecta si cambió con id(resultado))
```

---

## Tests (`tests/`)

Suite completa de tests unitarios. **114 tests, 114 passed** en ~10 segundos. Todos los tests son independientes del modelo serializado y los datos reales (gitignoreados): usan DataFrames en memoria y mocks de XGBClassifier.

```bash
pytest tests/          # ejecutar la suite completa
pytest tests/ -v       # con detalle por test
```

### Cobertura

| Fichero | Tests | Qué se prueba |
|---------|-------|---------------|
| `test_build_features.py` | 39 | Cada función `_features_*()`: columnas de salida, log1p, ratios, edge cases (setlist vacío, 0 vídeos, géneros vacíos) |
| `test_predict.py` | 33 | `construir_features()`: los 32 campos, cálculos matemáticos, robustez ante ceros/negativos/valores extremos. `predecir()`: estructura del output, probabilidades suman 1.0, nivel = argmax |
| `test_preprocess.py` | 20 | `_imputar()`: estrategias por grupo (NaN→0 en conteos, mediana en ratios, mediana-por-nivel en YouTube/gtrends). `cargar_datos()`: features excluidas por modo, escalado lineal, sin NaN tras imputación |
| `test_shap_explainer.py` | 7 | Constantes `LABEL_MAP`, `shap_waterfall_fig()`: tipo de retorno `(Figure, str)`, clase válida, clase = argmax de probabilidades |

### Fixtures (`tests/conftest.py`)

- DataFrames mínimos en memoria para cada fuente de datos (Spotify, Last.fm, YouTube, setlist.fm, Tendencias)
- `df_artist_features_minimal`: 20 artistas con NaN realistas (~40% en setlist.fm, ~20% en tendencias) para tests de preprocesado
- `mock_model` + `mock_metadata`: XGBClassifier simulado con `predict_proba` → `[0.2, 0.6, 0.2]` (clase predicha: `medio`)

---

## Agente IA (`src/agents/explicador.py`)

Integración de **LangChain + Groq** para explicar las predicciones y responder preguntas sobre ellas.

### Modelo

- Proveedor: **Groq** (inferencia ultra-rápida sobre hardware dedicado)
- Modelo: `llama-3.3-70b-versatile`
- Temperatura: `0.3` (respuestas consistentes y factuales)

### API pública

```python
from src.agents.explicador import generar_explicacion, chat, _system_prompt

# Explicación inicial (llamada única)
explicacion = generar_explicacion(resultado, nombre, info_adicional)

# Chat de seguimiento (mantiene historial)
respuesta = chat(
    historial=historial_previo,
    pregunta="¿Qué debería mejorar para llegar a nivel medio?",
    resultado=resultado,
    nombre=nombre,
    info_adicional=info,
)
```

### System prompt

`_system_prompt()` construye el contexto enviado al LLM con:

- Descripción del sistema de tiers (bajo/medio/alto con aforos)
- Métricas del modelo (182 artistas, 68.2% accuracy, 66.6% F1)
- Valores concretos del artista: Spotify, Last.fm, YouTube, conciertos
- Tier predicho con probabilidades
- Info adicional de texto libre (si se proporcionó)

El prompt completo es visible en la página **🤖 Análisis IA** mediante un expander colapsable.

### Dependencias

```
langchain-core>=0.3.0
langchain-groq>=0.2.0
```

`langchain-core` se instala automáticamente como dependencia de `langchain-groq`. No se necesita el paquete `langchain` completo.

---

## Decisiones y cambios de rumbo

### Ticketmaster — descartado

Cobertura insuficiente (~50% de artistas) y sin historial de eventos pasados. CSVs y colector eliminados.

### Songkick — descartado

Cobertura peor que Ticketmaster para artistas underground españoles. Scraping frágil y setlist.fm cubre mejor los mismos datos con API oficial.

### Spotify `popularity`, `followers`, `audio_features` — bloqueados desde noviembre 2024

Spotify bloqueó estos endpoints en noviembre 2024, incluso con OAuth. `popularity` habría sido la feature más discriminativa. El colector `spotify_artistas.py` fue eliminado — devolvía 403 en todos los casos.

---

## Problemas conocidos

### Matching de artistas

`scripts/00_resolver_identidades.py` usa **fuzzy matching** con umbrales altos (80% para Spotify/Last.fm/setlist.fm, 65% para YouTube). Los artistas que no superan el umbral se escriben en `pendientes_revision.csv` para revisión manual. Una vez resueltos, se añaden al registry con `manual_override=True`.

Los colectores de datos no hacen búsquedas: usan los IDs del registry directamente, eliminando los falsos positivos por completo en ejecuciones posteriores.

### Imputación de setlist.fm — artefacto detectado y corregido

El 41% de artistas no aparece en setlist.fm. El script `scripts/validar_setlistfm.py` confirmó que este NaN **no es aleatorio**: el 79% de artistas "bajo" no tiene datos en setlist.fm frente al solo 5.3% de artistas "alto" (test chi-cuadrado: χ²=63.3, p≈0.000). Al imputar NaN→0, el modelo aprendía implícitamente que `sl_num_conciertos=0` es señal de nivel bajo — un artefacto.

Al mismo tiempo, las features de setlist.fm tienen señal real: el modelo pierde **+6.4 puntos de F1 macro** sin ellas (0.669 → 0.605), por lo que eliminarlas tampoco es la solución.

**Corrección aplicada**: se añadió `sl_tiene_datos` (0/1), un flag binario creado en `preprocess.py` antes de cualquier imputación, que hace explícita la distinción entre "sin perfil en setlist.fm" y "0 conciertos documentados". El modelo puede ahora aprender la correlación de forma honesta en lugar de inferirla de un valor imputado.

### Correcciones manuales de plataformas

Las correcciones de identidad entre plataformas se gestionan en `config/artistas_registry.csv`:

- Artistas con canal incorrecto en YouTube: editar `yt_channel_id` y `yt_nombre_canal`, poner `manual_override=True`.
- Artistas sin canal de YouTube (Xico Palma, GREKKY): dejar `yt_channel_id` en blanco, `manual_override=True`, añadir nota en `notas`.
- Artistas con nombre distinto en Last.fm o setlist.fm: editar `lastfm_nombre` / `sl_nombre`, poner `manual_override=True`.

El flag `manual_override=True` garantiza que `scripts/00_resolver_identidades.py` nunca sobreescriba esas entradas.

### Validación de la imputación de setlist.fm

El script `scripts/validar_setlistfm.py` analiza si la imputación a 0 del 41% de NaN en `sl_num_conciertos` introduce un artefacto en el modelo. Ejecuta tres análisis:

1. **Test chi-cuadrado** — ¿el NaN está correlacionado con el tier? Si sí, el 0 imputado puede actuar como señal proxy del nivel bajo.
2. **Distribución real** — estadísticas de conciertos solo entre artistas con datos.
3. **Comparativa CV5** — modelo completo vs. modelo sin features de setlist.fm.

```bash
python -m scripts.validar_setlistfm   # requiere data/processed/artist_features.csv
```

**Resultados sobre el dataset de 182 artistas:**

| Tier | Sin datos en setlist.fm | Total |
|------|------------------------|-------|
| Bajo | ~79% (~64/81) | — |
| Medio | ~25% (~15/62) | — |
| Alto | ~5% (~2/39) | — |

χ²=63.3, df=2, **p≈0.000** → el NaN está fuertemente correlacionado con el tier.
Δ F1 macro sin setlist.fm: **−6.4 puntos** (0.669→0.605) → las features tienen señal real.

**Conclusión y acción tomada**: se añadió la feature `sl_tiene_datos` (flag binario 0/1) creada antes de la imputación, lo que permite al modelo aprender la correlación de forma explícita y honesta. Ver sección *Imputación de setlist.fm* en Problemas conocidos.
