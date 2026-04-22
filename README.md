# Predicción Musical TFM

## Descripción del proyecto

Proyecto de fin de máster para predecir el **tier de sala** de artistas de rap/urbano español — es decir, el tamaño máximo de recinto que un artista puede llenar en España. El dataset cubre 125 artistas de la escena urbana española, desde artistas completamente underground hasta nombres con presencia en festivales y palacios de deportes.

El pipeline recoge datos de 6 fuentes (Spotify, Last.fm, YouTube, setlist.fm, Google Trends + YouTube reciente) mediante sus APIs, aplica ingeniería de características, y entrena modelos de clasificación multiclase. Los resultados se analizan en notebooks Jupyter.

El foco está en artistas **poco conocidos o emergentes**, lo que implica datos escasos o inexistentes en las APIs estándar y requiere validación y corrección manual.

---

## Estructura del proyecto

```
src/
  data_collectors/   # Clientes de API para ingesta de datos raw
  features/          # Ingeniería de características y preprocesado
  models/            # Entrenamiento, evaluación y serialización de modelos
  utils/             # Helpers compartidos (logging, DB, config)
config/              # Constantes y carga de entorno
scripts/             # Scripts de entrada (generación de labels, ETL)
notebooks/           # Análisis exploratorio y modelado
  01_eda.ipynb       # EDA completo con Kruskal-Wallis y correlaciones
  02_modelado.ipynb  # Entrenamiento y comparativa de modelos
tests/               # Tests unitarios e integración
data/
  raw/               # CSVs originales de cada fuente (gitignored)
  processed/         # artist_features.csv — matriz de features unificada
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

**Distribución**: bajo=48 · medio=45 · alto=32 · **total=125 artistas**

La variable objetivo está en `artistas_labels.csv` como `nivel` (texto) y `target` (entero 1/2/3).
`nombre_buscado` es el identificador del artista — **nunca entra como variable predictora**.

---

### ¿Con qué lo predecimos? — Variables predictoras `X`

Señales de presencia digital, actividad musical y trayectoria recogidas de 5 APIs:

| Fuente | Features | ¿Qué mide? |
|--------|----------|------------|
| **Spotify** (discografía) | 6 | Madurez de carrera: años activo, volumen de lanzamientos, ritmo de publicación |
| **Spotify** (top tracks) | 3 | Estilo: duración de tracks, % explicit, % colaboraciones |
| **Last.fm** | 6 | Audiencia histórica: oyentes únicos, scrobbles totales, engagement de fans |
| **YouTube** | 7 | Presencia digital: suscriptores, vistas, engagement del canal |
| **setlist.fm** | 5 | Actividad en directo: conciertos, países, % en España |

**Total tras preprocesado** (`src/features/preprocess.py`):

| Modo | Features | Escalado | Modelos |
|------|----------|----------|---------|
| `arbol` | **26** | Sin escalar | Random Forest, XGBoost |
| `lineal` | **21** | RobustScaler | Regresión Logística, SVM |

En modo `lineal` se eliminan adicionalmente `lfm_oyentes`, `lfm_scrobbles`, `yt_suscriptores`, `yt_vistas_totales` (sustituidas por sus versiones log para evitar multicolinealidad) y `sp_pct_colabs` (r≈−0.02, solo añade ruido).

Las 4 features descartadas en ambos modos por el EDA (señal no significativa):
`sp_num_eps` · `sp_num_total_releases` · `sp_dias_desde_ultimo_release` · `yt_edad_canal_anos`

El preprocesado no genera ningún fichero — devuelve `X`, `y` en memoria listos para entrenar.
Verificado: **NaN en X = 0** en ambos modos sobre los 125 artistas.

---

## Fuentes de datos

### Spotify IDs (`src/data_collectors/spotify.py`)

Extrae el `artist_id` (Spotify URI) de cada artista a partir de `artistas.txt`. Es el identificador universal del ecosistema musical digital que usan el resto de fuentes como referencia.

- CSV de salida: `data/raw/spotify_ids.csv`
- Columnas: `nombre_buscado`, `nombre_spotify`, `artist_id`, `spotify_uri`, `spotify_url`, `match_score`

> **Nota**: Spotify eliminó `followers`, `popularity` y `genres` de su API Web en noviembre 2024, incluso con OAuth. No es un problema de código — esos campos ya no existen en ninguna respuesta de la API. Ver sección [Decisiones y cambios de rumbo](#decisiones-y-cambios-de-rumbo).

### Spotify Features (`src/data_collectors/spotify_features.py`)

Usa los IDs de `spotify_ids.csv` para extraer features de discografía y top tracks.

- `data/raw/spotify_discografia.csv` — una fila por artista: `num_albums`, `num_singles`, `num_eps`, `num_total_releases`, `primer_lanzamiento`, `ultimo_lanzamiento`, `anos_activo`, `releases_por_ano`
- `data/raw/spotify_top_tracks.csv` — una fila por canción (top 10/artista): `track_name`, `duration_ms`, `explicit`, `num_artistas`

> `popularity` y `audio_features` (`danceability`, `energy`, `valence`...) eliminados — Spotify los bloqueó en noviembre 2024.

### Last.fm (`src/data_collectors/lastfmapi.py`)

Fuente principal para **oyentes, scrobbles y géneros**, ya que Spotify no los proporciona.

- CSV de salida: `data/raw/lastfm_artistas.csv`
- Columnas: `nombre_buscado`, `nombre_lastfm`, `oyentes`, `scrobbles`, `generos`, `lastfm_url`, `mbid`, `match_score`, `artist_id`

**Cómo interpretar los datos de Last.fm:**
- `oyentes` = usuarios únicos históricos, no mensuales. Los números son menores que en Spotify porque Last.fm tiene menos usuarios.
- La correlación relativa entre artistas es válida: los artistas más populares en Spotify también lideran en Last.fm.
- `scrobbles` = reproducciones acumuladas totales. Buen indicador de **engagement del fan** — pocos oyentes con muchos scrobbles = base muy fiel.

### YouTube (`src/data_collectors/youtube.py`)

Presencia digital en YouTube: tamaño de audiencia, actividad y engagement.

- CSV de salida: `data/raw/youtube_artistas.csv`
- Columnas: `nombre_buscado`, `canal_youtube`, `suscriptores`, `vistas_totales`, `num_videos`, `fecha_creacion`

**Correcciones manuales aplicadas**: 17 artistas tenían canales incorrectos o desactualizados en el CSV original. Se corrigieron manualmente los suscriptores y se nularon las vistas de los canales erróneos (BEJO, Rels B, Morad, Dano, Cecilio G, Choclock, Yung Beef, entre otros). Dos artistas (Xico Palma, GREKKY) no tienen canal de YouTube — sus valores se dejaron a nulo.

### Tendencias (`src/data_collectors/tendencias.py`) — en desarrollo

Combina dos fuentes para medir el **buzz actual** de cada artista, con especial atención a artistas emergentes y poco conocidos donde otras señales son escasas:

- **Google Trends** (`pytrends`): interés de búsqueda semanal normalizado (0-100) para España en los últimos 12 meses. Para artistas con volumen muy bajo el valor es 0, lo que en sí mismo es información útil (= sin presencia en búsquedas web).
- **YouTube vídeos recientes** (YouTube Data API v3, misma key que `youtube.py`): vistas acumuladas de los últimos 5 vídeos del artista. Esta señal es más sensible que Google Trends para artistas pequeños, ya que un vídeo viral en un canal pequeño sí se refleja aquí.

- CSV de salida: `data/raw/tendencias.csv`
- Columnas previstas: `nombre_buscado`, `gtrends_interes_medio`, `gtrends_pico_maximo`, `yt_vistas_recientes`, `yt_videos_recientes`

#### Por qué se descartó TikTok

TikTok es hoy la principal plataforma de viralización para artistas emergentes del rap/urbano español. Sin embargo, no existe una API pública accesible para obtener datos de artistas de terceros:

- La **TikTok Research API** requiere aprobación institucional y está limitada a investigación académica formal.
- Las librerías no oficiales (`TikTokApi`) usan automatización de navegador, son inestables ante actualizaciones de TikTok y van contra los Términos de Servicio, lo que las hace inadecuadas para un proyecto reproducible.

Se optó por YouTube reciente como proxy de viralización digital, ya que los artistas que se viralizan en TikTok suelen subir simultáneamente sus vídeos a YouTube.

### setlist.fm (`src/data_collectors/setlistfm.py`)

Historial de conciertos (setlists) por artista. Fuente más directa para medir actividad en directo.

- CSV de salida: `data/raw/setlistfm_conciertos.csv`
- Columnas: `nombre`, `setlistfm_mbid`, `setlist_id`, `fecha`, `venue_id`, `venue_nombre`, `ciudad`, `pais`, `num_canciones`, `tiene_encore`, `canciones`, `url_setlist`
- Lógica incremental: omite artistas ya procesados en ejecuciones anteriores.
- Si un artista no tiene conciertos, se añade una fila con `nombre` y el resto vacío.

---

## Etiquetas del modelo (`data/raw/artistas_labels.csv`)

Fichero central con la variable objetivo. Los tiers originales (0-4) se fusionaron en 3 niveles para garantizar suficientes ejemplos por clase:

| `tier_sala` | `nivel` | Aforo equivalente | Descripción |
|-------------|---------|-------------------|-------------|
| 1 | `bajo` | < 200 personas | Sala Víbora · artistas sin sala propia |
| 2 | `medio` | 200 – 2.000 personas | Planta Baja, Copera (Granada) · El Sol, Razzmatazz, La Riviera |
| 3 | `alto` | > 2.000 personas | Palacio de Deportes · WiZink, Movistar Arena, estadios, festivales |

### Alcance geográfico de los tiers

Los tiers son **nacionales**, no locales. El criterio es la capacidad del recinto, independientemente de la ciudad:

- Un artista `medio` puede tocar en Planta Baja (Granada), Razzmatazz (Barcelona) o La Riviera (Madrid) — todos están en el mismo rango de aforo.
- La escena de Granada está sobrerepresentada en el dataset porque el etiquetado manual se hizo con conocimiento directo de esa escena, pero los tiers son válidos a nivel nacional.

No es necesario ampliar la lista de salas de referencia para el MVP — los tres tiers por capacidad ya cubren el espectro completo de la industria musical española.

### Proceso de etiquetado

El etiquetado es **semisupervisado**: aproximadamente la mitad de las etiquetas se infirieron automáticamente desde el historial de venues de setlist.fm (`scripts/generar_labels.py`), y la otra mitad se asignaron **manualmente** basándose en conocimiento directo de la escena musical española (sold-outs, actividad reciente, historial en Granada no capturado por las APIs).

Los casos inferidos automáticamente fueron revisados y corregidos donde el venue no era representativo (venues en el extranjero, salas pequeñas dentro de complejos grandes, falsos positivos).

### Columnas

- `nombre_buscado` — identificador del artista
- `tier_inferido` — tier calculado automáticamente desde setlist.fm
- `venue_mayor` — venue que determinó el tier inferido (solo referencia)
- `tier_sala` — **etiqueta definitiva usada como target**
- `nivel` — texto: bajo / medio / alto
- `notas` — justificación cuando fue corregido manualmente

---

## Feature Engineering (`src/features/build_features.py`)

Combina todas las fuentes raw en la matriz `data/processed/artist_features.csv`.
**125 artistas × 30 features** (+ 4 columnas de metadata/target).

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

### Imputación de NaN

| Columnas | NaN aprox. | Causa | Imputación |
|----------|------------|-------|------------|
| `sl_num_conciertos`, `sl_num_paises` | 44% | Sin registros en setlist.fm | **0** (sin conciertos = 0 conciertos) |
| `sl_avg_canciones`, `sl_pct_encore`, `sl_pct_espana` | 44% | Sin registros en setlist.fm | **Mediana global** (0 implicaría setlist vacío) |
| `yt_vistas_totales`, `yt_vistas_log`, `yt_num_videos`, `yt_vistas_por_*` | 14% | Canales corregidos sin datos de vistas | **Mediana por nivel** (evita mezclar artistas de distinto tier) |
| `sp_anos_activo`, `sp_releases_por_ano` | 3% | Sin lanzamientos en Spotify | **0** |
| `yt_suscriptores` | 2% | Sin canal YouTube | **0** |
| `lfm_*` | < 1% | No encontrado en Last.fm | **0** |

---

## Análisis exploratorio (`notebooks/01_eda.ipynb`)

El EDA analiza la distribución de las 30 features y su relación con el target usando:

- **Kruskal-Wallis** (H): test no paramétrico para medir si una feature discrimina significativamente entre niveles.
- **Correlación de Spearman** (r): fuerza y dirección de la relación feature-target.

**Principales hallazgos:**

| Feature | H (Kruskal-Wallis) | r (Spearman) | Conclusión |
|---------|-------------------|--------------|------------|
| `lfm_oyentes_log` | ~100 | +0.75 | Feature más discriminativa del dataset |
| `yt_suscriptores_log` | ~85 | +0.70 | Muy alta señal |
| `sl_num_conciertos` | ~60 | +0.60 | Directo mucho más relevante para artistas alto |
| `sp_num_albums` | ~30 | +0.35 | Señal moderada — carrera consolidada |
| `sp_num_eps` | ~0 | ~0 | Sin varianza útil — eliminada |
| `yt_edad_canal_anos` | ~1 | +0.10 | No significativa — eliminada |

Los **outliers extremos** (Quevedo ~1.8B vistas, Bad Gyal ~1.5B) se mantienen como datos válidos — son artistas `alto` reales. Se usa RobustScaler en modo lineal para mitigar su efecto en los coeficientes.

---

## Modelado (`notebooks/02_modelado.ipynb`)

### Estrategia de validación

Se usa **StratifiedKFold k=5** en lugar de train/test split. Con n=125 artistas un hold-out del 20% dejaría solo ~25 artistas de test — demasiado poco para métricas fiables. Con CV k=5 cada artista aparece exactamente **una vez en test**, evaluado por un modelo que no lo ha visto en entrenamiento (sin data leakage). Las métricas reportadas son la media ± desviación estándar de los 5 folds.

> **Caveat**: con n=125 la desviación estándar es alta (~±5-12 puntos). Diferencias de 2-3 puntos entre modelos no son estadísticamente significativas.

### Baselines

Los baselines son el suelo mínimo que cualquier modelo real debe superar. Se usan dos `DummyClassifier` que no aprenden nada:

| Baseline | Estrategia | Accuracy |
|----------|-----------|----------|
| `most_frequent` | Siempre predice `bajo` (clase mayoritaria, 48/125) | 38.4% |
| `stratified` | Predice aleatoriamente respetando la distribución de clases | ~25.6% |

El umbral relevante es el 38.4%: si un modelo no supera "predecir siempre bajo", es inútil.

### Modelos y resultados

| Modelo | Features | Escalado | `class_weight` | Accuracy (CV5) | F1 macro (CV5) |
|--------|----------|----------|----------------|---------------|----------------|
| Dummy (most_frequent) | — | — | — | 0.384 ± 0.020 | 0.185 ± 0.007 |
| Dummy (stratified) | — | — | — | 0.256 ± 0.093 | 0.221 ± 0.082 |
| Regresión Logística | 21 (lineal) | RobustScaler | `balanced` | 0.728 ± 0.059 | 0.719 ± 0.053 |
| SVM (kernel RBF) | 21 (lineal) | RobustScaler | `balanced` | 0.712 ± 0.059 | 0.698 ± 0.059 |
| Random Forest | 26 (arbol) | No | `balanced_subsample` | 0.736 ± 0.120 | 0.733 ± 0.124 |
| **XGBoost** | **26 (arbol)** | **No** | **—** | **0.792 ± 0.089** | **0.793 ± 0.087** |

**XGBoost es el mejor modelo** (+41 puntos sobre el baseline más fuerte). Todos los modelos reales superan claramente los baselines, confirmando que las features tienen poder predictivo real sobre el tier de sala.

Los modelos lineales (LR, SVM) son sorprendentemente competitivos usando solo 21 features escaladas, lo que sugiere que la relación entre las features log-transformadas y el target es aproximadamente lineal. Random Forest tiene más varianza (±0.120) que XGBoost (±0.089) — los árboles individuales son más sensibles a la partición del fold con este tamaño de dataset.

El notebook también incluye: matriz de confusión del mejor modelo (predicciones out-of-fold), feature importance (RF por MDI y XGBoost por gain), y análisis de los 26 artistas mal clasificados — casi todos en fronteras adyacentes (bajo↔medio o medio↔alto).

---

## Decisiones y cambios de rumbo

Durante el proyecto se exploraron varias fuentes y enfoques que finalmente se descartaron:

### Ticketmaster — descartado

Se integró la Discovery API v2 de Ticketmaster (`TICKETMASTER_API_KEY`), que devuelve artistas y eventos en España. Se descartó por dos motivos fundamentales:

1. **Cobertura insuficiente**: solo cubre ~50% de los artistas del dataset. Los artistas underground y emergentes (precisamente el foco del proyecto) directamente no aparecen.
2. **Sin historial**: Ticketmaster solo devuelve eventos próximos o muy recientes — no permite reconstruir la trayectoria pasada de un artista, que es lo que necesitamos para predecir su tier.

Los CSVs (`ticketmaster_artistas.csv`, `ticketmaster_eventos.csv`) y el colector (`src/data_collectors/ticketmaster.py`) fueron eliminados del proyecto.

### Songkick — descartado

Se exploró el scraping de Songkick (`scrapingsongkick.py`) para obtener datos de conciertos en España con información adicional (sold-out, precios, fans asistentes). Se descartó porque:

1. **Cobertura aún peor que Ticketmaster** para artistas underground de la escena española.
2. **Scraping frágil**: dependía de la estructura HTML del sitio, que cambia frecuentemente.
3. **setlist.fm cubre mejor los mismos datos** para este tipo de artistas, con API oficial y sin riesgo legal.

### Spotify `popularity`, `followers`, `audio_features` — bloqueados desde noviembre 2024

Se intentó extraer estas features como señales muy predictivas del tier:
- `popularity` (score 0-100) — habría sido la feature más discriminativa del modelo.
- `followers` — número de seguidores del artista en Spotify.
- `audio_features` (`danceability`, `energy`, `valence`, `tempo`...) — características musicales del track.

Spotify bloqueó los tres endpoints en noviembre 2024, incluso con OAuth y client credentials. Se creó un colector adicional (`src/data_collectors/spotify_artistas.py`) intentando acceder al endpoint directo de artista, pero también devuelve 403. El fichero permanece en el proyecto como documentación del intento pero no se usa en el pipeline.

### Salas de Granada (`granadavenues.csv`) — relegada a referencia

Se construyó un fichero de salas de Granada con capacidades para mapear venues de setlist.fm a tier. Finalmente el etiquetado se hizo directamente sobre `artistas_labels.csv` mediante keywords de venues conocidos a nivel nacional (no solo Granada), por lo que el fichero quedó como referencia de aforos pero no entra en el pipeline de features.

---

## Problemas conocidos

### Matching de artistas

Los scripts usan **fuzzy matching** (`difflib.SequenceMatcher`) con umbral 60% para evitar coger artistas incorrectos. Artistas con nombres cortos o genéricos son propensos a falsos positivos y se revisaron manualmente. El único caso sin datos en Last.fm es **MARCE**, que no tiene perfil en la plataforma — sus campos quedan vacíos y se imputan a 0 en el preprocesado.