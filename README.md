# Literature Review Pipeline

Repositorio para búsqueda, clustering, expansión y evaluación de
literatura científica con compatibilidad hacia atrás para los artefactos
históricos del proyecto.

------------------------------------------------------------------------

## 🧠 Requisitos del sistema

-   Python \>= 3.12\
-   Conda (Miniforge recomendado)\
-   pip (dentro del entorno Conda)
-   CMake y LLVM 20 (se instalan automáticamente en macOS con Homebrew desde scripts/bootstrap.sh)

------------------------------------------------------------------------

## ⚙️ Instalación del entorno

> **Entorno estándar del proyecto: conda `lit-review`.**
> Todo —pipeline, tests y bootstrap— debe ejecutarse en este entorno. Los
> objetivos de `make` ya lo invocan por **ruta absoluta**
> (`~/miniforge3/envs/lit-review/bin/...`) a propósito: en máquinas con
> `pyenv`, sus *shims* van antes en el `PATH` y ensombrecen el `python` de
> conda, así que `conda activate` y `conda run` **no** son fiables aquí.
> Si tu miniforge está en otra ruta:
> `make <target> CONDA_HOME=/ruta/a/miniforge3`.

### 1. Instalar Miniforge

https://conda-forge.org/download/

------------------------------------------------------------------------

### 2. Crear entorno Conda

``` bash
conda create -n lit-review python=3.12 -y
```

Activar entorno:

``` bash
conda activate lit-review
```

------------------------------------------------------------------------

### 3. Instalar dependencias

Primero, ejecuta el bootstrap del entorno para verificar CMake/LLVM 20 y preparar la instalación:

``` bash
bash scripts/bootstrap.sh
```

O, si prefieres usar Make:

``` bash
make setup
```

El bootstrap ya actualiza `pip`, `setuptools` y `wheel`, y luego instala `requirements.txt`.

------------------------------------------------------------------------

## 🚀 Ejecución del pipeline

La vía recomendada es `make`: garantiza el uso del entorno conda `lit-review`.

``` bash
make ingest            # ingestión + enriquecimiento + deduplicación
make cluster           # Fase 6: clustering BERTopic
make analyze-clusters  # Fase 6.5: caracterización de clusters
make retrieve          # Fase 7: retrieval inteligente
make pipeline          # las cuatro fases en orden
make test              # tests (pytest del entorno conda)
make help              # lista completa de targets y flags
```

Para pasar flags a un comando, usa `ARGS`:

``` bash
make ingest ARGS="--max-pdfs 10 --no-scopus"
make ingest ARGS="--skip-title-fixer"   # omitir corrección de títulos (no recomendado)
```

Equivale a invocar directamente —siempre con el python del entorno conda:

``` bash
~/miniforge3/envs/lit-review/bin/python -m src.cli ingest --max-pdfs 10
```

------------------------------------------------------------------------

## 📁 Estructura del proyecto

-   configs/: configuración del sistema y prompts
-   data/: datos crudos, intermedios y procesados
-   src/: código del pipeline
-   experiments/: ejecuciones, ablaciones y benchmarks
-   artifacts/: reportes, métricas y salidas finales
-   documentation/: especificaciones y decisiones arquitectónicas
-   tests/: pruebas unitarias e integración

------------------------------------------------------------------------

## 🔄 Flujo del pipeline

1.  Ingesta de PDFs\
2.  Corrección de metadatos (títulos / DOI)
    -   CrossRef como fuente determinista principal\
    -   OpenAI como fallback controlado\
3.  Enriquecimiento externo
    -   Scopus (si está habilitado)\
    -   ArXiv (fallback académico)\
4.  Normalización de metadatos\
5.  Deduplicación semántica\
6.  Exportación de artefactos

------------------------------------------------------------------------

## 🧪 Sistema de corrección de títulos

-   CrossRef es la fuente principal de validación de DOI/títulos\
-   OpenAI se usa únicamente como fallback cuando no hay coincidencias
    claras\
-   Todas las sugerencias se validan antes de ser aplicadas\
-   El sistema puede desactivarse con:

``` bash
--skip-title-fixer
```

------------------------------------------------------------------------

## 📦 Configuración avanzada

-   catalog_path puede pasarse al TitleFixAgent para validación externa\
-   Scopus y ArXiv pueden activarse/desactivarse desde CLI\
-   El pipeline es modular y extensible para experimentación

------------------------------------------------------------------------

## 🧭 Objetivo del proyecto

Construir un pipeline reproducible para:

-   Revisión sistemática asistida por IA\
-   Enriquecimiento automático de literatura científica\
-   Clustering semántico de papers\
-   Soporte a investigación doctoral en sistemas de decisión y
    optimización

------------------------------------------------------------------------

## 🧱 Estado del proyecto

En evolución activa: refactorización hacia arquitectura modular con
entornos Conda + Python 3.12.

Ejecutar clustering:
```bash
make cluster
```

Obtener visualizacion hierarchy (usa el python del entorno conda por ruta absoluta):
```bash
~/miniforge3/envs/lit-review/bin/python scripts/extract_hierarchy.py \
  --model models/clustering/bertopic_model_20260511_222220.pkl \
  --output-dir results/clustering \
  --linkage ward \
  --visualize
```