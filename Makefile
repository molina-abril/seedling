.PHONY: help check-env bootstrap setup ingest cluster analyze-clusters hierarchy retrieve freeze pipeline test

# ---------------------------------------------------------------------------
# Entorno estandar del proyecto: conda 'lit-review'.
#
# Se invoca por RUTA ABSOLUTA a proposito: en esta maquina los shims de pyenv
# van antes en el PATH y ensombrecen el `python` de conda, asi que ni
# `conda activate` ni `conda run` son fiables. Usar el binario del env
# directamente es la unica forma robusta.
# Si tu instalacion de conda/miniforge esta en otra ruta, sobreescribe:
#   make test CONDA_HOME=/ruta/a/miniforge3
# ---------------------------------------------------------------------------
CONDA_ENV  ?= lit-review
CONDA_HOME ?= $(HOME)/miniforge3
ENV_BIN    := $(CONDA_HOME)/envs/$(CONDA_ENV)/bin
PYTHON     := $(ENV_BIN)/python
PYTEST     := $(ENV_BIN)/pytest
PIP        := $(ENV_BIN)/pip

# Pasa flags extra a cualquier target con ARGS, por ejemplo:
#   make ingest   ARGS="--max-pdfs 10 --no-scopus"
#   make retrieve ARGS="--cluster-id -1 --max-iterations 8"
ARGS ?=

help:
	@echo "==================================================================="
	@echo " Literature Review Pipeline - targets"
	@echo " Entorno: conda '$(CONDA_ENV)'  ($(ENV_BIN))"
	@echo "==================================================================="
	@echo "  make setup            - Bootstrap del entorno (CMake/LLVM + requirements.txt)"
	@echo "  make bootstrap        - Solo instala requirements.txt en el env conda"
	@echo "  make ingest           - Ingesta de PDFs -> data/processed/papers.json"
	@echo "  make cluster          - Fase 6: clustering BERTopic"
	@echo "  make analyze-clusters - Fase 6.5: caracterizacion de clusters"
	@echo "  make hierarchy        - Exporta txt/hierarchy.txt (BERTopic) + scipy artefacts en results/clustering/"
	@echo "  make retrieve         - Fase 7: retrieval inteligente"
	@echo "  make freeze           - Congela un run de retrieve como dataset final (RUN=<ts> o el mas reciente)"
	@echo "  make pipeline         - ingest -> cluster -> hierarchy -> analyze-clusters -> retrieve"
	@echo "                          (ARGS se aplica al paso 'cluster'; usa ARGS=\"--hitl\" para pausa humana)"
	@echo "  make test             - Ejecuta pytest"
	@echo ""
	@echo " Flags extra: usa ARGS=\"...\"   (ej: make ingest ARGS=\"--max-pdfs 10\")"
	@echo ""
	@echo "--- ingest ---------------------------------------------------------"
	@echo "  --max-pdfs N           Maximo de PDFs a procesar             (def: todos)"
	@echo "  --no-scopus            Omite el enriquecimiento via Scopus"
	@echo "  --no-arxiv             Omite la busqueda en ArXiv"
	@echo "  --output PATH          Fichero de salida    (def: data/processed/papers.json)"
	@echo "  --skip-title-fixer     Desactiva la correccion de titulos/DOI"
	@echo "  --title-fixer-pages N  Paginas de PDF para el title fixer    (def: 2)"
	@echo ""
	@echo "--- cluster --------------------------------------------------------"
	@echo "  --papers-file PATH     Papers a clusterizar     (def: data/processed/papers.json)"
	@echo "  --output PATH          Clusters de salida    (def: results/clustering/clusters.json)"
	@echo "  --metrics-output PATH  Metricas    (def: results/clustering/clustering_metrics.json)"
	@echo "  --save-model           Guarda el modelo BERTopic entrenado   (def: activado)"
	@echo "  --model-dir PATH       Directorio del modelo      (def: models/clustering)"
	@echo ""
	@echo "--- analyze-clusters -----------------------------------------------"
	@echo "  --clusters-file PATH    Clusters de entrada (def: results/clustering/clusters.json)"
	@echo "  --enriched-papers PATH  Papers procesados   (def: data/processed/papers.json)"
	@echo "  --output PATH           Briefs (def: results/cluster_analysis/clusters_enriched.json)"
	@echo "  --model NAME            Modelo OpenAI                        (def: gpt-5.1)"
	@echo "  --hierarchy-file PATH   Arbol BERTopic legible              (def: txt/hierarchy.txt, 'none' para omitir)"
	@echo "  --cluster-id N          Analiza un cluster concreto    (-1 = todos, def: -1)"
	@echo ""
	@echo "--- hierarchy ------------------------------------------------------"
	@echo "  MODEL=PATH                Modelo BERTopic .pkl (def: el mas reciente en models/clustering/)"
	@echo "  ARGS=\"--linkage ward\"     Flags extra para scripts/extract_hierarchy.py"
	@echo "                            (--papers-file, --output-dir, --tree-output, --linkage, --visualize)"
	@echo "                            txt/hierarchy.txt es la salida principal (la que consume analyze-clusters)."
	@echo ""
	@echo "--- retrieve -------------------------------------------------------"
	@echo "  --clusters-file PATH      Clusters de entrada (def: results/clustering/clusters.json)"
	@echo "  --papers-dir PATH         Directorio de PDFs semilla           (def: papers)"
	@echo "  --cluster-id N            Cluster a procesar     (-1 = todos, def: -1)"
	@echo "  --output-dir PATH         Directorio de salida                 (def: results)"
	@echo "  --max-results N           Maximo de resultados por estrategia  (def: 30)"
	@echo "  --sampling-per-year N     Top-citados por anyo (~25=1 pagina Scopus)  (def: config)"
	@echo "  --sampling-window-years N Anyos recientes a muestrear              (def: config)"
	@echo "  --core-axes-min-hits N    Min hits Scopus para usar AND-of-axes (si no, OR) (def: 2000)"
	@echo "                            (-1 = todos los clusters; salta el ruido -1 automaticamente)"
	@echo "  --enriched-papers PATH    Papers procesados   (def: data/processed/papers.json)"
	@echo "  --max-iterations N        Iteraciones max del bucle de refinamiento (def: 5)"
	@echo "  --min-recall F            Umbral de recall para parar          (def: 0.90)"
	@echo "  --min-precision F         Suelo de precision estimada          (def: 0.25)"
	@echo "  --target-precision F      Precision objetivo para parar        (def: 0.60)"
	@echo "  --enriched-clusters PATH  Briefs de analyze-clusters (def: results/cluster_analysis/clusters_enriched.json)"
	@echo "  --scoring-config PATH     YAML de pesos del scorer (def: configs/retrieval/relevance.yaml)"
	@echo "  --bertopic-model PATH     Modelo .pkl para verificar membership (def: el mas reciente; 'none' = omitir)"

# Verifica que el entorno conda existe antes de cualquier target que lo use.
check-env:
	@test -x "$(PYTHON)" || { \
	  echo "ERROR: no encuentro el entorno conda '$(CONDA_ENV)' en:"; \
	  echo "       $(ENV_BIN)"; \
	  echo "  Crealo con:  conda create -n $(CONDA_ENV) python=3.12 -y"; \
	  echo "  e instala:   make setup"; \
	  echo "  (o sobreescribe la ruta: make <target> CONDA_HOME=/ruta/a/miniforge3)"; \
	  exit 1; }

bootstrap: check-env
	$(PIP) install -r requirements.txt

setup:
	@PYTHON="$(PYTHON)" bash scripts/bootstrap.sh

ingest: check-env
	$(PYTHON) -m src.cli ingest $(ARGS)

cluster: check-env
	$(PYTHON) -m src.cli cluster $(ARGS)

analyze-clusters: check-env
	$(PYTHON) -m src.cli analyze-clusters $(ARGS)

# Permite sobreescribir el modelo:  make hierarchy MODEL=models/clustering/bertopic_model_XYZ.pkl
MODEL ?=
hierarchy: check-env
	@MODEL_PATH="$(MODEL)"; \
	if [ -z "$$MODEL_PATH" ]; then \
	  MODEL_PATH=$$(ls -1 models/clustering/bertopic_model_*.pkl 2>/dev/null | sort | tail -n 1); \
	fi; \
	if [ -z "$$MODEL_PATH" ] || [ ! -f "$$MODEL_PATH" ]; then \
	  echo "ERROR: no se encontro ningun bertopic_model_*.pkl en models/clustering/."; \
	  echo "  Ejecuta 'make cluster' primero o pasa MODEL=ruta/al/modelo.pkl"; \
	  exit 1; \
	fi; \
	echo "Usando modelo: $$MODEL_PATH"; \
	$(PYTHON) scripts/extract_hierarchy.py --model "$$MODEL_PATH" $(ARGS)

retrieve: check-env
	$(PYTHON) -m src.cli retrieve $(ARGS)

# Congela un run de retrieve validado como dataset definitivo en
# results/retrieval/final/ (+ MANIFEST.json). Reproducibilidad total: los
# ficheros no cambian salvo que vuelvas a congelar.
#   make freeze                      # congela el run mas reciente
#   make freeze RUN=20260522_005947  # congela un run concreto
RUN ?=
freeze: check-env
	$(PYTHON) scripts/freeze_run.py $(if $(RUN),--run $(RUN),)

# Pipeline completo: ingest -> cluster -> hierarchy -> analyze-clusters -> retrieve.
#
# ARGS se enruta SOLO al paso `cluster` (es el unico paso con flags relevantes
# al flujo HITL). El resto de pasos usan sus defaults. Para correr el pipeline
# con pausa interactiva de revision humana:
#
#     make pipeline ARGS="--hitl"
#
# Para reproducir un review previo sin pausa:
#
#     make pipeline ARGS="--overrides results/clustering/cluster_review.yaml"
#
# Si necesitas flags en otros pasos, ejecutalos uno a uno con su propio ARGS.
pipeline: check-env
	$(MAKE) ingest ARGS=
	$(MAKE) cluster ARGS="$(ARGS)"
	$(MAKE) hierarchy ARGS=
	$(MAKE) analyze-clusters ARGS=
	$(MAKE) retrieve ARGS=

test: check-env
	$(PYTEST)
