# 📊 Plataforma Analítica CASEN 2024

![Versión de Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

Una plataforma interactiva e inteligente construida con Python, **Pandas** y **Gradio** para analizar la base de datos de la **Encuesta CASEN 2024 de Chile**. La aplicación procesa los microdatos brutos para ofrecer métricas sociodemográficas, indicadores de pobreza e ingresos, y cuenta con un Asistente de IA con modelos intercambiables para resolver dudas sobre las regiones de Chile.

> **Nota:** Este proyecto ha sido desarrollado como portafolio para demostrar capacidades en Ingeniería de Datos (ETL con `pandas` y optimización en formato `parquet`), Data Analytics, y desarrollo de aplicaciones web interactivas con `gradio` y APIs de Inteligencia Artificial.

---

## 📸 Vista Previa (Demo)
<!-- REEMPLAZAR_CON_TUS_CAPTURAS: Añade aquí 2 o 3 imágenes de tu proyecto funcionando para dar un gran primer impacto visual. -->
![Dashboard Principal](https://via.placeholder.com/800x400?text=Dashboard+Principal+-+A%C3%B1adir+Captura)
![Asistente IA](https://via.placeholder.com/800x400?text=Asistente+IA+-+A%C3%B1adir+Captura)

---

## 🚀 Características Principales

* **Motor Analítico Directo:** Cálculos ponderados en tiempo real utilizando el factor de expansión poblacional (`expr`), asegurando cifras 100% exactas respecto al gobierno.
* **Procesamiento Optimizado (ETL):** Un script convierte la base bruta de 1.5GB en formato STATA (`.dta`) a un archivo `.parquet` optimizado de solo 9.5MB, permitiendo tiempos de carga de `0.1s`.
* **Dashboard Interactivo:**
  * **Pobreza Territorial:** Tablas y gráficos de incidencia de pobreza multidimensional y por ingresos, con filtros a nivel regional y comunal.
  * **Perfil Demográfico:** Distribución de la población por edad, género, zona urbana/rural, y distribución de ingresos en quintiles.
  * **Carencias:** Análisis de los 20 indicadores de carencia.
* **Asistente IA Integrado:** Un chatbot con contexto inyectado del resumen de la CASEN para responder preguntas descriptivas sobre las regiones. Incluye un **selector de modelos en tiempo real** que permite elegir entre la rapidez de `Meta-Llama-3-8B` y el razonamiento analítico superior de `Qwen2.5-72B`.

---

## 🛠️ Tecnologías Utilizadas

* **Procesamiento de Datos:** `pandas`, `numpy`, `pyarrow`, `fastparquet`
* **Frontend y Visualización:** `gradio`
* **Inteligencia Artificial:** `huggingface_hub` (Inference API)
* **Variables de Entorno:** `python-dotenv`

---

## ⚙️ Instalación y Uso Local

Sigue estos pasos para levantar el dashboard en tu computadora local.

### 1. Clonar el repositorio y preparar el entorno
Clona el repositorio e instala las dependencias en un entorno virtual.
```bash
git clone https://github.com/JoseRicardoSE/Dashboard_Asistencia_IA_CASEN-2024.git
cd Dashboard_Asistencia_IA_CASEN-2024

# Crear y activar entorno virtual
python -m venv venv
# En Windows: venv\Scripts\activate
# En Mac/Linux: source venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt
```

### 2. Configurar Variables de Entorno (Opcional para la IA)
Para habilitar el Asistente de IA, necesitas un token gratuito de HuggingFace.
1. Crea una copia del archivo `.env.example` y renómbralo a `.env`
2. Pega tu token:
```env
HF_TOKEN=hf_XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
```
*Si no configuras esto, el dashboard funcionará perfectamente, pero la pestaña de la IA estará desactivada.*

### 3. Ejecutar la Aplicación
El repositorio incluye archivos `.parquet` de muestra o pre-procesados. Puedes ejecutar el dashboard inmediatamente:

```bash
python app.py
```
Abre tu navegador en `http://127.0.0.1:7860/`.

---

## 📊 Pipeline de Datos (Para Desarrolladores)

Si deseas descargar la base de datos completa de la CASEN y recrear los archivos `.parquet`, sigue este proceso:

1. Descarga el archivo STATA (`casen_2024.dta`) desde la página del Ministerio de Desarrollo Social y Familia.
2. Colócalo en la raíz del proyecto.
3. Ejecuta el script ETL:
```bash
python preprocesar_casen.py
```
Este script leerá el archivo de 1.5GB, filtrará las columnas esenciales, calculará los resúmenes y exportará los modelos comprimidos (`casen_2024_procesada.parquet`).

---

## 📂 Estructura del Proyecto

```text
plataforma-casen-2024/
│
├── app.py                         # Archivo principal con UI y motor analítico
├── preprocesar_casen.py           # Script ETL para procesar base bruta (.dta)
│
├── casen_2024_procesada.parquet   # Datos limpios y comprimidos (~10MB)
├── casen_2024_resumen_region.parquet
│
├── requirements.txt               # Dependencias de Python
├── .gitignore
├── .env.example                   # Plantilla para el token de IA
```

---

## 🤝 Autores
Desarrollado como trabajo práctico grupal de análisis de datos y desarrollo en Python.
[]
