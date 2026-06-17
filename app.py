# ==============================================================================
# PLATAFORMA ANALÍTICA CASEN 2024
# Motor: pandas (cálculo directo) + Gradio (visualización)
# ==============================================================================

import os
import logging
import numpy as np
import pandas as pd
import gradio as gr
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

# ==============================================================================
# 1. CARGA DE DATOS
# ==============================================================================

def cargar_datos():
    """Carga los parquet generados por preprocesar_casen.py"""
    try:
        df = pd.read_parquet("casen_2024_procesada.parquet")
        df_resumen = pd.read_parquet("casen_2024_resumen_region.parquet")
        logging.info(f"Datos reales cargados: {df.shape[0]:,} personas, {df['region'].nunique()} regiones, {df['comuna'].nunique()} comunas")
        return df, df_resumen, True
    except FileNotFoundError:
        logging.warning("Parquet no encontrado. Ejecuta primero: python preprocesar_casen.py")
        # Fallback mínimo
        mock = pd.DataFrame({
            "region": ["Metropolitana", "La Araucanía", "Maule"],
            "pobreza": ["Fuera de la pobreza"] * 3,
            "expr": [100, 100, 100]
        })
        mock_r = pd.DataFrame({
            "Región": ["Metropolitana", "La Araucanía", "Maule"],
            "Pobreza por Ingresos (%)": [13.3, 28.6, 24.3],
            "Pobreza Multidimensional (%)": [19.3, 15.2, 13.3]
        })
        return mock, mock_r, False

df, df_resumen, DATOS_REALES = cargar_datos()

# Listas útiles precalculadas
REGIONES = sorted(df["region"].dropna().unique().tolist())
COMUNAS = sorted(df["comuna"].dropna().unique().tolist()) if "comuna" in df.columns else []

# ==============================================================================
# 2. MOTOR ANALÍTICO (pandas puro)
# ==============================================================================

# Acortar nombres de región para que los gráficos no colapsen
MAPEO_REGIONES = {
    "Región de Arica y Parinacota": "Arica y Parinacota",
    "Región de Tarapacá": "Tarapacá",
    "Región de Antofagasta": "Antofagasta",
    "Región de Atacama": "Atacama",
    "Región de Coquimbo": "Coquimbo",
    "Región de Valparaíso": "Valparaíso",
    "Región Metropolitana de Santiago": "Metropolitana",
    "Región del Libertador Gral. Bernardo O'Higgins": "O'Higgins",
    "Región del Maule": "Maule",
    "Región de Ñuble": "Ñuble",
    "Región del Biobío": "Biobío",
    "Región de La Araucanía": "La Araucanía",
    "Región de Los Ríos": "Los Ríos",
    "Región de Los Lagos": "Los Lagos",
    "Región de Aysén del Gral. Carlos Ibáñez del Campo": "Aysén",
    "Región de Magallanes y de la Antártica Chilena": "Magallanes",
}

def acortar_nombre(nombre):
    return MAPEO_REGIONES.get(nombre, nombre)


def pobreza_por_territorio(nivel="region", territorio=None):
    """Calcula tasas de pobreza ponderadas por factor de expansión."""
    col = nivel  # "region" o "comuna"
    if col not in df.columns:
        return pd.DataFrame()

    if territorio and territorio != "Todas":
        # Si filtramos por región, mostramos comunas de esa región
        mask = df["region"] == territorio
        datos = df[mask].copy()
        col = "comuna"
    else:
        datos = df.copy()

    resultados = []
    for nombre, grupo in datos.groupby(col, observed=True):
        expr_total = grupo["expr"].sum()
        if expr_total == 0:
            continue

        fila = {"Territorio": acortar_nombre(nombre), "Personas (expandido)": f"{int(expr_total):,}"}

        # Pobreza por ingresos
        if "pobreza" in grupo.columns:
            pob_ext = grupo.loc[grupo["pobreza"] == "Pobreza extrema", "expr"].sum()
            pob_no_ext = grupo.loc[grupo["pobreza"] == "Pobreza no extrema", "expr"].sum()
            fila["Extrema (%)"] = round(pob_ext / expr_total * 100, 1)
            fila["No Extrema (%)"] = round(pob_no_ext / expr_total * 100, 1)
            fila["Total (%)"] = round((pob_ext + pob_no_ext) / expr_total * 100, 1)

        # Pobreza multidimensional
        if "pobreza_multi" in grupo.columns:
            pob_multi = grupo.loc[grupo["pobreza_multi"] == "Hogar en pobreza multidimensional", "expr"].sum()
            fila["Multidim. (%)"] = round(pob_multi / expr_total * 100, 1)

        # Ingreso per cápita promedio ponderado
        if "ypchautcor" in grupo.columns:
            mask_ing = grupo["ypchautcor"].notna()
            if mask_ing.sum() > 0:
                ing = np.average(grupo.loc[mask_ing, "ypchautcor"], weights=grupo.loc[mask_ing, "expr"])
                fila["Ingreso p/c (CLP)"] = f"${int(ing):,}"

        resultados.append(fila)

    resultado = pd.DataFrame(resultados)
    if not resultado.empty:
        resultado.sort_values("Total (%)", ascending=False, inplace=True)
    resultado.reset_index(drop=True, inplace=True)
    return resultado


def perfil_demografico(region=None):
    """Genera perfil demográfico de una región o del país."""
    datos = df[df["region"] == region].copy() if region and region != "Todas" else df.copy()
    expr_total = datos["expr"].sum()
    if expr_total == 0:
        return {}

    perfil = {}

    # Sexo
    if "sexo" in datos.columns:
        for val in datos["sexo"].dropna().unique():
            pct = datos.loc[datos["sexo"] == val, "expr"].sum() / expr_total * 100
            perfil[f"% {val}"] = f"{pct:.1f}%"

    # Edad promedio ponderada
    if "edad" in datos.columns:
        mask = datos["edad"].notna()
        if mask.sum() > 0:
            edad_prom = np.average(datos.loc[mask, "edad"], weights=datos.loc[mask, "expr"])
            perfil["Edad promedio"] = f"{edad_prom:.1f} años"

    # Área
    if "area" in datos.columns:
        for val in datos["area"].dropna().unique():
            pct = datos.loc[datos["area"] == val, "expr"].sum() / expr_total * 100
            perfil[f"% {val}"] = f"{pct:.1f}%"

    # Pueblos indígenas
    if "pueblos_indigenas" in datos.columns:
        mask_ind = datos["pueblos_indigenas"].str.contains("Pertenece", na=False)
        pct_ind = datos.loc[mask_ind, "expr"].sum() / expr_total * 100
        perfil["% Pueblos Indígenas"] = f"{pct_ind:.1f}%"

    # Discapacidad
    if "disc_wg" in datos.columns:
        mask_disc = datos["disc_wg"].str.contains("con discapacidad", na=False)
        pct_disc = datos.loc[mask_disc, "expr"].sum() / expr_total * 100
        perfil["% Discapacidad"] = f"{pct_disc:.1f}%"

    # Personas expandidas
    perfil["Población estimada"] = f"{int(expr_total):,}"
    perfil["Personas encuestadas"] = f"{len(datos):,}"

    return perfil


def carencias_multidimensionales(region=None):
    """Calcula tasas de carencia por cada indicador hh_d_*."""
    datos = df[df["region"] == region].copy() if region and region != "Todas" else df.copy()
    expr_total = datos["expr"].sum()

    carencias_cols = [c for c in datos.columns if c.startswith("hh_d_") and not c.endswith("_2015")]
    nombres_legibles = {
        "hh_d_asis": "Asistencia escolar",
        "hh_d_rez": "Rezago escolar",
        "hh_d_esc": "Escolaridad",
        "hh_d_ape": "Apego temprano",
        "hh_d_acc": "Acceso a salud",
        "hh_d_ali": "Alimentación",
        "hh_d_dpf": "Dependencia funcional",
        "hh_d_actsub": "Actividad/Subempleo",
        "hh_d_inf": "Informalidad",
        "hh_d_jub": "Jubilaciones",
        "hh_d_cui": "Cuidados",
        "hh_d_defcuali": "Déficit cualitativo viv.",
        "hh_d_defcuanti": "Déficit cuantitativo viv.",
        "hh_d_accesi": "Accesibilidad",
        "hh_d_medio": "Medio ambiente",
        "hh_d_apoyo": "Apoyo social",
        "hh_d_tsocial": "Trato social",
        "hh_d_conec": "Conectividad",
        "hh_d_seg": "Seguridad",
        "hh_d_contprev": "Cotización previsional",
    }

    resultados = []
    for col in carencias_cols:
        carentes = datos.loc[datos[col] == "Carente", "expr"].sum()
        pct = round(carentes / expr_total * 100, 1) if expr_total > 0 else 0
        nombre = nombres_legibles.get(col, col.replace("hh_d_", "").replace("_", " ").title())
        resultados.append({"Indicador": nombre, "Carencia (%)": pct})

    resultado = pd.DataFrame(resultados)
    resultado.sort_values("Carencia (%)", ascending=False, inplace=True)
    resultado.reset_index(drop=True, inplace=True)
    return resultado


def distribucion_ingresos(region=None):
    """Distribución de ingresos por quintil."""
    datos = df[df["region"] == region].copy() if region and region != "Todas" else df.copy()

    if "qaut" not in datos.columns or "ypchautcor" not in datos.columns:
        return pd.DataFrame()

    resultados = []
    for quintil in ["I", "II", "III", "IV", "V"]:
        mask = datos["qaut"] == quintil
        grupo = datos[mask]
        if len(grupo) == 0:
            continue
        expr_q = grupo["expr"].sum()
        mask_ing = grupo["ypchautcor"].notna()
        if mask_ing.sum() > 0:
            ing_prom = np.average(grupo.loc[mask_ing, "ypchautcor"], weights=grupo.loc[mask_ing, "expr"])
        else:
            ing_prom = 0
        resultados.append({
            "Quintil": f"Q{quintil}",
            "Ingreso p/c Promedio (CLP)": f"${int(ing_prom):,}",
            "Personas (expandido)": f"{int(expr_q):,}"
        })

    return pd.DataFrame(resultados)


# ==============================================================================
# 3. CONTEXTO MACROECONÓMICO
# ==============================================================================

def obtener_indicadores_economicos():
    try:
        respuesta = requests.get('https://mindicador.cl/api', timeout=10)
        respuesta.raise_for_status()
        datos = respuesta.json()
        uf = datos.get('uf', {}).get('valor', 'N/D')
        dolar = datos.get('dolar', {}).get('valor', 'N/D')
        return f"UF: ${uf} CLP | Dólar: ${dolar} CLP"
    except Exception:
        return "Indicadores no disponibles"

indicadores_macro = obtener_indicadores_economicos()

# ==============================================================================
# 4. ASISTENTE IA (usa pandas para responder + LLM para interpretar)
# ==============================================================================

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    llm_client_default = None
else:
    # Cliente por defecto por si se requiere, aunque lo inicializaremos dinámicamente
    llm_client_default = InferenceClient(model="meta-llama/Meta-Llama-3-8B-Instruct", token=HF_TOKEN)

def responder_pregunta(mensaje_usuario, historial_dialogo, modelo_seleccionado):
    if not HF_TOKEN:
        return "⚠️ Asistente IA desactivado. No se encontró el token de HuggingFace en el archivo .env (HF_TOKEN)."
    
    llm_client = InferenceClient(model=modelo_seleccionado, token=HF_TOKEN)

    """Inyecta el resumen regional real (no los 218K registros) al LLM."""
    # Preparar contexto compacto con datos reales
    resumen_json = df_resumen.to_json(orient="records", force_ascii=False)

    system_prompt = (
        "Rol: Eres un analista experto en la Encuesta CASEN 2024 de Chile.\n\n"
        "Reglas:\n"
        "1. Basa TODAS tus respuestas en los datos JSON proporcionados.\n"
        "2. PROHIBIDO inventar cifras. Si no tienes el dato, dilo.\n"
        "3. Responde en español, de forma clara y con cifras exactas del JSON.\n"
        "4. Cuando compares regiones, menciona las cifras de ambas.\n\n"
        f"Indicadores económicos actuales: {indicadores_macro}\n\n"
        f"Datos CASEN 2024 por región (tasas ponderadas con factor de expansión):\n{resumen_json}\n"
    )

    mensajes = [{"role": "system", "content": system_prompt}]
    for msg in historial_dialogo:
        if isinstance(msg, (list, tuple)) and len(msg) == 2:
            mensajes.append({"role": "user", "content": msg[0]})
            mensajes.append({"role": "assistant", "content": msg[1]})
        elif isinstance(msg, dict) and "role" in msg:
            mensajes.append(msg)

    mensajes.append({"role": "user", "content": mensaje_usuario})

    try:
        resp = llm_client.chat_completion(messages=mensajes, max_tokens=700, temperature=0.1)
        return resp.choices[0].message.content
    except Exception as e:
        return f"Error en inferencia: {str(e)}"

# ==============================================================================
# 5. INTERFAZ GRADIO
# ==============================================================================

def actualizar_dashboard(region_seleccionada):
    """Callback principal: actualiza todas las tablas y gráficos."""
    region = None if region_seleccionada == "Todo Chile" else region_seleccionada

    # Tabla de pobreza territorial
    tabla_pobreza = pobreza_por_territorio(territorio=region)

    # Gráfico de barras - pobreza por territorio
    if not tabla_pobreza.empty:
        cols_grafico = ["Territorio", "Total (%)", "Multidim. (%)"]
        cols_existentes = [c for c in cols_grafico if c in tabla_pobreza.columns]
        df_graf = tabla_pobreza[cols_existentes].copy()
        # Limitar a top 20 si hay muchas comunas
        if len(df_graf) > 20:
            df_graf = df_graf.head(20)
        df_melted = df_graf.melt(
            id_vars=["Territorio"],
            var_name="Dimension",
            value_name="Porcentaje"
        )
    else:
        df_melted = pd.DataFrame(columns=["Territorio", "Dimension", "Porcentaje"])

    # Perfil demográfico
    perfil = perfil_demografico(region)
    df_perfil = pd.DataFrame([
        {"Indicador": k, "Valor": v} for k, v in perfil.items()
    ])

    # Carencias multidimensionales
    tabla_carencias = carencias_multidimensionales(region)

    # Distribución de ingresos
    tabla_ingresos = distribucion_ingresos(region)

    titulo = f"Pobreza por {'Comunas' if region else 'Regiones'}"
    if region:
        titulo += f" - {region}"

    return tabla_pobreza, df_melted, df_perfil, tabla_carencias, tabla_carencias, tabla_ingresos, titulo


# Tema visual
tema = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="slate",
    font=[gr.themes.GoogleFont("Roboto"), "Arial", "sans-serif"]
)

with gr.Blocks(title="Plataforma CASEN 2024") as app:

    gr.Markdown("# 📊 Plataforma Analítica CASEN 2024 — Datos Reales")

    if DATOS_REALES:
        gr.Markdown(
            f"✅ **{df.shape[0]:,} personas encuestadas** | "
            f"**{df['region'].nunique()} regiones** | "
            f"**{df['comuna'].nunique() if 'comuna' in df.columns else 0} comunas** | "
            f"{indicadores_macro}"
        )
    else:
        gr.Markdown("⚠️ **Datos de demostración.** Ejecuta `python preprocesar_casen.py` para usar datos reales.")

    # Selector de región
    opciones = ["Todo Chile"] + REGIONES
    selector = gr.Dropdown(
        choices=opciones,
        value="Todo Chile",
        label="Seleccione región (o 'Todo Chile' para vista nacional)",
        info="Al seleccionar una región se muestran sus comunas"
    )

    with gr.Tabs():

        # ── TAB 1: Dashboard de Pobreza ──
        with gr.Tab("📈 Pobreza Territorial"):
            titulo_grafico = gr.Markdown("### Pobreza por Regiones")
            # Valores iniciales
            _, df_init_melted, _, _, _, _, _ = actualizar_dashboard("Todo Chile")
            grafico = gr.BarPlot(
                value=df_init_melted,
                x="Territorio",
                y="Porcentaje",
                color="Dimension",
                title="Incidencia de Pobreza (%)",
                tooltip=["Territorio", "Dimension", "Porcentaje"],
                x_label_angle=-45,
                height=450,
            )
            tabla_pob = gr.Dataframe(
                value=pobreza_por_territorio(),
                label="Datos de Pobreza",
                interactive=False,
                wrap=False
            )

        # ── TAB 2: Perfil Demográfico ──
        with gr.Tab("👤 Perfil Demográfico"):
            with gr.Row():
                with gr.Column(scale=1):
                    tabla_perfil = gr.Dataframe(
                        value=pd.DataFrame([{"Indicador": k, "Valor": v} for k, v in perfil_demografico().items()]),
                        label="Perfil Sociodemográfico",
                        interactive=False,
                        wrap=False
                    )
                with gr.Column(scale=1):
                    tabla_ing = gr.Dataframe(
                        value=distribucion_ingresos(),
                        label="Distribución de Ingresos por Quintil",
                        interactive=False,
                        wrap=False
                    )

        # ── TAB 3: Carencias Multidimensionales ──
        with gr.Tab("📉 Carencias Multidimensionales"):
            with gr.Row():
                with gr.Column(scale=2):
                    df_car_init = carencias_multidimensionales()
                    grafico_car = gr.BarPlot(
                        value=df_car_init,
                        x="Carencia (%)",
                        y="Indicador",
                        title="Carencias Multidimensionales (%)",
                        tooltip=["Indicador", "Carencia (%)"],
                        height=550,
                        sort="x",
                    )
                with gr.Column(scale=1):
                    tabla_car = gr.Dataframe(
                        value=df_car_init,
                        label="Indicadores de Carencia (% de la poblacion)",
                        interactive=False,
                        wrap=False
                    )

        # ── TAB 4: Asistente IA ──
        with gr.Tab("🤖 Asistente IA"):
            gr.Markdown(
                "### Consultas con Inteligencia Artificial\n"
                "El asistente tiene acceso al resumen regional de la CASEN 2024. "
                "Para cálculos exactos, usa las pestañas anteriores.\n\n"
                "*Ejemplo: '¿Qué región tiene mayor brecha entre pobreza por ingresos y multidimensional?'*"
            )
            
            modelo_dropdown = gr.Dropdown(
                choices=["meta-llama/Meta-Llama-3-8B-Instruct", "Qwen/Qwen2.5-72B-Instruct"],
                value="meta-llama/Meta-Llama-3-8B-Instruct",
                label="Seleccionar Modelo de IA",
                info="Qwen 72B es más inteligente, pero podría demorar más en responder."
            )
            
            gr.ChatInterface(
                fn=responder_pregunta,
                chatbot=gr.Chatbot(height=450),
                textbox=gr.Textbox(
                    placeholder="Escriba su consulta sobre la CASEN 2024...",
                    container=False, scale=8
                ),
                additional_inputs=[modelo_dropdown]
            )

    # Conectar selector con todos los outputs
    selector.change(
        fn=actualizar_dashboard,
        inputs=[selector],
        outputs=[tabla_pob, grafico, tabla_perfil, tabla_car, grafico_car, tabla_ing, titulo_grafico]
    )

if __name__ == "__main__":
    app.launch(share=True, theme=tema)
