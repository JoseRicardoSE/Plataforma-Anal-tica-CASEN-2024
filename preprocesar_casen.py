# ==============================================================================
# PREPROCESAMIENTO CASEN 2024: .dta (1.6 GB) → .parquet (~100 MB)
# ==============================================================================
# Ejecutar UNA SOLA VEZ:  python preprocesar_casen.py
#
# Genera:
#   1. casen_2024_procesada.parquet   → Base individual con variables clave
#   2. casen_2024_resumen_region.parquet → Resumen agregado por región (para app.py)
#   3. columnas_disponibles.txt       → Referencia de todas las 877 variables
# ==============================================================================

import pandas as pd
import numpy as np
import os
import sys
import warnings
import time

warnings.filterwarnings('ignore')

# ─── COLUMNAS VERIFICADAS (existentes en casen_2024.dta) ─────────────────────
# Cada columna fue comprobada contra la base real el 2026-06-17

COLUMNAS = [
    # ── Identificación ──
    'folio',              # int32  - ID del hogar
    'id_persona',         # int8   - ID persona dentro del hogar
    'id_vivienda',        # int32  - ID vivienda

    # ── Factor de expansión (OBLIGATORIO para estadísticas) ──
    'expr',               # int16  - Factor expansión regional

    # ── Territorio ──
    'region',             # category - "Región Metropolitana de Santiago", etc.
    'area',               # category - "Urbano" / "Rural"

    # ── Demografía ──
    'sexo',               # category - "Hombre" / "Mujer"
    'edad',               # int16
    'ecivil',             # category - "8. Soltero(a)", etc.
    'numper',             # int8   - Nº personas en el hogar
    'numnuc',             # int8   - Nº núcleos en el hogar

    # ── Pobreza (5 variables clave) ──
    'pobreza',            # category - "Fuera de la pobreza" / "Pobreza no extrema" / "Pobreza extrema"
    'pobreza_2013',       # category - Pobreza metodología 2013
    'pobreza_severa',     # category - "No pobreza" / "Solo pobreza multidimensional" / "Solo pobreza por ingresos" / "Pobreza Severa"
    'pobreza_multi',      # category - "Hogar en/fuera de pobreza multidimensional"
    'pobreza_multi_2015', # category - Pobreza multidimensional metodología 2015

    # ── Líneas de pobreza ──
    'li',                 # float32 - Línea de indigencia (ej: 158,063 CLP)
    'lp',                 # float32 - Línea de pobreza (ej: 237,094 CLP)
    'nae',                # float64 - Nº adultos equivalentes

    # ── Ingresos del hogar (corregidos, los que importan) ──
    'ytotcorh',           # int32   - Ingreso total corregido del hogar
    'yautcorh',           # int32   - Ingreso autónomo corregido del hogar
    'ymonecorh',          # float32 - Ingreso monetario corregido del hogar
    'yoprcorh',           # int32   - Ingreso ocupación principal corregido hogar
    'yoautcorh',          # int32   - Ingreso autónomo otra ocupación corregido hogar

    # ── Ingresos per cápita ──
    'ypchautcor',         # int32   - Ingreso per cápita autónomo corregido
    'ypchtotcor',         # int32   - Ingreso per cápita total corregido
    'ypchtrabcor',        # int32   - Ingreso per cápita del trabajo corregido
    'ypchmonecor',        # int32   - Ingreso per cápita monetario corregido
    'yae',                # float32 - Ingreso autónomo equivalente

    # ── Subsidios y transferencias ──
    'ysubh',              # int32   - Total subsidios del hogar
    'yauth',              # int32   - Ingreso autónomo del hogar
    'ytoth',              # int32   - Ingreso total del hogar (sin corregir)

    # ── Quintiles y deciles ──
    'dau',                # category - Decil autónomo (I a X)
    'qaut',               # category - Quintil autónomo (I a V)

    # ── Educación ──
    'e1',                 # category - Alfabetización: "Sí, lee y escribe" / "No"
    'e6a',                # category - Nivel educacional alcanzado
    'esc',                # float   - Escolaridad en años

    # ── Trabajo ──
    'activ',              # category - "Ocupados" / "Desocupados" / "Inactivos"
    'o15',                # category - Categoría ocupacional
    'rama1',              # category - Rama de actividad económica
    'contrato',           # category - Tiene contrato: "Sí" / "No"
    'cotiza',             # category - Cotiza: "Sí" / "No"

    # ── Salud ──
    's13',                # category - Sistema previsional salud (FONASA, Isapre, etc.)

    # ── Vivienda ──
    'v1',                 # category - Tipo de vivienda
    'v9',                 # category - Tenencia de la vivienda (propia, arrendada, etc.)
    'ind_hacina',         # category - Hacinamiento
    'ind_cal_glob',       # category - Calidad global vivienda
    'ind_mat',            # category - Materialidad vivienda
    'ten_viv',            # category - Tenencia vivienda (derivada)
    'ten_viv_f',          # category - Tenencia formal/informal

    # ── Variables derivadas socioeconómicas ──
    'tipohogar',          # category - "Nuclear Biparental", "Monoparental", etc.
    'analfabetismo',      # category - "Sabe leer y escribir" / "No sabe"
    'disc_wg',            # category - Discapacidad
    'lugar_nac',          # category - "Nacido(a) en Chile" / "Nacido(a) fuera de Chile"
    'pueblos_indigenas',  # category - Pertenencia a pueblos indígenas
    'men18c',             # category - ¿Hay menores de 18?
    'may60c',             # category - ¿Hay mayores de 60?
    'cuidador',           # category - ¿Es cuidador?

    # ── Indicadores de carencia multidimensional (hh_d_*) ──
    # Cada uno es "Carente" o "No carente"
    'hh_d_asis',          # Asistencia escolar
    'hh_d_rez',           # Rezago escolar
    'hh_d_esc',           # Escolaridad
    'hh_d_ape',           # Apego temprano
    'hh_d_acc',           # Acceso a salud
    'hh_d_ali',           # Alimentación
    'hh_d_dpf',           # Dependencia funcional
    'hh_d_actsub',        # Actividad laboral/subempleo
    'hh_d_inf',           # Informalidad
    'hh_d_jub',           # Jubilaciones
    'hh_d_cui',           # Cuidados
    'hh_d_defcuali',      # Déficit cualitativo vivienda
    'hh_d_defcuanti',     # Déficit cuantitativo vivienda
    'hh_d_accesi',        # Accesibilidad
    'hh_d_medio',         # Medio ambiente
    'hh_d_apoyo',         # Apoyo y participación social
    'hh_d_tsocial',       # Trato social
    'hh_d_conec',         # Conectividad
    'hh_d_seg',           # Seguridad
    'hh_d_contprev',      # Cotización previsional
]


def main():
    t_inicio = time.time()

    print("=" * 70)
    print("  PREPROCESAMIENTO CASEN 2024")
    print("  Base STATA (1.6 GB) → Parquet optimizado")
    print("=" * 70)

    # ─── Paso 1: Verificar archivos ──────────────────────────────────────────
    archivo_base = "casen_2024.dta"
    archivo_geo = "casen_2024_provincia_comuna.dta"

    if not os.path.exists(archivo_base):
        print(f"\n❌ No se encontró '{archivo_base}' en {os.getcwd()}")
        sys.exit(1)

    tamaño_mb = os.path.getsize(archivo_base) / (1024 * 1024)
    print(f"\n📁 Base principal: {archivo_base} ({tamaño_mb:.0f} MB)")

    tiene_geo = os.path.exists(archivo_geo)
    if tiene_geo:
        print(f"📁 Base geográfica: {archivo_geo} (disponible)")
    else:
        print(f"⚠️  Base geográfica: {archivo_geo} (no encontrada, se omitirá)")

    # ─── Paso 2: Leer metadatos para verificar columnas ──────────────────────
    print("\n🔍 Paso 1/5: Leyendo estructura del archivo...")
    df_muestra = pd.read_stata(archivo_base, chunksize=10, convert_categoricals=False)
    primera = next(df_muestra)
    columnas_disponibles = primera.columns.tolist()
    print(f"   ✅ {len(columnas_disponibles)} columnas en el archivo")

    # Guardar referencia completa de columnas
    with open("columnas_disponibles.txt", "w", encoding="utf-8") as f:
        f.write(f"TODAS LAS COLUMNAS DE casen_2024.dta ({len(columnas_disponibles)} total)\n")
        f.write("=" * 60 + "\n\n")
        for i, col in enumerate(columnas_disponibles, 1):
            dtype = primera[col].dtype
            f.write(f"{i:3d}. {col:<30s} ({dtype})\n")

    # ─── Paso 3: Filtrar columnas existentes ─────────────────────────────────
    print("\n📋 Paso 2/5: Verificando columnas solicitadas...")
    cols_ok = [c for c in COLUMNAS if c in columnas_disponibles]
    cols_no = [c for c in COLUMNAS if c not in columnas_disponibles]

    print(f"   ✅ Encontradas: {len(cols_ok)}/{len(COLUMNAS)}")
    if cols_no:
        print(f"   ⚠️  No encontradas ({len(cols_no)}): {', '.join(cols_no)}")

    if not cols_ok:
        print("\n❌ ERROR CRÍTICO: Ninguna columna coincide.")
        sys.exit(1)

    # ─── Paso 4: Cargar datos ────────────────────────────────────────────────
    print(f"\n⏳ Paso 3/5: Cargando {len(cols_ok)} columnas...")
    print(f"   (Puede tomar 1-3 minutos para {tamaño_mb:.0f} MB)")

    t_carga = time.time()
    df = pd.read_stata(archivo_base, columns=cols_ok)
    t_carga_fin = time.time()

    print(f"   ✅ {df.shape[0]:,} filas × {df.shape[1]} columnas en {t_carga_fin - t_carga:.1f}s")

    mem_antes = df.memory_usage(deep=True).sum() / (1024**2)
    print(f"   📊 Uso RAM: {mem_antes:.1f} MB")

    # ─── Paso 4b: Merge con comunas/provincias ──────────────────────────────
    if tiene_geo:
        print("\n🗺️  Vinculando comunas y provincias...")
        df_geo = pd.read_stata(archivo_geo)
        # El archivo geo tiene: folio, id_persona, expp, expc, provincia, comuna
        cols_merge = ['folio', 'id_persona']
        cols_traer = ['provincia', 'comuna']

        # Verificar que las columnas de merge existan en ambos
        if all(c in df.columns for c in cols_merge) and all(c in df_geo.columns for c in cols_merge + cols_traer):
            df = df.merge(df_geo[cols_merge + cols_traer], on=cols_merge, how='left')
            n_con_comuna = df['comuna'].notna().sum()
            print(f"   ✅ Merge exitoso. {n_con_comuna:,}/{df.shape[0]:,} filas con comuna asignada")
        else:
            print(f"   ⚠️  Columnas incompatibles, omitiendo merge")

    # ─── Paso 5: Optimizar tipos de datos ────────────────────────────────────
    print("\n🔧 Paso 4/5: Optimizando tipos de datos...")

    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = df[col].astype('float32')

    for col in df.select_dtypes(include=['int64']).columns:
        col_max = df[col].max()
        col_min = df[col].min()
        if col_min >= -32768 and col_max <= 32767:
            df[col] = df[col].astype('int16')
        elif col_min >= -2_147_483_648 and col_max <= 2_147_483_647:
            df[col] = df[col].astype('int32')

    mem_despues = df.memory_usage(deep=True).sum() / (1024**2)
    print(f"   ✅ RAM: {mem_antes:.0f} MB → {mem_despues:.0f} MB (ahorro: {mem_antes - mem_despues:.0f} MB)")

    # ─── Paso 6: Guardar Parquet ─────────────────────────────────────────────
    print("\n💾 Paso 5/5: Guardando archivos...")

    # 6a. Base individual completa
    salida = "casen_2024_procesada.parquet"
    df.to_parquet(salida, index=False, compression='snappy')
    tam_salida = os.path.getsize(salida) / (1024**2)
    print(f"   ✅ {salida} ({tam_salida:.1f} MB)")

    # 6b. Resumen agregado por región (para app.py)
    print("\n📊 Generando resumen regional agregado...")

    # Calcular estadísticas ponderadas por región
    # La variable 'pobreza' tiene categorías; necesitamos crear dummies
    df_resumen_list = []

    for reg, grupo in df.groupby('region'):
        expr_total = grupo['expr'].sum()
        n_personas = len(grupo)

        fila = {'Región': reg, 'n_muestra': n_personas, 'n_expandido': expr_total}

        # Pobreza por ingresos (ponderada)
        if 'pobreza' in grupo.columns:
            pob_extrema = grupo.loc[grupo['pobreza'] == 'Pobreza extrema', 'expr'].sum()
            pob_no_ext = grupo.loc[grupo['pobreza'] == 'Pobreza no extrema', 'expr'].sum()
            pob_total = pob_extrema + pob_no_ext
            fila['Pobreza por Ingresos (%)'] = round(pob_total / expr_total * 100, 1) if expr_total > 0 else 0
            fila['Pobreza Extrema (%)'] = round(pob_extrema / expr_total * 100, 1) if expr_total > 0 else 0

        # Pobreza multidimensional (ponderada)
        if 'pobreza_multi' in grupo.columns:
            pob_multi = grupo.loc[grupo['pobreza_multi'] == 'Hogar en pobreza multidimensional', 'expr'].sum()
            fila['Pobreza Multidimensional (%)'] = round(pob_multi / expr_total * 100, 1) if expr_total > 0 else 0

        # Ingreso promedio ponderado per cápita
        if 'ypchautcor' in grupo.columns:
            mask = grupo['ypchautcor'].notna()
            if mask.sum() > 0:
                ingreso_pond = np.average(grupo.loc[mask, 'ypchautcor'], weights=grupo.loc[mask, 'expr'])
                fila['Ingreso Per Cápita Promedio (CLP)'] = round(ingreso_pond)

        df_resumen_list.append(fila)

    df_resumen = pd.DataFrame(df_resumen_list)
    df_resumen.sort_values('Pobreza por Ingresos (%)', ascending=False, inplace=True)

    salida_resumen = "casen_2024_resumen_region.parquet"
    df_resumen.to_parquet(salida_resumen, index=False)
    print(f"   ✅ {salida_resumen}")

    # Mostrar el resumen
    print(f"\n{'='*70}")
    print("  RESUMEN REGIONAL DE POBREZA CASEN 2024")
    print(f"{'='*70}")
    pd.set_option('display.max_columns', 10)
    pd.set_option('display.width', 120)
    pd.set_option('display.max_colwidth', 40)
    print(df_resumen.to_string(index=False))

    # ─── Resumen final ───────────────────────────────────────────────────────
    t_total = time.time() - t_inicio

    print(f"\n{'='*70}")
    print(f"  ✅ COMPLETADO en {t_total:.0f} segundos")
    print(f"{'='*70}")
    print(f"""
  Archivos generados:
    1. {salida} ({tam_salida:.1f} MB) ← Base individual
    2. {salida_resumen} ← Resumen por región para app.py
    3. columnas_disponibles.txt ← Referencia de 877 variables

  Para usar en app.py:
    df_casen = pd.read_parquet("casen_2024_procesada.parquet")
    df_resumen = pd.read_parquet("casen_2024_resumen_region.parquet")
""")


if __name__ == "__main__":
    main()
