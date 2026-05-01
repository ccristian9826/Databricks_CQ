# Databricks notebook source
# MAGIC %md
# MAGIC ## **CREACIÓN DEL ENTORNO**

# COMMAND ----------

# MAGIC %md
# MAGIC 1.1 Creamos nuestra esturctura dentro de Unity Catalog

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE CATALOG IF NOT EXISTS sesion1;
# MAGIC CREATE SCHEMA IF NOT EXISTS sesion1.data COMMENT 'This is customer catalog';
# MAGIC CREATE VOLUME IF NOT EXISTS sesion1.data.landing;

# COMMAND ----------

# MAGIC %md
# MAGIC 1.2 Incorporacion de Datos a un Volumen

# COMMAND ----------

# MAGIC %sh
# MAGIC curl -L https://raw.githubusercontent.com/regarcia-magister/EAM-ETL-IA/refs/heads/main/sesion%201-2/data/nutrients_csvfile.csv -o /Volumes/sesion1/data/landing/nutrients_csvfile.csv

# COMMAND ----------

# MAGIC %md
# MAGIC 1.3 Crear una Tabla

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE sesion1.data.tabla_simple (
# MAGIC   id INT,
# MAGIC   letra STRING,
# MAGIC   valor DOUBLE
# MAGIC );
# MAGIC
# MAGIC INSERT INTO sesion1.data.tabla_simple VALUES
# MAGIC   (1, 'A', 10.5),
# MAGIC   (2, 'B', 20.0),
# MAGIC   (3, 'C', 30.75);

# COMMAND ----------

# MAGIC %sql
# MAGIC Select * from sesion1.data.tabla_simple;

# COMMAND ----------

# MAGIC %md
# MAGIC 1.4 Crear una Vista

# COMMAND ----------

# DBTITLE 1,Crear vista vw_tabla_simple
# MAGIC %sql
# MAGIC CREATE OR REPLACE VIEW sesion1.data.vw_tabla_simple AS
# MAGIC SELECT
# MAGIC   id,
# MAGIC   letra
# MAGIC FROM sesion1.data.tabla_simple
# MAGIC WHERE valor > 15;

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from sesion1.data.vw_tabla_simple

# COMMAND ----------

# MAGIC %md
# MAGIC 1.5 Crear una función

# COMMAND ----------

# MAGIC %sql
# MAGIC create schema if not exists sesion1.security;

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE FUNCTION sesion1.security.fn_mayor_que_10(x DOUBLE)
# MAGIC RETURNS BOOLEAN
# MAGIC RETURN x > 12;

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT *
# MAGIC FROM sesion1.data.tabla_simple
# MAGIC WHERE sesion1.security.fn_mayor_que_10(valor);

# COMMAND ----------

# MAGIC %md
# MAGIC 1.6 Creacion de DataFrame

# COMMAND ----------

path_data_demo = "/Volumes/sesion1/data/landing/data_demo.csv"
df = spark.read.csv(
    path=path_data_demo,
    header=True,     
    inferSchema=True
)

# COMMAND ----------

# MAGIC %md
# MAGIC 1.7 Visualizar el Data Frame

# COMMAND ----------

display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2. Delta Tables

# COMMAND ----------

# MAGIC %md
# MAGIC 2.1 Subir un CSV al volumen

# COMMAND ----------

# MAGIC %sh
# MAGIC curl -L https://raw.githubusercontent.com/regarcia-magister/EAM-ETL-IA/refs/heads/main/sesion%201-2/data/ventas_2025.csv -o /Volumes/sesion1/data/landing/ventas.csv
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC 2.2 Crear el Data Frame

# COMMAND ----------

from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType

base_path = "dbfs:/Volumes/sesion1/data/landing/"

# Definir el esquema manualmente
schema = StructType([
    # Nombre, Tipo de dato, Requerido
    StructField("id", StringType(), True),
    StructField("fecha", StringType(), True),
    StructField("producto", StringType(), True),
    StructField("cantidad", IntegerType(), True),
    StructField("precio", DoubleType(), True)
])

df_csv = spark.read.csv(
    path=base_path+"ventas.csv",
    header=True,
    schema=schema,
    sep=","
)

# COMMAND ----------

# MAGIC %md
# MAGIC 2.3 Creamos la tabla Delta

# COMMAND ----------


df_csv.write.format("delta").mode("overwrite").saveAsTable("sesion1.data.productos")

# COMMAND ----------

# MAGIC %md
# MAGIC 2.4 Leer Tabla Delta

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM sesion1.data.productos;

# COMMAND ----------

# MAGIC %md
# MAGIC 2.4.1 Lectura de la tabla Delta con DF de Spark

# COMMAND ----------

df_delta = spark.read.table("sesion1.data.productos")

df_delta.show()

# COMMAND ----------

# MAGIC %md
# MAGIC 2.5 Modificar una Tabla

# COMMAND ----------

# DBTITLE 1,Insert new record into productos
# MAGIC %sql
# MAGIC -- INSERT: añadir un nuevo registro
# MAGIC INSERT INTO sesion1.data.productos (id, fecha, producto, cantidad, precio)
# MAGIC VALUES ('A001', '2025-11-25', 'Torre E-138', 3, 120000.50);

# COMMAND ----------

# MAGIC %md
# MAGIC 2.5.1 Update

# COMMAND ----------

# MAGIC %sql
# MAGIC -- UPDATE (modify): modificar un registro existente
# MAGIC UPDATE sesion1.data.productos
# MAGIC SET cantidad = 5,
# MAGIC     precio   = 119000.00
# MAGIC WHERE id = '1';

# COMMAND ----------

# MAGIC %md
# MAGIC 2.5.2 Delete

# COMMAND ----------

# MAGIC %sql
# MAGIC -- DELETE: eliminar un registro
# MAGIC DELETE FROM sesion1.data.productos
# MAGIC WHERE id = 'A001';

# COMMAND ----------

# MAGIC %md
# MAGIC 2.5.3 Merge

# COMMAND ----------

from delta.tables import DeltaTable

delta_table = DeltaTable.forName(spark, "sesion1.data.productos")  

# Nuevos datos a insertar/actualizar
columns = ["id", "fecha", "producto", "cantidad", "precio"]

nuevos_datos = [(3, "2025-05-24", "Monitor", 1, 179.99), (4, "2025-05-24", "Impresora", 2, 89.99)]
df_updates = spark.createDataFrame(nuevos_datos, columns)


delta_table.alias("target").merge(
    df_updates.alias("source"),
    "target.id = source.id") \
  .whenMatchedUpdateAll() \
  .whenNotMatchedInsertAll() \
  .execute()

# COMMAND ----------

# MAGIC %md
# MAGIC 2.6 Time Travel

# COMMAND ----------

# MAGIC %md
# MAGIC 2.6.1 Ver el historial de versiones de la tabla

# COMMAND ----------

# MAGIC %sql
# MAGIC DESCRIBE HISTORY sesion1.data.productos;

# COMMAND ----------

# MAGIC %md
# MAGIC 2.6.2 consultar versión anterior (Time Travel por versión)

# COMMAND ----------

# DBTITLE 1,Cell 43
# MAGIC %sql
# MAGIC SELECT * FROM sesion1.data.productos VERSION AS OF 8;
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC 2.6.3 Consultar por TimeStamp

# COMMAND ----------

# DBTITLE 1,Cell 45
# MAGIC %sql
# MAGIC SELECT * 
# MAGIC FROM sesion1.data.productos 
# MAGIC TIMESTAMP AS OF '2026-04-30T05:11:32';
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC 2.6.4 Restaurar la versión anterior

# COMMAND ----------

# DBTITLE 1,Cell 47
# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE sesion1.data.productos_v3
# MAGIC AS SELECT * FROM sesion1.data.productos VERSION AS OF 6;

# COMMAND ----------

# MAGIC %md
# MAGIC 3 Introducción a PySpark

# COMMAND ----------

# MAGIC %md
# MAGIC 3.0 Preparación entorno

# COMMAND ----------

# MAGIC %sql
# MAGIC create catalog if not exists sesion1;
# MAGIC create schema if not exists sesion1.sparkintro;
# MAGIC create volume if not exists sesion1.sparkintro.landing;

# COMMAND ----------

# MAGIC %md
# MAGIC 3.1 Almacenar datos en el volumen

# COMMAND ----------

# MAGIC %sh
# MAGIC curl -L https://raw.githubusercontent.com/regarcia-magister/EAM-ETL-IA/refs/heads/main/sesion%201-2/data/spark_intro.csv -o /Volumes/sesion1/sparkintro/landing/spark_intro.csv
# MAGIC
# MAGIC %sh
# MAGIC curl -L https://raw.githubusercontent.com/regarcia-magister/EAM-ETL-IA/refs/heads/main/sesion%201-2/data/dim_spark_intro.csv -o /Volumes/sesion1/sparkintro/landing/dim_spark_intro.csv

# COMMAND ----------

# MAGIC %md
# MAGIC 3.2 Crear el DF

# COMMAND ----------

from pyspark.sql import functions as F

# Read CSV from volume as a Spark DataFrame
sales_df = (
    spark.read
    .option("header", True)
    .option("inferSchema", True)  # Let Spark infer column types for now
    .csv("/Volumes/sesion1/sparkintro/landing/spark_intro.csv")
)

display(sales_df)

# COMMAND ----------

# MAGIC %md
# MAGIC 3.3 Select(): elegir columnas

# COMMAND ----------

# Select a subset of columns
sales_simple_df = sales_df.select("order_id", "order_date", "country", "units_sold")
display(sales_simple_df)

# COMMAND ----------

# MAGIC %md
# MAGIC 3.4 filter() / where(): filtrar filas

# COMMAND ----------

# DBTITLE 1,Filter sales for Spain & big orders
# Filter sales only for Spain
spain_sales_df = sales_df.filter(F.col("country") == "Spain")
display(spain_sales_df)

# Filter sales with more than 3 units sold
big_orders_df = sales_df.filter(F.col("units_sold") > 3)
display(big_orders_df)

# The previous error occurred because this line is misplaced:
# '3.4 filter() / where(): filtrar filas' should NOT be part of the code cell, so it is removed.


# COMMAND ----------

# MAGIC %md
# MAGIC 3.5 withColumn(): crear o transformar columnas

# COMMAND ----------

# DBTITLE 1,Create new column: total_sales
from pyspark.sql import functions as F

# Create a new column with total sales amount
sales_with_total_df = sales_df.withColumn(
    "total_sales",
    F.col("units_sold") * F.col("unit_price")  # Simple numeric expression
)

display(sales_with_total_df)

# COMMAND ----------

# MAGIC %md
# MAGIC 3.6 groupBy().agg(): agregaciones

# COMMAND ----------

# Aggregate total units and total sales by country
sales_country_df = (
    sales_with_total_df
    .groupBy("country")
    .agg(
        F.sum("units_sold").alias("total_units"),
        F.sum("total_sales").alias("total_sales_amount")
    )
)

display(sales_country_df)

# COMMAND ----------

# MAGIC %md
# MAGIC 3.7 orderBy(): ordenar datos

# COMMAND ----------

# Order countries by total sales amount (descending)
sorted_sales_country_df = sales_country_df.orderBy(F.col("total_sales_amount").desc())
display(sorted_sales_country_df)

# COMMAND ----------

# MAGIC %md
# MAGIC 3.8 join(): combinar datos de distintas tablas

# COMMAND ----------

# DBTITLE 1,Cell 65
# Read product dimension from volume
product_dim_df = (
    spark.read
    .option("header", True)
    .option("inferSchema", True)
    .csv("/Volumes/sesion1/sparkintro/landing/dim_spark_intro.csv")
)

display(product_dim_df)

# COMMAND ----------

# Join sales with product dimension on 'product'
sales_enriched_df = (
    sales_with_total_df.alias("s")
    .join(
        product_dim_df.alias("p"),
        on="product",   # Join key
        how="left"     # Keep all sales even if some product has no dimension row
    )
)

display(sales_enriched_df)

# COMMAND ----------

# MAGIC %md
# MAGIC 3.9 Write(): Guardar el resultado

# COMMAND ----------

# Save aggregated sales by country as a managed Delta table
(
    sorted_sales_country_df
    .write
    .mode("overwrite")          # Overwrite existing table if it exists
    .saveAsTable("sesion1.sparkintro.sales_by_country")
)

# Check that we can read it back as a table
result_df = spark.table("sesion1.sparkintro.sales_by_country")
display(result_df)


# COMMAND ----------

# MAGIC %md
# MAGIC ### 4. Analisis Exploratorio con Spark

# COMMAND ----------

# MAGIC %md
# MAGIC 4.1 Dimension, Tipos de Datos y primeras filas

# COMMAND ----------

# Número de filas y columnas
print(f"Dimensiones del DataFrame: {df.count()} filas, {len(df.columns)} columnas")

# Tipos de datos
df.printSchema()

# Mostrar las primeras filas
df.show(5)

# COMMAND ----------

# MAGIC %md
# MAGIC 4.2 Verificacion de Nulos

# COMMAND ----------

from pyspark.sql.functions import isnan, when, count, col

display(
    df.select([
    count(
        when(
            isnan(c), c
        )
    ).alias(c) for c in df.columns])
)

# COMMAND ----------

# MAGIC %md
# MAGIC **4.3 Info Estadistica**

# COMMAND ----------

# MAGIC %md
# MAGIC 4.3.1 Metodo Describe

# COMMAND ----------


df.describe().show()

# COMMAND ----------

# MAGIC %md
# MAGIC 4.3.2 Metodo Summary

# COMMAND ----------

display(df.summary())

# COMMAND ----------

# MAGIC %md
# MAGIC 4.3.3 Matriz de Correlación

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.stat import Correlation
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Seleccionar columnas numéricas de Spark
numeric_cols = [c for c, t in df.dtypes if t in ("int", "bigint", "double", "float", "decimal")]

# 2. Crear columna vectorial para Spark ML
# Pasar de Spark a Pandas
pdf = df.toPandas()

# Calcular matriz de correlación (solo numéricas)
corr_df = pdf.select_dtypes(include=['number']).corr()

# Visualización
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(10, 8))

mask = np.triu(np.ones_like(corr_df, dtype=bool))

sns.heatmap(
    corr_df,
    annot=True,
    cmap='Blues',
    linewidths=3,
    mask=mask
)

plt.show()

# 3. Calcular la matriz de correlación (Pearson)
corr_df = pdf.select_dtypes(include=['number']).corr()

# 4. Pasar la matriz a formato adecuado (ya está en DataFrame)
pd.set_option('display.max_columns', None)

# 5. Dibujar el heatmap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 8))

mask_heatmap = np.triu(np.ones_like(corr_df, dtype=bool))

sns.heatmap(
    data=corr_df,
    annot=True,
    linewidths=3,
    cmap='Blues',
    mask=mask_heatmap
)

plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC 4.3.4 Conteo de Outliers sobre la columna "time on market"

# COMMAND ----------

from pyspark.sql import functions as F

# Calcular Q1 y Q3 usando approxQuantile (eficiente en Spark distribuido)
q1, q3 = df.approxQuantile("time_on_market", [0.25, 0.75], 0.05)

# Calcular IQR
iqr = q3 - q1

# Definir límites inferior y superior usando la regla clásica
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

# Filtrar los outliers fuera de los límites
outliers = df.filter(
    (F.col("time_on_market") < lower_bound) | 
    (F.col("time_on_market") > upper_bound)
)

# Mostrar los valores calculados
print(f"Q1: {q1}, Q3: {q3}, IQR: {iqr}")
print(f"Lower bound: {lower_bound}, Upper bound: {upper_bound}")

# Mostrar los outliers detectados
display(outliers)

#Conteo de outliers
print(f"Total de outliers detectados: {outliers.count()}")

# COMMAND ----------

# MAGIC %md
# MAGIC 4.3.5 Z-Score

# COMMAND ----------

from pyspark.sql import functions as F

# Calcular la media y la desviación estándar
mean_stddev = df.select(
    F.mean("time_on_market").alias("mean"),
    F.stddev("time_on_market").alias("stddev")
).first()

mean_value = mean_stddev["mean"]
stddev_value = mean_stddev["stddev"]

# Calcular el Z-score
df_with_zscore = df.withColumn(
    "zscore_time_on_market",
    (F.col("time_on_market") - mean_value) / stddev_value
)

# Filtrar los outliers
outliers = df_with_zscore.filter(
    (F.abs(F.col("zscore_time_on_market")) > 3)
)

# Mostrar los outliers
display(outliers)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5 DButils

# COMMAND ----------

# MAGIC %md
# MAGIC 5.1 Obtenermos productos unicos

# COMMAND ----------

# Cargar la tabla Delta registrada en el metastore
df = spark.table("sesion1.data.productos")

# Obtener valores únicos de la columna 'producto'
lista_productos = [row['producto'] for row in df.select("producto").distinct().collect()]

lista_productos

# COMMAND ----------

# MAGIC %md
# MAGIC 5.2 Creamos un widget Dropdown

# COMMAND ----------

dbutils.widgets.dropdown(
    name="producto",
    defaultValue=lista_productos[0],   # First element as default
    choices=lista_productos
)

# COMMAND ----------

# MAGIC %md
# MAGIC 5.3 Creamos un widget text para rango minimo de precios

# COMMAND ----------

dbutils.widgets.text("min_price", "0", "Precio mínimo")

# COMMAND ----------

# MAGIC %md
# MAGIC 5.4 Consulta con paranetros de Widget

# COMMAND ----------

# MAGIC %md
# MAGIC 5.4.1 Con PySpark

# COMMAND ----------

# DBTITLE 1,Cell 95
from pyspark.sql.functions import col, avg

# Get widget values
producto_sel = dbutils.widgets.get("producto")          # dropdown
min_price = float(dbutils.widgets.get("min_price"))     # text

# Load the table
df = spark.table("sesion1.data.productos")

# Apply filters
df_filtrado = (
    df
    .filter(col("producto") == producto_sel)
    .filter(col("precio") >= min_price)
)

# Calculate average price
df_media = df_filtrado.agg(avg("precio").alias("precio_medio"))

display(df_media)

# COMMAND ----------

# MAGIC %md
# MAGIC 5.4.2 Con SQL

# COMMAND ----------

# DBTITLE 1,Cell 97
# MAGIC %sql
# MAGIC SELECT 
# MAGIC   AVG(precio) AS precio_medio
# MAGIC FROM 
# MAGIC   sesion1.data.productos
# MAGIC WHERE 
# MAGIC   producto = :producto
# MAGIC   AND precio >= CAST(:min_price AS DOUBLE)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 6 Creacion de Dashboard con SQL

# COMMAND ----------

# MAGIC %md
# MAGIC 6.1 Visualizaciones rápidas

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM sesion1.data.productos;

# COMMAND ----------

# MAGIC %md
# MAGIC **6.2 Creacion de un Dashboard**

# COMMAND ----------

# MAGIC %sql
# MAGIC -- 1. Crear la tabla con nombres y tipos de columnas
# MAGIC CREATE OR REPLACE TABLE sesion1.data.dim_producto (
# MAGIC     product_id       INT,
# MAGIC     nombre_producto  STRING,
# MAGIC     categoria        STRING,
# MAGIC     tipo_dispositivo STRING,
# MAGIC     es_periferico    BOOLEAN,
# MAGIC     es_entrada       BOOLEAN,
# MAGIC     es_salida        BOOLEAN
# MAGIC )
# MAGIC USING DELTA;
# MAGIC
# MAGIC -- 2. Insertar los datos
# MAGIC INSERT INTO sesion1.data.dim_producto VALUES
# MAGIC     (1,  'Mouse',        'Accesorios',     'Entrada',         TRUE,  TRUE,  FALSE),
# MAGIC     (2,  'Teclado',      'Accesorios',     'Entrada',         TRUE,  TRUE,  FALSE),
# MAGIC     (3,  'Webcam',       'Multimedia',     'Entrada/Salida',  TRUE,  TRUE,  TRUE),
# MAGIC     (4,  'Microfono',    'Multimedia',     'Entrada',         TRUE,  TRUE,  FALSE),
# MAGIC     (5,  'Altavoces',    'Multimedia',     'Salida',          TRUE,  FALSE, TRUE),
# MAGIC     (6,  'Disco Duro',   'Almacenamiento', 'Almacenamiento',  FALSE, FALSE, FALSE),
# MAGIC     (7,  'Auriculares',  'Multimedia',     'Entrada/Salida',  TRUE,  TRUE,  TRUE),
# MAGIC     (8,  'USB',          'Accesorios',     'Almacenamiento',  FALSE, FALSE, FALSE),
# MAGIC     (9,  'Impresora',    'Oficina',        'Salida',          TRUE,  FALSE, TRUE),
# MAGIC     (10, 'Monitor',      'Oficina',        'Salida',          TRUE,  FALSE, TRUE);

# COMMAND ----------

# MAGIC %md
# MAGIC **6.2.1 Creacion de un Dashboard mediante consulta SQL**

# COMMAND ----------

# MAGIC %md
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC
