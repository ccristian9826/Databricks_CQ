# Databricks notebook source
# ============================================================
# PASO 1: Verificación de training_images en Silver
# ============================================================

from pyspark.sql import functions as F
from pyspark.sql import SparkSession

CATALOG       = "proyecto_smart_claims"
SILVER_SCHEMA = "silver"

spark = SparkSession.builder.getOrCreate()

# ------------------------------------------------------------
# Cargar la tabla
# ------------------------------------------------------------
training_images = spark.table(f"{CATALOG}.{SILVER_SCHEMA}.training_images")

# ------------------------------------------------------------
# 1. Verificar que la tabla existe y tiene filas
# ------------------------------------------------------------
print("=" * 60)
print("1. EXISTENCIA Y TAMAÑO DE LA TABLA")
print("=" * 60)

total_filas = training_images.count()
print(f"Total de imágenes registradas: {total_filas:,}")

if total_filas == 0:
    raise ValueError("❌ La tabla training_images está vacía. Revisa la capa Silver antes de continuar.")
else:
    print("✅ La tabla tiene registros.")

# ------------------------------------------------------------
# 2. Verificar que las columnas esperadas existen
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("2. COLUMNAS DISPONIBLES EN LA TABLA")
print("=" * 60)

columnas_esperadas = ["path", "image_name", "label", "content"]
columnas_presentes = training_images.columns

print(f"Columnas en la tabla: {columnas_presentes}")

faltantes = [c for c in columnas_esperadas if c not in columnas_presentes]

if faltantes:
    print(f"\n⚠️  Columnas faltantes: {faltantes}")
    print("   Verifica que la capa Silver incluyó todas las columnas necesarias.")
else:
    print("\n✅ Todas las columnas esperadas están presentes.")

# ------------------------------------------------------------
# 3. Verificar que la columna 'content' existe y tiene datos
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("3. VALIDACIÓN DE COLUMNA BINARIA (content)")
print("=" * 60)

if "content" in columnas_presentes:
    nulos_content = training_images.filter(F.col("content").isNull()).count()
    print(f"Imágenes con content nulo:     {nulos_content:,}")
    print(f"Imágenes con content presente: {total_filas - nulos_content:,}")

    if nulos_content > 0:
        print("\n⚠️  Hay imágenes sin contenido binario. Muestra de paths afectados:")
        training_images.filter(F.col("content").isNull()) \
            .select("path", "image_name", "label") \
            .show(10, truncate=False)
    else:
        print("✅ Todas las imágenes tienen contenido binario.")
else:
    print("⚠️  La columna 'content' no existe en la tabla.")
    print("   Las imágenes fueron registradas solo por path, sin contenido binario cargado.")

# ------------------------------------------------------------
# 4. Verificar la columna 'label'
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("4. VALIDACIÓN DE LABELS")
print("=" * 60)

nulos_label  = training_images.filter(F.col("label").isNull() | (F.col("label") == "")).count()
print(f"Imágenes con label nulo o vacío: {nulos_label:,}")

if nulos_label > 0:
    print("\n❌ Hay imágenes sin label. Muestra de casos problemáticos:")
    training_images.filter(F.col("label").isNull() | (F.col("label") == "")) \
        .select("path", "image_name", "label") \
        .show(10, truncate=False)
    raise ValueError("❌ Existen labels vacíos. Corrige la capa Silver antes de continuar.")
else:
    print("✅ Todos los registros tienen label.")

# ------------------------------------------------------------
# 5. Distribución de labels
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("5. DISTRIBUCIÓN DE LABELS")
print("=" * 60)

distribucion = (
    training_images
    .groupBy("label")
    .agg(
        F.count("*").alias("cantidad"),
        F.round(F.count("*") / total_filas * 100, 2).alias("porcentaje_%")
    )
    .orderBy("label")
)

distribucion.show(truncate=False)

# Detectar desbalance severo (alguna clase con menos del 5%)
minimo_pct = distribucion.agg(F.min("porcentaje_%")).collect()[0][0]
if minimo_pct < 5.0:
    print(f"⚠️  Hay clases con menos del 5% de representación ({minimo_pct}%).")
    print("   Considera técnicas de balanceo antes de entrenar.")
else:
    print("✅ Distribución de clases balanceada.")

# ------------------------------------------------------------
# 6. Verificar nombres de archivo y paths
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("6. VALIDACIÓN DE PATHS E IMAGE_NAME")
print("=" * 60)

nulos_path       = training_images.filter(F.col("path").isNull()).count()
nulos_image_name = training_images.filter(
    F.col("image_name").isNull() | (F.col("image_name") == "")
).count()

print(f"Paths nulos:       {nulos_path:,}")
print(f"image_name nulos:  {nulos_image_name:,}")

if nulos_path == 0 and nulos_image_name == 0:
    print("✅ Todos los registros tienen path e image_name válidos.")
else:
    print("⚠️  Hay registros con path o image_name inválidos. Revisa la carga en Bronze.")

# Verificar extensiones esperadas
print("\n>>> Extensiones de archivo encontradas:")
training_images.withColumn(
    "extension",
    F.lower(F.regexp_extract(F.col("image_name"), r"\.(\w+)$", 1))
).groupBy("extension").count().orderBy("extension").show()

# ------------------------------------------------------------
# 7. Muestra final
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("7. MUESTRA DE REGISTROS")
print("=" * 60)

cols_muestra = [c for c in ["path", "image_name", "label"] if c in columnas_presentes]
training_images.select(*cols_muestra).show(10, truncate=False)

# ------------------------------------------------------------
# VEREDICTO FINAL
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("VEREDICTO FINAL")
print("=" * 60)

if total_filas > 0 and nulos_label == 0 and nulos_path == 0 and nulos_image_name == 0:
    print("✅ La tabla training_images está lista para entrenar.")
else:
    print("❌ La tabla NO está lista. Revisa los puntos marcados con ⚠️ o ❌ antes de continuar.")

# COMMAND ----------

# DBTITLE 1,Cell 2
# ============================================================
# PASO 2: Instalación e importación de librerías
# ============================================================

# ------------------------------------------------------------
# Instalación de librerías
# ------------------------------------------------------------
%pip install transformers datasets Pillow mlflow torch torchvision accelerate --quiet

# Reiniciar el kernel después de instalar para que los paquetes
# queden disponibles correctamente en el entorno de Databricks
dbutils.library.restartPython()

# COMMAND ----------

# ------------------------------------------------------------
# Imports — correr en celda separada después del restart
# ------------------------------------------------------------

# Procesamiento de imágenes
from PIL import Image
import io
import numpy as np

# Datasets y Transformers (HuggingFace)
from datasets import Dataset
from transformers import (
    AutoFeatureExtractor,
    AutoModelForImageClassification,
    TrainingArguments,
    Trainer,
)

# PyTorch
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

# MLflow
import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient

# PySpark
from pyspark.sql import functions as F
from pyspark.sql import SparkSession

# Utilitarios
import os
import random

spark = SparkSession.builder.getOrCreate()

# ------------------------------------------------------------
# Verificación del entorno
# ------------------------------------------------------------
print("=" * 60)
print("VERIFICACIÓN DEL ENTORNO")
print("=" * 60)

# Versiones
import transformers
import datasets
import PIL

print(f"Python:          {__import__('sys').version.split()[0]}")
print(f"PyTorch:         {torch.__version__}")
print(f"Transformers:    {transformers.__version__}")
print(f"Datasets:        {datasets.__version__}")
print(f"Pillow:          {PIL.__version__}")
print(f"MLflow:          {mlflow.__version__}")
print(f"NumPy:           {np.__version__}")

# GPU disponible
print("\n" + "=" * 60)
print("DISPONIBILIDAD DE GPU")
print("=" * 60)

if torch.cuda.is_available():
    print(f"✅ GPU disponible: {torch.cuda.get_device_name(0)}")
    print(f"   Memoria total:  {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    DEVICE = "cuda"
else:
    print("⚠️  No hay GPU disponible. El entrenamiento usará CPU (más lento).")
    DEVICE = "cpu"

print(f"\nDispositivo activo: {DEVICE}")

# MLflow activo
print("\n" + "=" * 60)
print("MLFLOW")
print("=" * 60)

print(f"Tracking URI: {mlflow.get_tracking_uri()}")
print("✅ MLflow listo.")

print("\n✅ Entorno listo para entrenar.")

# COMMAND ----------

# DBTITLE 1,Cell 4
# ============================================================
# PASO 3: Definir el experimento en MLflow
# ============================================================

import mlflow
from mlflow.tracking import MlflowClient

# ------------------------------------------------------------
# Nombre del experimento
# El experimento agrupa todas las corridas del entrenamiento.
# Si no existe, MLflow lo crea automáticamente.
# Si ya existe, MLflow lo reutiliza y agrega nuevas corridas.
# ------------------------------------------------------------

EXPERIMENT_NAME = "/Users/sebastiandavilahenao@gmail.com/proyecto_smart_claims/clasificacion_danios"

mlflow.set_experiment(EXPERIMENT_NAME)

# ------------------------------------------------------------
# Verificar que el experimento quedó registrado
# ------------------------------------------------------------
client    = MlflowClient()
experimento = client.get_experiment_by_name(EXPERIMENT_NAME)

print("=" * 60)
print("EXPERIMENTO MLFLOW")
print("=" * 60)
print(f"Nombre:      {experimento.name}")
print(f"ID:          {experimento.experiment_id}")
print(f"Estado:      {'✅ Activo' if experimento.lifecycle_stage == 'active' else '⚠️ ' + experimento.lifecycle_stage}")
print(f"Ubicación:   {experimento.artifact_location}")

print("""
============================================================
¿QUÉ ES UN EXPERIMENTO EN MLFLOW?
============================================================

  EXPERIMENTO: clasificacion_danios
  │
  ├── Corrida 1 → primer entrenamiento (ej: lr=2e-5, epochs=3)
  │     ├── Parámetros: learning_rate, num_epochs, batch_size...
  │     ├── Métricas:   accuracy, loss, eval_accuracy...
  │     ├── Artefactos: confusion matrix, gráficas...
  │     └── Modelo:     checkpoint guardado
  │
  ├── Corrida 2 → segundo entrenamiento (ej: lr=5e-5, epochs=5)
  │     └── ...
  │
  └── Corrida N → cada vez que se reentrene queda registrado aquí

  → Todas las corridas quedan trazables y comparables.
  → El mejor modelo se puede promover a producción desde MLflow.
============================================================
""")

print("✅ Experimento listo. Todas las corridas del entrenamiento")
print(f"   quedarán guardadas en: {EXPERIMENT_NAME}")

# COMMAND ----------

# ============================================================
# PASO 4: Preparar las imágenes para el entrenamiento
# ============================================================

from PIL import Image
import io
import numpy as np
from pyspark.sql import functions as F
from pyspark.sql.types import BinaryType
import mlflow

CATALOG       = "proyecto_smart_claims"
SILVER_SCHEMA = "silver"
GOLD_SCHEMA   = "gold"

# Tamaño estándar que esperan los modelos de visión tipo ViT
IMAGE_SIZE = (224, 224)

# ------------------------------------------------------------
# Cargar la tabla Silver de imágenes de entrenamiento
# ------------------------------------------------------------
print("=" * 60)
print("CARGANDO IMÁGENES DESDE SILVER")
print("=" * 60)

training_images = spark.table(f"{CATALOG}.{SILVER_SCHEMA}.training_images")

print(f"Total imágenes: {training_images.count():,}")
print(f"Columnas disponibles: {training_images.columns}")

# ------------------------------------------------------------
# Función de redimensionamiento
# Recibe bytes de imagen y devuelve bytes redimensionados
# No cambia el significado de la imagen, solo su tamaño
# ------------------------------------------------------------

def redimensionar_imagen(imagen_bytes: bytes, size: tuple = IMAGE_SIZE) -> bytes:
    """
    Abre la imagen desde bytes, la redimensiona al tamaño
    estándar requerido por el modelo y la devuelve como bytes.
    Convierte RGB si es necesario.
    """
    img = Image.open(io.BytesIO(imagen_bytes))
    
    # Convertir a RGB si la imagen está en otro modo (ej: RGBA, L)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Redimensionar
    img_resized = img.resize(size, Image.Resampling.LANCZOS)
    
    # Convertir de vuelta a bytes
    buffer = io.BytesIO()
    img_resized.save(buffer, format='JPEG')
    return buffer.getvalue()

# COMMAND ----------

# ============================================================
# PASO 5: Construir el dataset de entrenamiento y validación
# ============================================================

from PIL import Image
from datasets import Dataset
import io
import numpy as np
import mlflow

CATALOG       = "proyecto_smart_claims"
SILVER_SCHEMA = "silver"

# Proporción de datos para validación (20%)
VAL_SIZE = 0.2
SEED     = 42

# ------------------------------------------------------------
# Cargar la tabla Silver con imágenes de entrenamiento
# ------------------------------------------------------------
print("=" * 60)
print("CARGANDO TABLA silver.training_images")
print("=" * 60)

training_ready = spark.table(f"{CATALOG}.{SILVER_SCHEMA}.training_images")

total = training_ready.count()
print(f"Total imágenes disponibles: {total:,}")
training_ready.groupBy("label").count().orderBy("label").show()

# ------------------------------------------------------------
# Diccionario de clases
# Relaciona cada label (texto) con un identificador numérico
# El modelo no trabaja con texto sino con números
# ------------------------------------------------------------
print("=" * 60)
print("DICCIONARIO DE CLASES")
print("=" * 60)

labels_distintos = sorted([
    row["label"] for row in training_ready.select("label").distinct().collect()
])

# label → número (para entrenar)
label2id = {label: idx for idx, label in enumerate(labels_distintos)}

# número → label (para interpretar predicciones)
id2label = {idx: label for label, idx in label2id.items()}

print("Relación label → id numérico:")
for label, idx in label2id.items():
    print(f"  '{label}' → {idx}")

print(f"\nTotal de clases: {len(label2id)}")

# ------------------------------------------------------------
# Convertir la tabla Spark a lista de Python
# para construir el dataset de HuggingFace
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("CONVIRTIENDO TABLA A DATASET")
print("=" * 60)

filas = training_ready.select("image_name", "label", "content").collect()

def construir_registro(fila):
    """
    Convierte una fila de Spark en un diccionario con:
    - image:    objeto PIL listo para el modelo
    - label:    identificador numérico de la clase
    - label_str: nombre original del label (para referencia)
    """
    img = Image.open(io.BytesIO(fila["content"])).convert("RGB")
    return {
        "image":     img,
        "label":     label2id[fila["label"]],
        "label_str": fila["label"],
    }

registros = [construir_registro(f) for f in filas]
print(f"✅ {len(registros):,} registros construidos.")

# ------------------------------------------------------------
# Crear el dataset de HuggingFace y dividir
# ------------------------------------------------------------
print("\n" + "=" * 60)
print(f"DIVIDIENDO: {int((1-VAL_SIZE)*100)}% entrenamiento / {int(VAL_SIZE*100)}% validación")
print("=" * 60)

dataset_completo = Dataset.from_list(registros)

split = dataset_completo.train_test_split(
    test_size  = VAL_SIZE,
    seed       = SEED
)

dataset_train = split["train"]
dataset_val   = split["test"]

print(f"Imágenes de entrenamiento: {len(dataset_train):,}")
print(f"Imágenes de validación:    {len(dataset_val):,}")

# Verificar distribución en cada split
print("\nDistribución en entrenamiento:")
train_labels = dataset_train["label_str"]
for label in labels_distintos:
    n = train_labels.count(label)
    print(f"  {label}: {n:,} ({n/len(dataset_train)*100:.1f}%)")

print("\nDistribución en validación:")
val_labels = dataset_val["label_str"]
for label in labels_distintos:
    n = val_labels.count(label)
    print(f"  {label}: {n:,} ({n/len(dataset_val)*100:.1f}%)")

# ------------------------------------------------------------
# Registrar en MLflow
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("REGISTRANDO EN MLFLOW")
print("=" * 60)

with mlflow.start_run(run_name="preparacion_dataset", nested=True):
    mlflow.log_param("total_imagenes",       total)
    mlflow.log_param("val_size",             VAL_SIZE)
    mlflow.log_param("seed",                 SEED)
    mlflow.log_param("num_clases",           len(label2id))
    mlflow.log_param("clases",               str(labels_distintos))
    mlflow.log_metric("train_size",          len(dataset_train))
    mlflow.log_metric("val_size_real",       len(dataset_val))

print("✅ Parámetros del dataset registrados en MLflow.")

# ------------------------------------------------------------
# Resumen final
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("RESUMEN PASO 5")
print("=" * 60)
print(f"  Dataset entrenamiento:  {len(dataset_train):,} imágenes")
print(f"  Dataset validación:     {len(dataset_val):,} imágenes")
print(f"  Clases:                 {labels_distintos}")
print(f"  label2id:               {label2id}")
print(f"  id2label:               {id2label}")
print("\n✅ Paso 5 completo. dataset_train y dataset_val listos para el modelo.")

# COMMAND ----------

# ============================================================
# PASO 6: Convertir el binario a imagen
# ============================================================

from PIL import Image
from transformers import AutoImageProcessor
import io
import numpy as np

# ------------------------------------------------------------
# Modelo base que se usará para entrenar
# ViT (Vision Transformer) preentrenado en ImageNet
# El feature extractor sabe exactamente qué formato necesita
# ------------------------------------------------------------

MODEL_CHECKPOINT = "google/vit-base-patch16-224"

print("=" * 60)
print("CARGANDO IMAGE PROCESSOR")
print("=" * 60)

image_processor = AutoImageProcessor.from_pretrained(MODEL_CHECKPOINT)

print(f"Modelo base:       {MODEL_CHECKPOINT}")
print(f"Tamaño esperado:   {image_processor.size}")
print(f"Media (mean):      {image_processor.image_mean}")
print(f"Desviación (std):  {image_processor.image_std}")
print("✅ Image processor cargado.")

# ------------------------------------------------------------
# ¿Qué hace el image processor?
#
# El binario JPEG no puede entrar directamente al modelo.
# El image processor hace tres cosas:
#
#   1. Abre la imagen PIL y la convierte a array numérico
#   2. Normaliza los valores de píxel (0-255 → rango estándar)
#      usando la media y desviación del preentrenamiento
#   3. Devuelve tensores con la forma exacta que el modelo espera:
#      [canales, altura, anchura] → [3, 224, 224]
# ------------------------------------------------------------

# ------------------------------------------------------------
# Función de transformación
# Recibe un batch del dataset y devuelve los pixel_values
# que el modelo puede leer directamente
# ------------------------------------------------------------

def transformar_imagenes(batch):
    """
    Convierte el contenido binario de cada imagen en tensores
    normalizados listos para el modelo ViT.

    El dataset de HuggingFace llama esta función en batches,
    por eso recibe una lista de imágenes y no una sola.
    """
    imagenes_pil = []

    for item in batch["image"]:
        # Si ya es un objeto PIL (viene del paso anterior), usarlo directo
        if isinstance(item, Image.Image):
            img = item.convert("RGB")
        # Si todavía es binario, convertirlo a PIL primero
        elif isinstance(item, (bytes, bytearray)):
            img = Image.open(io.BytesIO(item)).convert("RGB")
        else:
            raise ValueError(f"Tipo de imagen no reconocido: {type(item)}")

        imagenes_pil.append(img)

    # El image processor normaliza y convierte a tensor
    encoded = image_processor(images=imagenes_pil, return_tensors="pt")

    # Agregar los pixel_values al batch
    batch["pixel_values"] = encoded["pixel_values"]

    return batch

# ------------------------------------------------------------
# Aplicar la transformación sobre train y validación
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("APLICANDO TRANSFORMACIÓN A LOS DATASETS")
print("=" * 60)

dataset_train_encoded = dataset_train.map(
    transformar_imagenes,
    batched      = True,
    batch_size   = 16,
    desc         = "Transformando entrenamiento",
)

dataset_val_encoded = dataset_val.map(
    transformar_imagenes,
    batched      = True,
    batch_size   = 16,
    desc         = "Transformando validación",
)

print(f"✅ Entrenamiento transformado: {len(dataset_train_encoded):,} imágenes")
print(f"✅ Validación transformada:    {len(dataset_val_encoded):,} imágenes")

# ------------------------------------------------------------
# Configurar formato para PyTorch
# Solo las columnas que el modelo necesita
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("CONFIGURANDO FORMATO PYTORCH")
print("=" * 60)

dataset_train_encoded.set_format(
    type    = "torch",
    columns = ["pixel_values", "label"]
)

dataset_val_encoded.set_format(
    type    = "torch",
    columns = ["pixel_values", "label"]
)

print("Columnas activas para el modelo: ['pixel_values', 'label']")
print("✅ Formato PyTorch configurado.")

# ------------------------------------------------------------
# Verificación visual: inspeccionar un ejemplo transformado
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("VERIFICACIÓN: INSPECCIONANDO UN EJEMPLO")
print("=" * 60)

ejemplo = dataset_train_encoded[0]

print(f"Claves disponibles:     {list(ejemplo.keys())}")
print(f"Forma de pixel_values:  {ejemplo['pixel_values'].shape}")
print(f"Tipo de tensor:         {ejemplo['pixel_values'].dtype}")
print(f"Label numérico:         {ejemplo['label'].item()} → '{id2label[ejemplo['label'].item()]}'")
print(f"Valor mínimo de píxel:  {ejemplo['pixel_values'].min():.4f}")
print(f"Valor máximo de píxel:  {ejemplo['pixel_values'].max():.4f}")

# Verificar que la normalización es correcta
# Los valores deben estar en rango aproximado [-3, 3] tras normalizar
if ejemplo["pixel_values"].min() >= -4 and ejemplo["pixel_values"].max() <= 4:
    print("\n✅ Normalización correcta. Valores en rango esperado.")
else:
    print("\n⚠️  Los valores de píxel están fuera del rango esperado.")
    print("   Verifica que el image processor corresponde al modelo.")

print("\n✅ Paso 6 completo.")
print("   dataset_train_encoded y dataset_val_encoded listos para el Trainer.")

# COMMAND ----------

# ============================================================
# PASO 7: Seleccionar y cargar el modelo base
# ============================================================

from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
)
import torch
import mlflow

# ------------------------------------------------------------
# ¿Por qué no entrenamos desde cero?
#
# Un modelo entrenado desde cero necesita millones de imágenes
# y semanas de cómputo para aprender a reconocer bordes,
# texturas, formas y objetos.
#
# ViT (Vision Transformer) ya fue entrenado con ImageNet
# (14 millones de imágenes, 1000 clases). Ya sabe reconocer
# patrones visuales generales.
#
# Lo que hacemos aquí se llama Fine-Tuning:
#   → Tomamos ese conocimiento general
#   → Reemplazamos solo la capa final de clasificación
#   → La adaptamos a nuestras clases de daños de vehículos
#   → Entrenamos pocas épocas para ajustar al nuevo problema
# ------------------------------------------------------------

MODEL_CHECKPOINT = "google/vit-base-patch16-224"

print("=" * 60)
print("MODELO BASE SELECCIONADO")
print("=" * 60)
print(f"Checkpoint:  {MODEL_CHECKPOINT}")
print(f"Arquitectura: Vision Transformer (ViT)")
print(f"Preentrenado en: ImageNet-21k")
print(f"Clases originales del modelo: 1000")
print(f"Clases de nuestro problema:   {len(label2id)}")

# ------------------------------------------------------------
# Cargar el Image Processor
# Es el procesador que prepara las imágenes para el modelo.
# Ya fue cargado en el Paso 6, pero se define aquí también
# para que este bloque sea independiente y claro.
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("CARGANDO IMAGE PROCESSOR")
print("=" * 60)

image_processor = AutoImageProcessor.from_pretrained(MODEL_CHECKPOINT)

print(f"✅ Image processor cargado.")
print(f"   Tamaño de imagen esperado: {image_processor.size}")
print(f"   Media de normalización:    {image_processor.image_mean}")
print(f"   Desviación estándar:       {image_processor.image_std}")

# ------------------------------------------------------------
# Cargar el modelo de clasificación
#
# ignore_mismatched_sizes=True le dice al modelo que está bien
# que la capa final tenga un tamaño distinto al original
# (1000 clases → nuestras clases), sin lanzar error.
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("CARGANDO MODELO DE CLASIFICACIÓN")
print("=" * 60)

model = AutoModelForImageClassification.from_pretrained(
    MODEL_CHECKPOINT,
    num_labels            = len(label2id),   # número de clases del problema
    id2label              = id2label,         # número → nombre de clase
    label2id              = label2id,         # nombre → número
    ignore_mismatched_sizes = True            # permite reemplazar la capa final
)

# Mover el modelo al dispositivo disponible (GPU o CPU)
model = model.to(DEVICE)

print(f"✅ Modelo cargado correctamente.")
print(f"   Dispositivo: {DEVICE}")
print(f"   Clases configuradas:")
for idx, label in id2label.items():
    print(f"     {idx} → '{label}'")

# ------------------------------------------------------------
# Inspeccionar la arquitectura del modelo
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("ARQUITECTURA DEL MODELO")
print("=" * 60)

# Parámetros totales
total_params = sum(p.numel() for p in model.parameters())

# Parámetros entrenables (los que se van a ajustar)
params_entrenables = sum(p.numel() for p in model.parameters() if p.requires_grad)

# Parámetros congelados (el conocimiento general que se conserva)
params_congelados = total_params - params_entrenables

print(f"Parámetros totales:      {total_params:,}")
print(f"Parámetros entrenables:  {params_entrenables:,}")
print(f"Parámetros congelados:   {params_congelados:,}")

# Mostrar solo la capa de clasificación (la que se reemplazó)
print(f"\nCapa de clasificación reemplazada:")
print(f"  {model.classifier}")

# ------------------------------------------------------------
# Verificar que el modelo hace una predicción de prueba
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("VERIFICACIÓN: PREDICCIÓN DE PRUEBA")
print("=" * 60)

model.eval()
with torch.no_grad():
    ejemplo        = dataset_train_encoded[0]
    pixel_values   = ejemplo["pixel_values"].unsqueeze(0).to(DEVICE)
    outputs        = model(pixel_values=pixel_values)
    logits         = outputs.logits
    pred_idx       = logits.argmax(dim=-1).item()
    pred_label     = id2label[pred_idx]
    label_real     = id2label[ejemplo["label"].item()]

print(f"Label real:      '{label_real}'")
print(f"Predicción:      '{pred_label}'  (antes de entrenar, puede ser incorrecta)")
print(f"Logits shape:    {logits.shape}  → un valor por cada clase")
print("\n⚠️  La predicción antes de entrenar es aleatoria. Esto es normal.")
print("   El Fine-Tuning del siguiente paso corregirá esto.")

# ------------------------------------------------------------
# Registrar en MLflow
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("REGISTRANDO EN MLFLOW")
print("=" * 60)

with mlflow.start_run(run_name="seleccion_modelo", nested=True):
    mlflow.log_param("model_checkpoint",     MODEL_CHECKPOINT)
    mlflow.log_param("arquitectura",         "ViT-base-patch16-224")
    mlflow.log_param("num_clases",           len(label2id))
    mlflow.log_param("dispositivo",          DEVICE)
    mlflow.log_param("total_params",         total_params)
    mlflow.log_param("params_entrenables",   params_entrenables)

print("✅ Parámetros del modelo registrados en MLflow.")

print("\n✅ Paso 7 completo.")
print(f"   Modelo '{MODEL_CHECKPOINT}' listo para Fine-Tuning")
print(f"   con {len(label2id)} clases: {list(label2id.keys())}")

# COMMAND ----------

# ============================================================
# PASO 8: Preprocesar las imágenes para el modelo
# ============================================================

from transformers import AutoImageProcessor
from PIL import Image
import torch
import numpy as np
import io

# Cargar el image processor
MODEL_CHECKPOINT = "google/vit-base-patch16-224"
image_processor = AutoImageProcessor.from_pretrained(MODEL_CHECKPOINT)

# ------------------------------------------------------------
# ¿Qué diferencia hay entre el Paso 6 y el Paso 8?
#
# Paso 6 → convirtió el binario a imagen PIL (abrir el archivo)
# Paso 8 → convierte la imagen PIL a tensores numéricos
#           normalizados con los valores exactos del modelo
#
# Es el último paso de preparación antes del entrenamiento.
# El Trainer llamará esta función automáticamente en cada batch.
# ------------------------------------------------------------

print("=" * 60)
print("FUNCIÓN DE PREPROCESAMIENTO")
print("=" * 60)

# ------------------------------------------------------------
# Función de preprocesamiento para el Trainer
#
# El Trainer de HuggingFace espera una función que reciba
# un batch del dataset y devuelva los campos procesados.
# Esta función se llama collate_fn o preprocessing_function.
# ------------------------------------------------------------

def preprocesar_batch(batch):
    """
    Recibe un batch con imágenes PIL y labels numéricos.
    Devuelve pixel_values normalizados listos para el modelo.

    El image_processor aplica:
      1. Redimensionamiento al tamaño esperado (224x224)
      2. Normalización por canal usando media y std de ImageNet
      3. Conversión a tensor PyTorch con forma [3, 224, 224]
    """
    imagenes = []

    for img in batch["image"]:
        # Aceptar PIL Image o binario por robustez
        if isinstance(img, Image.Image):
            imagenes.append(img.convert("RGB"))
        elif isinstance(img, (bytes, bytearray)):
            imagenes.append(
                Image.open(io.BytesIO(img)).convert("RGB")
            )
        else:
            raise ValueError(f"Formato no soportado: {type(img)}")

    # Image processor normaliza y convierte a tensor
    encoded = image_processor(images=imagenes, return_tensors="pt")
    batch["pixel_values"] = encoded["pixel_values"]

    return batch


# ------------------------------------------------------------
# Collate function
#
# El Trainer arma mini-batches durante el entrenamiento.
# Esta función le dice cómo agrupar múltiples ejemplos
# en un solo tensor listo para pasar al modelo.
# ------------------------------------------------------------

def collate_fn(ejemplos):
    """
    Agrupa una lista de ejemplos individuales en un batch.
    Apila los pixel_values y los labels en tensores únicos.
    """
    pixel_values = torch.stack([e["pixel_values"] for e in ejemplos])
    labels       = torch.tensor([e["label"] for e in ejemplos])

    return {
        "pixel_values": pixel_values,
        "labels":       labels,
    }

print("✅ preprocesar_batch definida.")
print("✅ collate_fn definida.")

# ------------------------------------------------------------
# Aplicar preprocesamiento sobre train y validación
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("APLICANDO PREPROCESAMIENTO")
print("=" * 60)

dataset_train_processed = dataset_train.map(
    preprocesar_batch,
    batched    = True,
    batch_size = 16,
    desc       = "Preprocesando entrenamiento",
)

dataset_val_processed = dataset_val.map(
    preprocesar_batch,
    batched    = True,
    batch_size = 16,
    desc       = "Preprocesando validación",
)

# Fijar formato PyTorch
dataset_train_processed.set_format(
    type    = "torch",
    columns = ["pixel_values", "label"]
)

dataset_val_processed.set_format(
    type    = "torch",
    columns = ["pixel_values", "label"]
)

print(f"✅ Entrenamiento preprocesado: {len(dataset_train_processed):,} imágenes")
print(f"✅ Validación preprocesada:    {len(dataset_val_processed):,} imágenes")

# ------------------------------------------------------------
# Verificación detallada de un ejemplo
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("VERIFICACIÓN DE UN EJEMPLO PREPROCESADO")
print("=" * 60)

ejemplo = dataset_train_processed[0]

print(f"Claves del ejemplo:      {list(ejemplo.keys())}")
print(f"Forma pixel_values:      {ejemplo['pixel_values'].shape}")
print(f"Tipo de dato:            {ejemplo['pixel_values'].dtype}")
print(f"Label:                   {ejemplo['label'].item()} → '{id2label[ejemplo['label'].item()]}'")
print(f"Valor mínimo de píxel:   {ejemplo['pixel_values'].min():.4f}")
print(f"Valor máximo de píxel:   {ejemplo['pixel_values'].max():.4f}")
print(f"Media de píxel:          {ejemplo['pixel_values'].mean():.4f}")

# ------------------------------------------------------------
# Verificación del collate_fn con un mini-batch de prueba
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("VERIFICACIÓN DEL COLLATE_FN")
print("=" * 60)

mini_batch  = [dataset_train_processed[i] for i in range(4)]
batch_listo = collate_fn(mini_batch)

print(f"pixel_values shape: {batch_listo['pixel_values'].shape}")
print(f"  → [batch_size=4, canales=3, alto=224, ancho=224]")
print(f"labels shape:       {batch_listo['labels'].shape}")
print(f"labels values:      {batch_listo['labels'].tolist()}")
print(f"  → {[id2label[l] for l in batch_listo['labels'].tolist()]}")
print(f"\n✅ El collate_fn arma batches correctamente.")

# ------------------------------------------------------------
# Resumen de lo que entra y sale
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("RESUMEN: QUÉ ENTRA Y QUÉ SALE")
print("=" * 60)
print("""
  ENTRADA (imagen PIL)
    → Objeto Image RGB de 224x224 px
    → Valores de píxel entre 0 y 255

  PROCESAMIENTO (image_processor)
    → Normalización canal a canal con media y std de ImageNet
    → Conversión a tensor float32

  SALIDA (pixel_values)
    → Tensor de forma [3, 224, 224]
    → Valores en rango aproximado [-3, 3]
    → Listo para entrar directamente al modelo ViT
""")

print("✅ Paso 8 completo.")
print("   dataset_train_processed, dataset_val_processed")
print("   y collate_fn listos para el Trainer del Paso 9.")

# COMMAND ----------

# DBTITLE 1,Cell 10
# ============================================================
# PASO 9: Configurar los parámetros de entrenamiento
# ============================================================

from transformers import TrainingArguments
import numpy as np
import mlflow
import torch


# COMMAND ----------

# DBTITLE 1,Untitled

# ------------------------------------------------------------
# ¿Qué es cada parámetro?
#
# ÉPOCA (epoch): una pasada completa por todos los datos
#   → con pocas imágenes, 5-10 épocas suele ser suficiente
#
# BATCH SIZE: cuántas imágenes se procesan a la vez
#   → más grande = más memoria GPU, más estable el gradiente
#   → más pequeño = menos memoria, más ruido en el aprendizaje
#
# LEARNING RATE: qué tan grandes son los ajustes del modelo
#   → muy alto = el modelo no converge, salta sin aprender
#   → muy bajo = el modelo aprende muy lento
#   → 2e-5 es un valor estándar para Fine-Tuning con ViT
#
# WARMUP: épocas iniciales con lr muy bajo antes de subir
#   → evita que el modelo rompa los pesos preentrenados
#   → al principio el modelo es sensible, necesita ajustes suaves
# ------------------------------------------------------------

import os

# Limpiar variables de entorno de distributed training ANTES de crear TrainingArguments
for var in ["RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT", "LOCAL_RANK"]:
    os.environ.pop(var, None)

OUTPUT_DIR = "/tmp/smart_claims_vit"

print("=" * 60)
print("CONFIGURACIÓN DE ENTRENAMIENTO - MODO AGRESIVO")
print("=" * 60)
print("Se aplicarán parámetros optimizados para dataset pequeño:")
print("  • Épocas: 20 (más iteraciones de aprendizaje)")
print("  • Batch size: 4 (4x más actualizaciones de gradiente)")
print("  • Warmup steps: 10 (inicio gradual del learning rate)")
print("  • Learning rate: 5e-5 (ligeramente más alto)")
print("\n⏱️  Tiempo estimado: 20-30 minutos")
print("=" * 60)

training_args = TrainingArguments(
    output_dir = OUTPUT_DIR,

    # ── Épocas: aumentado a 20 para más aprendizaje ──────────
    num_train_epochs = 20,

    # ── Batch size: reducido a 4 para más actualizaciones ────
    per_device_train_batch_size = 4,
    per_device_eval_batch_size  = 4,

    # ── Learning rate: ligeramente más alto ──────────────────
    learning_rate          = 5e-5,
    lr_scheduler_type      = "cosine",
    warmup_steps           = 10,  # 10 pasos de warmup gradual

    eval_strategy          = "epoch",
    save_strategy          = "epoch",
    load_best_model_at_end = True,
    metric_for_best_model  = "accuracy",
    greater_is_better      = True,

    logging_strategy       = "epoch",
    report_to              = "none",

    seed                   = 42,

    fp16                   = torch.cuda.is_available(),
    dataloader_num_workers = 0,
    remove_unused_columns  = False,
)

# Configura logging_dir por variable de entorno
os.environ["TENSORBOARD_LOGGING_DIR"] = f"{OUTPUT_DIR}/logs"

print("\n✅ TrainingArguments configurado con parámetros agresivos.")

# COMMAND ----------



report_to = "mlflow", # enviar métricas directamente a MLflow
# ------------------------------------------------------------
# Mostrar la configuración completa
# ------------------------------------------------------------
print("=" * 60)
print("CONFIGURACIÓN DE ENTRENAMIENTO")
print("=" * 60)
print(f"  Épocas:                   {training_args.num_train_epochs}")
print(f"  Batch train:              {training_args.per_device_train_batch_size}")
print(f"  Batch eval:               {training_args.per_device_eval_batch_size}")
print(f"  Learning rate:            {training_args.learning_rate}")
print(f"  LR scheduler:             {training_args.lr_scheduler_type}")
print(f"  Warmup ratio:             {training_args.warmup_ratio}")
print(f"  Evaluación:               por {training_args.eval_strategy}")
print(f"  Mejor modelo según:       {training_args.metric_for_best_model}")
print(f"  Precisión mixta (fp16):   {training_args.fp16}")
print(f"  Directorio de salida:     {training_args.output_dir}")
print(f"  Reporte a:                MLflow")

# ------------------------------------------------------------
# Función de métricas
#
# El Trainer llama esta función al final de cada época
# con las predicciones y los labels reales.
# Debe devolver un diccionario con las métricas a registrar.
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("FUNCIÓN DE MÉTRICAS")
print("=" * 60)

def compute_metrics(eval_pred):
    """
    Calcula accuracy a partir de las predicciones del modelo.

    eval_pred contiene:
      - logits: puntuación cruda por clase  [n_ejemplos, n_clases]
      - labels: clase real de cada ejemplo  [n_ejemplos]
    """
    logits, labels = eval_pred

    # La clase predicha es la de mayor puntuación
    predicciones = np.argmax(logits, axis=-1)

    # Accuracy: porcentaje de predicciones correctas
    accuracy = (predicciones == labels).mean()

    # Accuracy por clase
    accuracy_por_clase = {}
    for idx, nombre in id2label.items():
        mask = labels == idx
        if mask.sum() > 0:
            acc_clase = (predicciones[mask] == labels[mask]).mean()
            accuracy_por_clase[f"accuracy_{nombre}"] = round(float(acc_clase), 4)

    return {
        "accuracy": round(float(accuracy), 4),
        **accuracy_por_clase,
    }

print("✅ compute_metrics definida.")
print("   Métricas que calcula:")
print("   → accuracy global")
print("   → accuracy por cada clase de daño")


# COMMAND ----------


# ------------------------------------------------------------
# Estimación de pasos de entrenamiento
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("ESTIMACIÓN DE PASOS")
print("=" * 60)

pasos_por_epoca = len(dataset_train_processed) // training_args.per_device_train_batch_size
pasos_totales   = pasos_por_epoca * training_args.num_train_epochs
pasos_warmup    = int(pasos_totales * (training_args.warmup_ratio or 0))

print(f"  Imágenes de entrenamiento: {len(dataset_train_processed):,}")
print(f"  Pasos por época:           {pasos_por_epoca:,}")
print(f"  Pasos totales:             {pasos_totales:,}")
print(f"  Pasos de warmup:           {pasos_warmup:,}")

# ------------------------------------------------------------
# Registrar configuración en MLflow
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("REGISTRANDO EN MLFLOW")
print("=" * 60)

with mlflow.start_run(run_name="configuracion_entrenamiento", nested=True):
    mlflow.log_param("num_train_epochs",            training_args.num_train_epochs)
    mlflow.log_param("per_device_train_batch_size", training_args.per_device_train_batch_size)
    mlflow.log_param("per_device_eval_batch_size",  training_args.per_device_eval_batch_size)
    mlflow.log_param("learning_rate",               training_args.learning_rate)
    mlflow.log_param("lr_scheduler_type",           training_args.lr_scheduler_type)
    mlflow.log_param("warmup_ratio",                training_args.warmup_ratio)
    mlflow.log_param("seed",                        training_args.seed)
    mlflow.log_param("fp16",                        training_args.fp16)
    mlflow.log_param("pasos_totales",               pasos_totales)
    mlflow.log_param("pasos_warmup",                pasos_warmup)

print("✅ Configuración registrada en MLflow.")

print("\n✅ Paso 9 completo.")
print("   training_args y compute_metrics listos para el Trainer.")

# COMMAND ----------

# DBTITLE 1,PASO 10: Entrenar el modelo
# ============================================================
# PASO 10: Entrenar el modelo con Fine-Tuning
# ============================================================

from transformers import Trainer
import mlflow

print("=" * 60)
print("INICIALIZANDO TRAINER")
print("=" * 60)

# ------------------------------------------------------------
# Crear el Trainer
#
# El Trainer orquesta todo el proceso de entrenamiento:
#   → Carga batches usando collate_fn
#   → Pasa los datos al modelo
#   → Calcula la pérdida
#   → Actualiza los pesos mediante backpropagation
#   → Evalúa en cada época usando compute_metrics
#   → Guarda checkpoints
#   → Registra métricas en MLflow
# ------------------------------------------------------------

trainer = Trainer(
    model           = model,
    args            = training_args,
    train_dataset   = dataset_train_processed,
    eval_dataset    = dataset_val_processed,
    data_collator   = collate_fn,
    compute_metrics = compute_metrics,
)

print("✅ Trainer inicializado correctamente.")
print(f"   Modelo:           {MODEL_CHECKPOINT}")
print(f"   Datos train:      {len(dataset_train_processed)} imágenes")
print(f"   Datos validación: {len(dataset_val_processed)} imágenes")
print(f"   Épocas:           {training_args.num_train_epochs}")
print(f"   Batch size:       {training_args.per_device_train_batch_size}")

# ------------------------------------------------------------
# Iniciar entrenamiento dentro de un run de MLflow
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("INICIANDO ENTRENAMIENTO")
print("=" * 60)
print("⏳ Este proceso puede tardar varios minutos...")
print()

with mlflow.start_run(run_name="entrenamiento_vit", nested=True):
    
    # Registrar parámetros clave
    mlflow.log_param("model", MODEL_CHECKPOINT)
    mlflow.log_param("train_size", len(dataset_train_processed))
    mlflow.log_param("val_size", len(dataset_val_processed))
    mlflow.log_param("epochs", training_args.num_train_epochs)
    mlflow.log_param("batch_size", training_args.per_device_train_batch_size)
    mlflow.log_param("learning_rate", training_args.learning_rate)
    
    # Entrenar
    train_result = trainer.train()
    
    # Registrar métricas finales de entrenamiento
    mlflow.log_metrics({
        "train_loss_final": train_result.training_loss,
        "train_steps": train_result.global_step,
    })
    
    print("\n" + "=" * 60)
    print("ENTRENAMIENTO COMPLETADO")
    print("=" * 60)
    print(f"   Pérdida final:    {train_result.training_loss:.4f}")
    print(f"   Pasos totales:    {train_result.global_step}")
    print(f"   Tiempo total:     {train_result.metrics.get('train_runtime', 'N/A')} seg")

# ------------------------------------------------------------
# Evaluación final en el conjunto de validación
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("EVALUACIÓN FINAL")
print("=" * 60)

eval_results = trainer.evaluate()

print(f"   Pérdida validación: {eval_results.get('eval_loss', 'N/A'):.4f}")
print(f"   Accuracy:           {eval_results.get('eval_accuracy', 'N/A'):.4f}")
print()
print("   Accuracy por clase:")
for key, value in eval_results.items():
    if key.startswith('eval_accuracy_'):
        clase = key.replace('eval_accuracy_', '')
        print(f"     • {clase}: {value:.4f}")

# Registrar resultados de evaluación en MLflow
with mlflow.start_run(run_name="evaluacion_final", nested=True):
    for key, value in eval_results.items():
        if isinstance(value, (int, float)):
            mlflow.log_metric(key, value)

print("\n✅ Evaluación completada y registrada en MLflow.")

# ------------------------------------------------------------
# Guardar el modelo entrenado
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("GUARDANDO MODELO")
print("=" * 60)

model_save_path = f"{OUTPUT_DIR}/modelo_final"
trainer.save_model(model_save_path)

print(f"✅ Modelo guardado en: {model_save_path}")
print(f"   El modelo incluye:")
print(f"     • Pesos del modelo entrenado")
print(f"     • Configuración del modelo")
print(f"     • Mapeos id2label y label2id")

print("\n" + "=" * 60)
print("✅ PASO 10 COMPLETO")
print("=" * 60)
print(f"Modelo '{MODEL_CHECKPOINT}' fine-tuneado exitosamente.")
print(f"Accuracy final: {eval_results.get('eval_accuracy', 'N/A'):.2%}")
print(f"\nPróximos pasos:")
print(f"  1. Revisar métricas en MLflow")
print(f"  2. Probar predicciones con nuevas imágenes")
print(f"  3. Registrar modelo en Unity Catalog si el desempeño es bueno")

# COMMAND ----------

# ============================================================
# PASO 11: Construir el pipeline de inferencia
# ============================================================

from transformers import pipeline, AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import io
import torch
import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient
from pyspark.sql import functions as F

CATALOG       = "proyecto_smart_claims"
SILVER_SCHEMA = "silver"
MODEL_CHECKPOINT = "google/vit-base-patch16-224"
OUTPUT_DIR       = "/tmp/smart_claims_vit"

# ------------------------------------------------------------
# Reconstruir diccionario de clases
# (por si el clúster se reinició y las variables se perdieron)
# ------------------------------------------------------------
print("=" * 60)
print("RECONSTRUYENDO DICCIONARIO DE CLASES")
print("=" * 60)

labels_distintos = sorted([
    row["label"] for row in
    spark.table(f"{CATALOG}.{SILVER_SCHEMA}.training_images")
    .select("label").distinct().collect()
])

label2id = {label: idx for idx, label in enumerate(labels_distintos)}
id2label  = {idx: label for label, idx in label2id.items()}

print(f"Clases encontradas: {labels_distintos}")
print(f"Total clases:       {len(label2id)}")

# ------------------------------------------------------------
# Cargar el modelo — intenta desde /tmp/ primero,
# si no existe lo carga desde MLflow
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("CARGANDO MODELO ENTRENADO")
print("=" * 60)

import os

if os.path.exists(f"{OUTPUT_DIR}/modelo_final"):
    print(f"✅ Modelo encontrado en {OUTPUT_DIR}/modelo_final")
    print("   Cargando desde disco local...")
    model = AutoModelForImageClassification.from_pretrained(
        f"{OUTPUT_DIR}/modelo_final",
        local_files_only   = True,
        num_labels         = len(label2id),
        id2label           = id2label,
        label2id           = label2id,
        ignore_mismatched_sizes = True,
    )
else:
    print(f"⚠️  Modelo no encontrado en {OUTPUT_DIR}/modelo_final")
    print("   Cargando desde MLflow...")
    try:
        model = mlflow.pytorch.load_model(
            "models:/smart_claims_clasificador_danios/latest"
        )
        print("✅ Modelo cargado desde MLflow.")
    except Exception as e:
        print(f"❌ No se pudo cargar desde MLflow: {e}")
        print("   Cargando modelo base sin fine-tuning como fallback...")
        model = AutoModelForImageClassification.from_pretrained(
            MODEL_CHECKPOINT,
            num_labels              = len(label2id),
            id2label                = id2label,
            label2id                = label2id,
            ignore_mismatched_sizes = True,
        )
        print("⚠️  Se cargó el modelo base SIN fine-tuning.")
        print("   Las predicciones no serán confiables.")

# Determinar dispositivo
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model  = model.to(DEVICE)
model.eval()

print(f"✅ Modelo listo en dispositivo: {DEVICE}")

# ------------------------------------------------------------
# Cargar el image processor
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("CARGANDO IMAGE PROCESSOR")
print("=" * 60)

image_processor = AutoImageProcessor.from_pretrained(MODEL_CHECKPOINT)
print(f"✅ Image processor cargado.")
print(f"   Tamaño esperado: {image_processor.size}")

# ------------------------------------------------------------
# Crear el pipeline de clasificación
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("CONSTRUYENDO PIPELINE DE INFERENCIA")
print("=" * 60)

clasificador = pipeline(
    task             = "image-classification",
    model            = model,
    image_processor  = image_processor,   # AutoImageProcessor en lugar de feature_extractor
    device           = 0 if torch.cuda.is_available() else -1,
)

print("✅ Pipeline de inferencia creado.")
print(f"   Tarea:       image-classification")
print(f"   Modelo:      {MODEL_CHECKPOINT}")
print(f"   Dispositivo: {'GPU' if torch.cuda.is_available() else 'CPU'}")
print(f"   Clases:      {list(label2id.keys())}")

# ------------------------------------------------------------
# Funciones auxiliares
# ------------------------------------------------------------

def predecir_imagen(imagen_input, top_k: int = None):
    """
    Recibe una imagen en bytes, PIL o ruta
    y devuelve las predicciones ordenadas por confianza.
    """
    if top_k is None:
        top_k = len(label2id)

    if isinstance(imagen_input, (bytes, bytearray)):
        img = Image.open(io.BytesIO(imagen_input)).convert("RGB")
    elif isinstance(imagen_input, Image.Image):
        img = imagen_input.convert("RGB")
    elif isinstance(imagen_input, str):
        img = Image.open(imagen_input).convert("RGB")
    else:
        raise ValueError(f"Formato no soportado: {type(imagen_input)}")

    return clasificador(img, top_k=top_k)


def mostrar_prediccion(resultados: list, imagen_nombre: str = "", label_real: str = ""):
    """
    Imprime los resultados de predicción de forma legible
    con barra visual de confianza.
    """
    titulo = f"PREDICCIÓN: {imagen_nombre}" if imagen_nombre else "PREDICCIÓN"
    print(f"\n{'─' * 55}")
    print(titulo)
    if label_real:
        print(f"Label real: '{label_real}'")
    print(f"{'─' * 55}")
    for i, r in enumerate(resultados):
        barra   = "█" * int(r["score"] * 30)
        espacio = "░" * (30 - len(barra))
        es_top  = " ← TOP 1" if i == 0 else ""
        print(f"  {r['label']:20s} {r['score']:6.2%}  {barra}{espacio}{es_top}")
    print(f"{'─' * 55}")
    if resultados:
        top1        = resultados[0]["label"]
        es_correcto = top1 == label_real if label_real else None
        if es_correcto is not None:
            print(f"  ¿Top-1 correcto?: {'✅ Sí' if es_correcto else '❌ No'}")


print("\n✅ Funciones predecir_imagen() y mostrar_prediccion() definidas.")

# ------------------------------------------------------------
# Prueba del pipeline con imágenes reales de Silver
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("PRUEBA DEL PIPELINE CON IMÁGENES DE VALIDACIÓN")
print("=" * 60)

muestra_val = (
    spark.table(f"{CATALOG}.{SILVER_SCHEMA}.training_images")
    .filter(F.col("content").isNotNull())
    .limit(3)
    .collect()
)

if not muestra_val:
    print("⚠️  No se encontraron imágenes con contenido binario en Silver.")
else:
    correctas = 0
    for fila in muestra_val:
        try:
            resultados = predecir_imagen(fila["content"])
            mostrar_prediccion(
                resultados,
                imagen_nombre = fila["image_name"],
                label_real    = fila["label"]
            )
            if resultados[0]["label"] == fila["label"]:
                correctas += 1
        except Exception as e:
            print(f"❌ Error procesando {fila['image_name']}: {e}")

    print(f"\n  Accuracy en muestra: {correctas}/{len(muestra_val)} "
          f"({correctas/len(muestra_val):.0%})")

# ------------------------------------------------------------
# Registrar en MLflow
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("REGISTRANDO PIPELINE EN MLFLOW")
print("=" * 60)

# Sin nested=True para evitar error si no hay run padre activo
with mlflow.start_run(run_name="pipeline_inferencia"):
    mlflow.log_param("pipeline_task",  "image-classification")
    mlflow.log_param("modelo_base",    MODEL_CHECKPOINT)
    mlflow.log_param("num_clases",     len(label2id))
    mlflow.log_param("clases",         str(list(label2id.keys())))
    mlflow.log_param("dispositivo",    DEVICE)

    try:
        mlflow.pytorch.log_model(
            pytorch_model         = model,
            artifact_path         = "modelo_clasificacion_danios",
            registered_model_name = "smart_claims_clasificador_danios",
        )
        print("✅ Modelo registrado en MLflow como:")
        print("   'smart_claims_clasificador_danios'")
    except Exception as e:
        print(f"⚠️  No se pudo registrar el modelo en MLflow: {e}")

# ------------------------------------------------------------
# Resumen
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("RESUMEN DEL PIPELINE DE INFERENCIA")
print("=" * 60)
print("""
  ENTRADA
    → Imagen en bytes, PIL o ruta de archivo

  PREPROCESAMIENTO (automático)
    → Conversión a RGB
    → Redimensionamiento a 224x224
    → Normalización con media y std de ImageNet

  MODELO
    → Forward pass por el ViT fine-tuneado
    → Logits por cada clase de daño

  SALIDA
    → Lista de clases con score de confianza
    → Ordenada de mayor a menor probabilidad
""")

print("✅ Paso 11 completo.")
print(f"   Pipeline 'clasificador' listo para predecir.")
print(f"   Clases disponibles: {list(label2id.keys())}")

# COMMAND ----------

# ============================================================
# PASO 12: Envolver el modelo para registrarlo en MLflow
# ============================================================

import mlflow
import mlflow.pyfunc
import torch
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification
import io
import os
import json
import pandas as pd
from mlflow.models import infer_signature

CATALOG          = "proyecto_smart_claims"
SILVER_SCHEMA    = "silver"
MODEL_CHECKPOINT = "google/vit-base-patch16-224"
OUTPUT_DIR       = "/tmp/smart_claims_vit"

# ------------------------------------------------------------
# ¿Por qué necesitamos un wrapper?
#
# El modelo tal como está es un objeto PyTorch.
# MLflow puede guardarlo, pero cuando alguien lo cargue
# después necesitaría saber:
#   → qué procesamiento aplicar a la imagen de entrada
#   → cómo interpretar los números que devuelve el modelo
#   → qué formato tiene la salida
#
# El wrapper encapsula TODO eso en una sola clase:
#   entrada: bytes o PIL Image
#   salida:  diccionario con label predicho y score
#
# Así cualquiera puede usar el modelo sin saber cómo funciona
# por dentro — solo le pasan una imagen y reciben una respuesta.
# ------------------------------------------------------------

# ------------------------------------------------------------
# Clase wrapper compatible con MLflow PythonModel
# ------------------------------------------------------------

class ClasificadorDaniosWrapper(mlflow.pyfunc.PythonModel):
    """
    Wrapper del modelo de clasificación de daños de vehículos.

    Encapsula el preprocesamiento, la inferencia y el
    postprocesamiento en una sola clase registrable en MLflow.

    Entrada:  imagen en bytes, PIL Image o ruta de archivo
    Salida:   diccionario con label, score y todas las clases
    """

    def __init__(self, model, image_processor, id2label, label2id):
        self.model           = model
        self.image_processor = image_processor
        self.id2label        = id2label
        self.label2id        = label2id

    def load_context(self, context):
        """
        Se llama automáticamente cuando MLflow carga el modelo.
        Reconstruye el modelo y el processor desde los artefactos.
        """
        model_path = context.artifacts["modelo_path"]

        self.image_processor = AutoImageProcessor.from_pretrained(
            context.artifacts["processor_path"]
        )

        id2label_path = context.artifacts["id2label_path"]
        with open(id2label_path, "r") as f:
            raw = json.load(f)
            self.id2label = {int(k): v for k, v in raw.items()}
            self.label2id = {v: int(k) for k, v in raw.items()}

        self.model = AutoModelForImageClassification.from_pretrained(
            model_path,
            local_files_only        = True,
            num_labels              = len(self.id2label),
            id2label                = self.id2label,
            label2id                = self.label2id,
            ignore_mismatched_sizes = True,
        )
        self.model.eval()

    def _abrir_imagen(self, imagen_input):
        """
        Convierte cualquier formato de entrada a PIL Image RGB.
        Acepta: bytes, bytearray, PIL Image, str (ruta)
        """
        if isinstance(imagen_input, (bytes, bytearray)):
            return Image.open(io.BytesIO(imagen_input)).convert("RGB")
        elif isinstance(imagen_input, Image.Image):
            return imagen_input.convert("RGB")
        elif isinstance(imagen_input, str):
            return Image.open(imagen_input).convert("RGB")
        elif isinstance(imagen_input, np.ndarray):
            return Image.fromarray(imagen_input.astype(np.uint8)).convert("RGB")
        else:
            raise ValueError(f"Formato de imagen no soportado: {type(imagen_input)}")

    def predict(self, context, model_input):
        """
        Método principal de inferencia.
        MLflow llama este método para hacer predicciones.

        model_input puede ser:
          - una imagen individual (bytes, PIL, str, numpy)
          - una lista de imágenes para batch inference
          - un pandas DataFrame con columna 'image'

        Retorna lista de dicts con:
          - label:      clase predicha (top 1)
          - score:      confianza de la predicción (0 a 1)
          - all_scores: score de cada clase disponible
        """
        # Normalizar entrada a lista de imágenes
        if isinstance(model_input, pd.DataFrame):
            imagenes_raw = model_input["image"].tolist()
        elif isinstance(model_input, list):
            imagenes_raw = model_input
        else:
            imagenes_raw = [model_input]

        # Convertir todas las entradas a PIL
        imagenes_pil = [self._abrir_imagen(img) for img in imagenes_raw]

        # Preprocesar con el image processor
        inputs = self.image_processor(
            images         = imagenes_pil,
            return_tensors = "pt"
        )

        # Inferencia sin gradientes (más rápido y menos memoria)
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits  = outputs.logits

        # Convertir logits a probabilidades con softmax
        probabilidades = torch.nn.functional.softmax(logits, dim=-1)
        probabilidades = probabilidades.numpy()

        # Construir resultado por cada imagen
        resultados = []
        for probs in probabilidades:
            idx_top1 = int(np.argmax(probs))

            resultado = {
                "label":      self.id2label[idx_top1],
                "score":      round(float(probs[idx_top1]), 4),
                "all_scores": {
                    self.id2label[i]: round(float(p), 4)
                    for i, p in enumerate(probs)
                }
            }
            resultados.append(resultado)

        # Si entró una sola imagen, devolver un solo dict
        return resultados if len(resultados) > 1 else resultados[0]


# ------------------------------------------------------------
# Guardar artefactos necesarios para el wrapper
# ------------------------------------------------------------
print("=" * 60)
print("PREPARANDO ARTEFACTOS")
print("=" * 60)

# Guardar id2label como JSON para que load_context lo pueda leer
id2label_path = f"{OUTPUT_DIR}/id2label.json"
os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(id2label_path, "w") as f:
    json.dump(id2label, f, indent=2)

print(f"✅ id2label guardado en: {id2label_path}")
print(f"   Contenido: {id2label}")

# Guardar el image processor a un directorio local
processor_path = f"{OUTPUT_DIR}/image_processor"
image_processor.save_pretrained(processor_path)
print(f"✅ Image processor guardado en: {processor_path}")

# Verificar que el modelo entrenado existe
modelo_path = f"{OUTPUT_DIR}/modelo_final"

if os.path.exists(modelo_path):
    print(f"✅ Modelo encontrado en: {modelo_path}")
else:
    print(f"⚠️  Modelo no encontrado en {modelo_path}")
    print(f"   Se usará el checkpoint base: {MODEL_CHECKPOINT}")
    modelo_path = MODEL_CHECKPOINT

# ------------------------------------------------------------
# Instanciar el wrapper
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("INSTANCIANDO EL WRAPPER")
print("=" * 60)

wrapper = ClasificadorDaniosWrapper(
    model           = model,
    image_processor = image_processor,
    id2label        = id2label,
    label2id        = label2id,
)

print("✅ Wrapper instanciado correctamente.")

# ------------------------------------------------------------
# Prueba del wrapper antes de registrar
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("PRUEBA DEL WRAPPER")
print("=" * 60)

from pyspark.sql import functions as F

muestra = (
    spark.table(f"{CATALOG}.{SILVER_SCHEMA}.training_images")
    .filter(F.col("content").isNotNull())
    .limit(3)
    .collect()
)

correctas = 0
example_input = None
example_output = None

for fila in muestra:
    try:
        resultado = wrapper.predict(context=None, model_input=fila["content"])
        label_predicho = resultado["label"]
        score          = resultado["score"]
        label_real     = fila["label"]
        es_correcto    = label_predicho == label_real

        print(f"\n  Imagen:    {fila['image_name']}")
        print(f"  Real:      '{label_real}'")
        print(f"  Predicho:  '{label_predicho}' ({score:.2%})")
        print(f"  Scores:    {resultado['all_scores']}")
        print(f"  Correcto:  {'✅ Sí' if es_correcto else '❌ No'}")

        if es_correcto:
            correctas += 1
        
        # Guardar ejemplo para la firma
        if example_input is None:
            example_input = pd.DataFrame({"image": [fila["content"]]})
            example_output = resultado

    except Exception as e:
        print(f"❌ Error en {fila['image_name']}: {e}")

print(f"\n  Accuracy en prueba: {correctas}/{len(muestra)} ({correctas/len(muestra):.0%})")

# ------------------------------------------------------------
# Crear firma del modelo
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("CREANDO FIRMA DEL MODELO")
print("=" * 60)

signature = infer_signature(example_input, example_output)
print(f"✅ Firma creada:")
print(f"   Inputs:  {signature.inputs}")
print(f"   Outputs: {signature.outputs}")

# ------------------------------------------------------------
# Registrar el wrapper en MLflow
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("REGISTRANDO WRAPPER EN MLFLOW")
print("=" * 60)

artifacts = {
    "modelo_path":    modelo_path,
    "processor_path": processor_path,
    "id2label_path":  id2label_path,
}

with mlflow.start_run(run_name="registro_wrapper"):

    mlflow.log_param("modelo_base",  MODEL_CHECKPOINT)
    mlflow.log_param("num_clases",   len(label2id))
    mlflow.log_param("clases",       str(list(label2id.keys())))

    mlflow.pyfunc.log_model(
        artifact_path         = "clasificador_danios_wrapper",
        python_model          = wrapper,
        artifacts             = artifacts,
        registered_model_name = "smart_claims_clasificador_danios_v2",
        signature             = signature,
        pip_requirements      = [
            "transformers",
            "torch",
            "Pillow",
            "numpy",
        ],
    )

    print("✅ Wrapper registrado en MLflow como:")
    print("   'smart_claims_clasificador_danios_v2'")

# ------------------------------------------------------------
# Resumen
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("RESUMEN DEL WRAPPER")
print("=" * 60)
print("""
  ClasificadorDaniosWrapper
  │
  ├── _abrir_imagen()   → acepta bytes, PIL, str, numpy
  ├── predict()         → inferencia individual o en batch
  └── load_context()    → reconstruye el modelo al cargar desde MLflow

  ENTRADA
    → bytes / PIL Image / ruta / numpy array
    → lista de imágenes para batch

  SALIDA por imagen
    → label:      clase predicha (top 1)
    → score:      confianza (0.0 a 1.0)
    → all_scores: score de todas las clases
""")

print("✅ Paso 12 completo.")
print("   El wrapper está registrado y listo para")
print("   ser cargado y usado en producción.")

# COMMAND ----------

# ============================================================
# PASO 13: Definir muestra de entrada y firma del modelo
# ============================================================

import mlflow
from mlflow.models.signature import infer_signature, ModelSignature
from mlflow.types.schema import Schema, ColSpec, TensorSpec
import pandas as pd
import numpy as np
from PIL import Image
import io
from pyspark.sql import functions as F

CATALOG       = "proyecto_smart_claims"
SILVER_SCHEMA = "silver"

# ------------------------------------------------------------
# ¿Qué es la firma del modelo?
#
# La firma documenta formalmente:
#   → INPUT:  qué tipo de datos espera el modelo como entrada
#   → OUTPUT: qué tipo de datos devuelve como salida
#
# Sin firma, MLflow guarda el modelo pero no sabe validar
# si alguien le pasa datos incorrectos al usarlo después.
#
# Con firma, MLflow puede:
#   → Validar entradas antes de hacer la predicción
#   → Documentar el contrato del modelo automáticamente
#   → Detectar incompatibilidades al servir el modelo
# ------------------------------------------------------------

# ------------------------------------------------------------
# PASO A: Tomar muestra real de imágenes desde Silver
# ------------------------------------------------------------
print("=" * 60)
print("PASO A: TOMANDO MUESTRA REAL DE SILVER")
print("=" * 60)

muestra_filas = (
    spark.table(f"{CATALOG}.{SILVER_SCHEMA}.training_images")
    .filter(F.col("content").isNotNull())
    .limit(5)
    .collect()
)

if not muestra_filas:
    raise ValueError("❌ No hay imágenes con contenido binario en Silver.")

print(f"✅ {len(muestra_filas)} imágenes tomadas como muestra.")

# ------------------------------------------------------------
# PASO B: Construir el DataFrame de entrada de muestra
#
# El wrapper espera un DataFrame con columna 'image'
# donde cada fila es el contenido binario de una imagen.
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("PASO B: CONSTRUYENDO DATAFRAME DE MUESTRA")
print("=" * 60)

# Construir muestra de entrada como DataFrame
input_sample = pd.DataFrame({
    "image": [bytes(fila["content"]) for fila in muestra_filas]
})

print(f"Shape del DataFrame de entrada:  {input_sample.shape}")
print(f"Tipo de dato en columna 'image': {input_sample['image'].dtype}")
print(f"Tamaño promedio por imagen:      "
      f"{input_sample['image'].apply(len).mean() / 1024:.1f} KB")
print(f"\nMuestra (primeros bytes de cada imagen):")
for i, row in input_sample.iterrows():
    nombre = muestra_filas[i]["image_name"]
    label  = muestra_filas[i]["label"]
    print(f"  [{i}] {nombre:30s} | label: {label:15s} | {len(row['image']):,} bytes")

# ------------------------------------------------------------
# PASO C: Generar predicciones de muestra para inferir la firma
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("PASO C: GENERANDO PREDICCIONES DE MUESTRA")
print("=" * 60)

predicciones_muestra = []

for fila in muestra_filas:
    try:
        resultado = wrapper.predict(context=None, model_input=fila["content"])
        predicciones_muestra.append({
            "label":      resultado["label"],
            "score":      resultado["score"],
            "all_scores": str(resultado["all_scores"]),
        })
    except Exception as e:
        print(f"⚠️  Error en predicción: {e}")
        predicciones_muestra.append({
            "label":      "error",
            "score":      0.0,
            "all_scores": "{}",
        })

output_sample = pd.DataFrame(predicciones_muestra)

print(f"Shape del DataFrame de salida:  {output_sample.shape}")
print(f"\nPredicciones generadas:")
print(output_sample[["label", "score"]].to_string(index=False))

# ------------------------------------------------------------
# PASO D: Inferir la firma automáticamente
#
# infer_signature analiza los DataFrames de entrada y salida
# y construye el schema automáticamente.
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("PASO D: INFERIR FIRMA AUTOMÁTICAMENTE")
print("=" * 60)

firma_inferida = infer_signature(
    model_input  = input_sample,
    model_output = output_sample,
)

print("✅ Firma inferida automáticamente.")
print(f"\nFIRMA DEL MODELO:")
print(f"\n  INPUT schema:")
for col in firma_inferida.inputs.inputs:
    print(f"    → {col.name}: {col.type}")

print(f"\n  OUTPUT schema:")
for col in firma_inferida.outputs.inputs:
    print(f"    → {col.name}: {col.type}")

# ------------------------------------------------------------
# PASO E: Definir firma manual como respaldo
#
# La firma manual es más explícita y documentada.
# Se usa si infer_signature no captura bien los tipos.
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("PASO E: FIRMA MANUAL COMO RESPALDO")
print("=" * 60)

firma_manual = ModelSignature(
    inputs  = Schema([
        ColSpec(type="binary", name="image"),
    ]),
    outputs = Schema([
        ColSpec(type="string", name="label"),
        ColSpec(type="double", name="score"),
        ColSpec(type="string", name="all_scores"),
    ])
)

print("✅ Firma manual definida.")
print(f"\n  INPUT:")
print(f"    → image:      binary  (contenido de la imagen en bytes)")
print(f"\n  OUTPUT:")
print(f"    → label:      string  (clase predicha, ej: 'dent', 'scratch')")
print(f"    → score:      double  (confianza entre 0.0 y 1.0)")
print(f"    → all_scores: string  (JSON con score de todas las clases)")

# Usar firma inferida si es válida, si no la manual
firma_final = firma_inferida if firma_inferida else firma_manual
print(f"\n✅ Firma final seleccionada: {'inferida' if firma_inferida else 'manual'}")

# ------------------------------------------------------------
# PASO F: Validar que la firma funciona con la muestra
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("PASO F: VALIDANDO FIRMA CON MUESTRA")
print("=" * 60)

errores_validacion = 0
for i, fila in enumerate(muestra_filas):
    try:
        pred = wrapper.predict(context=None, model_input=fila["content"])

        # Verificar que la salida tiene los campos esperados
        assert "label"      in pred, "Falta campo 'label'"
        assert "score"      in pred, "Falta campo 'score'"
        assert "all_scores" in pred, "Falta campo 'all_scores'"

        # Verificar tipos
        assert isinstance(pred["label"],      str),   "label debe ser string"
        assert isinstance(pred["score"],      float), "score debe ser float"
        assert isinstance(pred["all_scores"], dict),  "all_scores debe ser dict"

        # Verificar rango del score
        assert 0.0 <= pred["score"] <= 1.0, "score fuera de rango [0, 1]"

        # Verificar que el label predicho existe en el diccionario
        assert pred["label"] in label2id, f"label '{pred['label']}' no reconocido"

        print(f"  ✅ [{i}] {fila['image_name']:30s} → '{pred['label']}' ({pred['score']:.2%})")

    except AssertionError as e:
        print(f"  ❌ [{i}] {fila['image_name']}: {e}")
        errores_validacion += 1
    except Exception as e:
        print(f"  ❌ [{i}] Error inesperado: {e}")
        errores_validacion += 1

if errores_validacion == 0:
    print(f"\n✅ Todas las validaciones pasaron. La firma es correcta.")
else:
    print(f"\n⚠️  {errores_validacion} validaciones fallaron. Revisa el wrapper.")

# ------------------------------------------------------------
# Registrar muestra y firma en MLflow
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("REGISTRANDO EN MLFLOW")
print("=" * 60)

# Guardar el image processor a un directorio local
processor_path = f"{OUTPUT_DIR}/image_processor"
image_processor.save_pretrained(processor_path)
print(f"✅ Image processor guardado en: {processor_path}")

with mlflow.start_run(run_name="firma_modelo"):

    mlflow.log_param("input_schema",  "binary image")
    mlflow.log_param("output_schema", "label, score, all_scores")
    mlflow.log_param("muestra_size",  len(muestra_filas))
    mlflow.log_param("clases",        str(list(label2id.keys())))

    # Registrar el wrapper con firma y muestra de entrada
    mlflow.pyfunc.log_model(
        artifact_path         = "clasificador_con_firma",
        python_model          = wrapper,
        artifacts             = {
            "modelo_path":    modelo_path,
            "processor_path": processor_path,
            "id2label_path":  id2label_path,
        },
        signature             = firma_final,
        input_example         = input_sample.head(2),
        registered_model_name = "smart_claims_clasificador_danios_v2",
        pip_requirements      = [
            "transformers",
            "torch",
            "Pillow",
            "numpy",
        ],
    )

    print("✅ Modelo registrado con firma y muestra de entrada.")
    print("   Nombre: 'smart_claims_clasificador_danios_v2'")

# ------------------------------------------------------------
# Resumen
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("RESUMEN PASO 13")
print("=" * 60)
print(f"""
  Muestra de entrada:   {len(muestra_filas)} imágenes reales de Silver
  Formato entrada:      DataFrame con columna 'image' (binary)
  Formato salida:       DataFrame con label, score, all_scores

  FIRMA DEL MODELO
  ┌─────────────┬──────────┬─────────────────────────────────┐
  │ Campo       │ Tipo     │ Descripción                     │
  ├─────────────┼──────────┼─────────────────────────────────┤
  │ image       │ binary   │ Bytes de la imagen              │
  ├─────────────┼──────────┼─────────────────────────────────┤
  │ label       │ string   │ Clase predicha (top 1)          │
  │ score       │ double   │ Confianza entre 0.0 y 1.0       │
  │ all_scores  │ string   │ Score de cada clase disponible  │
  └─────────────┴──────────┴─────────────────────────────────┘
""")

print("✅ Paso 13 completo.")
print("   Firma definida, validada y registrada en MLflow.")
print("   El modelo está listo para el registro formal.")

# COMMAND ----------

# ============================================================
# PASO 14: Registrar el modelo en MLflow
# ============================================================

import mlflow
import mlflow.pyfunc
import mlflow.pytorch
from mlflow.tracking import MlflowClient
from mlflow.models.signature import infer_signature
import torch
import pandas as pd
import os

CATALOG          = "proyecto_smart_claims"
SILVER_SCHEMA    = "silver"
MODEL_CHECKPOINT = "google/vit-base-patch16-224"
OUTPUT_DIR       = "/tmp/smart_claims_vit"
EXPERIMENT_NAME  = "/Users/sebastiandavilahenao@gmail.com/proyecto_smart_claims/clasificacion_danios"
NOMBRE_MODELO    = "smart_claims_clasificador_danios"

# ------------------------------------------------------------
# ¿Cuál es la diferencia entre entrenamiento y registro?
#
# ENTRENAMIENTO (Paso 10):
#   → El modelo aprende con los datos
#   → Se guardan checkpoints en /tmp/
#   → Se registran métricas por época en MLflow
#   → Es temporal: /tmp/ se puede limpiar
#
# REGISTRO (este paso):
#   → El modelo queda guardado formalmente en MLflow
#   → Tiene versión, firma, dependencias y metadatos
#   → Es permanente: sobrevive reinicios del clúster
#   → Desde aquí se puede promover a Staging o Production
#   → Cualquier equipo puede cargarlo con una línea de código
# ------------------------------------------------------------

client = MlflowClient()

# ------------------------------------------------------------
# PASO A: Verificar estado antes de registrar
# ------------------------------------------------------------
print("=" * 60)
print("PASO A: VERIFICACIÓN PREVIA AL REGISTRO")
print("=" * 60)

# Verificar modelo entrenado
modelo_disponible = os.path.exists(f"{OUTPUT_DIR}/modelo_final")
print(f"Modelo en disco (/tmp/):  {'✅ Disponible' if modelo_disponible else '⚠️  No encontrado'}")

# Verificar wrapper
wrapper_ok = wrapper is not None
print(f"Wrapper instanciado:      {'✅ Listo' if wrapper_ok else '❌ No definido'}")

# Verificar firma
firma_ok = firma_final is not None
print(f"Firma del modelo:         {'✅ Definida' if firma_ok else '❌ No definida'}")

# Verificar artefactos
print(f"id2label.json:            {'✅ Existe' if os.path.exists(id2label_path) else '❌ No existe'}")

# Verificar experimento MLflow
experimento = client.get_experiment_by_name(EXPERIMENT_NAME)
print(f"Experimento MLflow:       {'✅ ' + experimento.experiment_id if experimento else '❌ No encontrado'}")

if not wrapper_ok or not firma_ok:
    raise RuntimeError("❌ Faltan elementos necesarios. Corre los pasos 12 y 13 primero.")

print("\n✅ Todo listo para el registro formal.")

# ------------------------------------------------------------
# PASO B: Crear corrida de registro en MLflow
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("PASO B: INICIANDO CORRIDA DE REGISTRO")
print("=" * 60)

mlflow.set_experiment(EXPERIMENT_NAME)

with mlflow.start_run(run_name="registro_formal_modelo") as run:

    RUN_ID = run.info.run_id
    print(f"Run ID:        {RUN_ID}")
    print(f"Experimento:   {EXPERIMENT_NAME}")

    # ── Parámetros del modelo ────────────────────────────────────
    mlflow.log_param("modelo_base",        MODEL_CHECKPOINT)
    mlflow.log_param("arquitectura",       "ViT-base-patch16-224")
    mlflow.log_param("num_clases",         len(label2id))
    mlflow.log_param("clases",             str(list(label2id.keys())))
    mlflow.log_param("dispositivo",        DEVICE)
    mlflow.log_param("fine_tuning",        True)
    mlflow.log_param("epochs_entrenados",  training_args.num_train_epochs)
    mlflow.log_param("learning_rate",      training_args.learning_rate)
    mlflow.log_param("batch_size",         training_args.per_device_train_batch_size)

    # ── Métricas finales ─────────────────────────────────────────
    if eval_results:
        for key, value in eval_results.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(key, value)
        print(f"✅ Métricas registradas:")
        print(f"   eval_accuracy: {eval_results.get('eval_accuracy', 'N/A')}")
        print(f"   eval_loss:     {eval_results.get('eval_loss', 'N/A')}")

    # ── Registrar el modelo con firma, muestra y wrapper ─
    print("\n⏳ Registrando modelo como artefacto en MLflow...")

    # Guardar el image processor a un directorio local
    processor_path = f"{OUTPUT_DIR}/image_processor"
    image_processor.save_pretrained(processor_path)
    print(f"✅ Image processor guardado en: {processor_path}")

    model_info = mlflow.pyfunc.log_model(
        artifact_path         = "clasificador_danios",
        python_model          = wrapper,
        artifacts             = {
            "modelo_path":    f"{OUTPUT_DIR}/modelo_final" if modelo_disponible else MODEL_CHECKPOINT,
            "processor_path": processor_path,
            "id2label_path":  id2label_path,
        },
        signature             = firma_final,
        input_example         = input_sample.head(2),
        registered_model_name = NOMBRE_MODELO,
        pip_requirements      = [
            "transformers>=4.30.0",
            "torch>=2.0.0",
            "Pillow>=9.0.0",
            "numpy>=1.23.0",
        ],
    )

    print(f"✅ Modelo registrado correctamente.")
    print(f"   URI del modelo: {model_info.model_uri}")

# ------------------------------------------------------------
# PASO C: Verificar el registro en el Model Registry
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("PASO C: VERIFICANDO REGISTRO EN MODEL REGISTRY")
print("=" * 60)

try:
    modelo_registrado = client.get_registered_model(NOMBRE_MODELO)
    print(f"Nombre:         {modelo_registrado.name}")
    print(f"Descripción:    {modelo_registrado.description or 'Sin descripción'}")

    versiones = client.search_model_versions(f"name='{NOMBRE_MODELO}'")
    print(f"\nVersiones registradas: {len(versiones)}")
    for v in versiones:
        print(f"  Versión {v.version}:")
        print(f"    Run ID:   {v.run_id}")
        print(f"    Estado:   {v.current_stage}")
        print(f"    URI:      {v.source}")

except Exception as e:
    print(f"⚠️  No se pudo consultar el Model Registry: {e}")

# ------------------------------------------------------------
# PASO D: Agregar descripción y tags al modelo registrado
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("PASO D: AGREGANDO DESCRIPCIÓN Y TAGS")
print("=" * 60)

try:
    # Descripción del modelo
    client.update_registered_model(
        name        = NOMBRE_MODELO,
        description = (
            "Modelo de clasificación de daños de vehículos. "
            f"Fine-tuning de {MODEL_CHECKPOINT} con {len(label2id)} clases: "
            f"{list(label2id.keys())}. "
            "Proyecto Smart Claims."
        )
    )

    # Tags con metadata útil
    client.set_registered_model_tag(NOMBRE_MODELO, "proyecto",      "smart_claims")
    client.set_registered_model_tag(NOMBRE_MODELO, "tipo",          "clasificacion_imagenes")
    client.set_registered_model_tag(NOMBRE_MODELO, "modelo_base",   MODEL_CHECKPOINT)
    client.set_registered_model_tag(NOMBRE_MODELO, "usuario",       "sebastiandavilahenao@gmail.com")

    print("✅ Descripción y tags agregados.")

except Exception as e:
    print(f"⚠️  No se pudieron agregar tags: {e}")

# ------------------------------------------------------------
# PASO E: Promover el modelo a Staging (opcional)
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("PASO E: PROMOCIÓN A STAGING (OPCIONAL)")
print("=" * 60)

try:
    versiones = client.search_model_versions(f"name='{NOMBRE_MODELO}'")
    if versiones:
        ultima_version = max([int(v.version) for v in versiones])
        print(f"Promoviendo versión {ultima_version} a Staging...")

        client.transition_model_version_stage(
            name    = NOMBRE_MODELO,
            version = ultima_version,
            stage   = "Staging",
        )

        print(f"✅ Versión {ultima_version} promovida a Staging.")
        print("   Ahora el modelo puede ser evaluado antes de ir a Production.")
    else:
        print("⚠️  No se encontraron versiones del modelo.")

except Exception as e:
    print(f"⚠️  Error al promover a Staging: {e}")

# ------------------------------------------------------------
# Resumen final
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("RESUMEN PASO 14")
print("=" * 60)
print(f"""
  MODELO REGISTRADO EN MLFLOW
  ┌────────────────────────────────────────────────────────────┐
  │ Nombre del modelo:     {NOMBRE_MODELO:35s} │
  │ Experimento MLflow:    {EXPERIMENT_NAME:35s} │
  │ Modelo base:           {MODEL_CHECKPOINT:35s} │
  │ Número de clases:      {len(label2id):35} │
  │ Estado:                Registrado y versionado            │
  │ Etapa:                 Staging (listo para evaluación)    │
  └────────────────────────────────────────────────────────────┘

  CÓMO CARGAR EL MODELO EN CUALQUIER NOTEBOOK:
  ------------------------------------------------
  import mlflow
  
  # Opción 1: Por nombre y stage
  model = mlflow.pyfunc.load_model(f"models:/{NOMBRE_MODELO}/Staging")
  
  # Opción 2: Por nombre y versión
  model = mlflow.pyfunc.load_model(f"models:/{NOMBRE_MODELO}/1")
  
  # Usar el modelo
  predicciones = model.predict(input_dataframe)
  
  ------------------------------------------------
  
  SIGUIENTE PASO:
    1. Evaluar el modelo en Staging con datos nuevos
    2. Si el modelo pasa validación → promover a Production
    3. Integrar en pipeline de inferencia batch o real-time
""")

print("✅ Paso 14 completo.")
print("   El modelo está formalmente registrado y listo para usarse.")
print(f"   Consulta el Model Registry en MLflow para ver todas las versiones.")


# COMMAND ----------

# ============================================================
# PASO 15: Publicar el modelo en Unity Catalog
# ============================================================

import mlflow
import mlflow.pyfunc
from mlflow.tracking import MlflowClient

CATALOG          = "proyecto_smart_claims"
SILVER_SCHEMA    = "silver"
MODEL_SCHEMA     = "models"           # schema dentro del catalog para modelos
MODEL_CHECKPOINT = "google/vit-base-patch16-224"
NOMBRE_MODELO    = "smart_claims_clasificador_danios"

# Nombre completo en Unity Catalog: catalog.schema.model
UC_MODEL_NAME    = f"{CATALOG}.{MODEL_SCHEMA}.{NOMBRE_MODELO}"
ALIAS            = "prod"

# ------------------------------------------------------------
# ¿Qué es Unity Catalog y por qué registrar aquí?
#
# MLflow Model Registry (Paso 14):
#   → Guarda el modelo en el workspace de Databricks
#   → Permite versiones y stages (Staging, Production)
#   → Accesible solo dentro del workspace
#
# Unity Catalog:
#   → Es la capa de gobernanza centralizada de Databricks
#   → El modelo queda como un activo gobernado (como una tabla)
#   → Permite control de acceso, linaje y auditoría
#   → Accesible desde cualquier workspace del mismo metastore
#   → Se referencia con: catalog.schema.model_name
#
# ALIAS vs VERSIÓN:
#   → Versión: número fijo (ej: versión 3)
#              si sube una nueva versión, hay que actualizar el código
#   → Alias:   apuntador simbólico (ej: "prod")
#              siempre apunta a la versión correcta sin tocar el código
#              cambiar de versión = solo mover el alias
# ------------------------------------------------------------

# Configurar MLflow para usar Unity Catalog
mlflow.set_registry_uri("databricks-uc")
client = MlflowClient()

# ------------------------------------------------------------
# PASO A: Crear el schema 'models' si no existe
# ------------------------------------------------------------
print("=" * 60)
print("PASO A: VERIFICANDO SCHEMA EN UNITY CATALOG")
print("=" * 60)

try:
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{MODEL_SCHEMA}")
    print(f"✅ Schema '{CATALOG}.{MODEL_SCHEMA}' listo.")
except Exception as e:
    print(f"⚠️  No se pudo crear el schema: {e}")

# ------------------------------------------------------------
# PASO B: Registrar el modelo en Unity Catalog
#
# Se toma el modelo del run del Paso 14 y se registra
# con el nombre completo catalog.schema.model
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("PASO B: REGISTRANDO MODELO EN UNITY CATALOG")
print("=" * 60)

try:
    # Registrar desde el run del Paso 14
    model_uri = f"runs:/{RUN_ID}/clasificador_danios"
    print(f"Fuente (Run ID): {RUN_ID}")
    print(f"URI del modelo:  {model_uri}")
    print(f"Destino UC:      {UC_MODEL_NAME}")
    print(f"\n⏳ Registrando en Unity Catalog...")

    version_uc = mlflow.register_model(
        model_uri = model_uri,
        name      = UC_MODEL_NAME,
    )

    print(f"\n✅ Modelo registrado en Unity Catalog.")
    print(f"   Nombre:   {version_uc.name}")
    print(f"   Versión:  {version_uc.version}")
    print(f"   Estado:   {version_uc.status}")

except Exception as e:
    print(f"❌ Error al registrar en Unity Catalog: {e}")
    print("   Verifica que el schema existe y tienes permisos.")
    raise

# ------------------------------------------------------------
# PASO C: Agregar descripción y tags en Unity Catalog
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("PASO C: AGREGANDO DESCRIPCIÓN Y TAGS")
print("=" * 60)

try:
    # Descripción del modelo registrado
    client.update_registered_model(
        name        = UC_MODEL_NAME,
        description = (
            f"Clasificador de daños de vehículos. "
            f"Fine-tuning de {MODEL_CHECKPOINT} con {len(label2id)} clases: "
            f"{list(label2id.keys())}. "
            f"Proyecto Smart Claims — capa Gold."
        )
    )

    # Tags del modelo
    client.set_registered_model_tag(UC_MODEL_NAME, "proyecto",       "smart_claims")
    client.set_registered_model_tag(UC_MODEL_NAME, "tipo",           "clasificacion_imagenes")
    client.set_registered_model_tag(UC_MODEL_NAME, "modelo_base",    MODEL_CHECKPOINT)
    client.set_registered_model_tag(UC_MODEL_NAME, "num_clases",     str(len(label2id)))
    client.set_registered_model_tag(UC_MODEL_NAME, "clases",         str(list(label2id.keys())))
    client.set_registered_model_tag(UC_MODEL_NAME, "equipo",         "data_science")
    client.set_registered_model_tag(UC_MODEL_NAME, "capa",           "gold")

    # Tags de la versión específica
    client.set_model_version_tag(
        name    = UC_MODEL_NAME,
        version = version_uc.version,
        key     = "accuracy",
        value   = str(eval_results.get("eval_accuracy", "N/A"))
    )
    client.set_model_version_tag(
        name    = UC_MODEL_NAME,
        version = version_uc.version,
        key     = "epochs",
        value   = str(training_args.num_train_epochs)
    )
    client.set_model_version_tag(
        name    = UC_MODEL_NAME,
        version = version_uc.version,
        key     = "learning_rate",
        value   = str(training_args.learning_rate)
    )

    print("✅ Descripción y tags agregados.")

except Exception as e:
    print(f"⚠️  No se pudieron agregar tags: {e}")

# ------------------------------------------------------------
# PASO D: Asignar el alias 'prod'
#
# El alias permite cargar el modelo sin saber el número
# de versión. Si en el futuro se entrena una nueva versión,
# solo se mueve el alias y todo el código que lo usa
# sigue funcionando sin cambios.
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("PASO D: ASIGNANDO ALIAS 'prod'")
print("=" * 60)

try:
    client.set_registered_model_alias(
        name    = UC_MODEL_NAME,
        alias   = ALIAS,
        version = version_uc.version,
    )

    print(f"✅ Alias '{ALIAS}' asignado a la versión {version_uc.version}.")
    print(f"\n   Cómo cargar el modelo con el alias:")
    print(f"   modelo = mlflow.pyfunc.load_model(")
    print(f"       'models:/{UC_MODEL_NAME}@{ALIAS}'")
    print(f"   )")

except Exception as e:
    print(f"⚠️  No se pudo asignar el alias: {e}")

# ------------------------------------------------------------
# PASO E: Verificar que el alias funciona
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("PASO E: VERIFICANDO ALIAS Y CARGA DESDE UC")
print("=" * 60)

try:
    # Cargar usando el alias en lugar del número de versión
    modelo_desde_uc = mlflow.pyfunc.load_model(
        f"models:/{UC_MODEL_NAME}@{ALIAS}"
    )
    print(f"✅ Modelo cargado desde Unity Catalog con alias '{ALIAS}'.")

    # Predicción de prueba
    pred_uc = modelo_desde_uc.predict(input_sample.head(1))
    print(f"✅ Predicción de prueba exitosa:")
    print(f"   {pred_uc}")

except Exception as e:
    print(f"⚠️  No se pudo cargar con alias: {e}")
    print(f"   Intenta cargar por versión explícita:")
    print(f"   mlflow.pyfunc.load_model(")
    print(f"       'models:/{UC_MODEL_NAME}/{version_uc.version}'")
    print(f"   )")

# ------------------------------------------------------------
# PASO F: Consultar versiones disponibles en UC
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("PASO F: VERSIONES EN UNITY CATALOG")
print("=" * 60)

try:
    versiones_uc = client.search_model_versions(f"name='{UC_MODEL_NAME}'")
    print(f"Total versiones registradas: {len(versiones_uc)}")
    print()
    for v in versiones_uc:
        alias_activo = "← prod" if str(v.version) == str(version_uc.version) else ""
        print(f"  Versión {v.version}  {alias_activo}")
        print(f"    Run ID:   {v.run_id}")
        print(f"    Estado:   {v.status}")
        print(f"    URI:      {v.source}")
        print()

except Exception as e:
    print(f"⚠️  No se pudieron listar versiones: {e}")

# ------------------------------------------------------------
# Resumen final
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("RESUMEN FINAL - MODELO EN UNITY CATALOG")
print("=" * 60)
print(f"""
  REGISTRO EN UNITY CATALOG
  ─────────────────────────
  Nombre completo:  {UC_MODEL_NAME}
  Versión activa:   {version_uc.version}
  Alias:            {ALIAS}

  CÓMO USARLO (sin depender del número de versión)
  ─────────────────────────────────────────────────
  import mlflow

  modelo = mlflow.pyfunc.load_model(
      "models:/{UC_MODEL_NAME}@{ALIAS}"
  )
  resultado = modelo.predict(input_sample)

  CÓMO ACTUALIZAR EL ALIAS CUANDO HAYA NUEVA VERSIÓN
  ────────────────────────────────────────────────────
  client.set_registered_model_alias(
      name    = "{UC_MODEL_NAME}",
      alias   = "{ALIAS}",
      version = <nueva_version>,   # solo cambiar esto
  )
  # El código que usa el modelo no necesita cambiar.
""")

print("✅ Paso 15 completo.")
print(f"   Modelo publicado en Unity Catalog como activo gobernado.")
print(f"   Referencia: models:/{UC_MODEL_NAME}@{ALIAS}")

# COMMAND ----------

# DBTITLE 1,PASO 16: Función de predicción en Spark
# ============================================================
# PASO 16: Función de predicción en Spark
# ============================================================

import mlflow.pyfunc
import pandas as pd
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, MapType

CATALOG       = "proyecto_smart_claims"
SILVER_SCHEMA = "silver"
GOLD_SCHEMA   = "gold"
MODEL_SCHEMA  = "models"
NOMBRE_MODELO = "smart_claims_clasificador_danios"
ALIAS         = "prod"

# Nombre completo en Unity Catalog
UC_MODEL_NAME = f"{CATALOG}.{MODEL_SCHEMA}.{NOMBRE_MODELO}"

# ------------------------------------------------------------
# PASO A: Verificar modelo en Unity Catalog
# ------------------------------------------------------------
print("=" * 60)
print("PASO A: VERIFICANDO MODELO EN UNITY CATALOG")
print("=" * 60)

try:
    # Cargar modelo desde Unity Catalog
    modelo = mlflow.pyfunc.load_model(f"models:/{UC_MODEL_NAME}@{ALIAS}")
    print(f"✅ Modelo cargado desde Unity Catalog")
    print(f"   Nombre: {UC_MODEL_NAME}")
    print(f"   Alias:  {ALIAS}")
except Exception as e:
    raise RuntimeError(
        f"❌ No se pudo cargar el modelo desde Unity Catalog.\n"
        f"   Modelo: {UC_MODEL_NAME}@{ALIAS}\n"
        f"   Error: {e}\n"
        f"   Solución: Ejecutar Cell 19 (Paso 15) para registrar el modelo en UC."
    )

# ------------------------------------------------------------
# PASO B: Función para procesar imágenes en batches
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("PASO B: CREANDO FUNCIÓN DE PREDICCIÓN")
print("=" * 60)

def predecir_imagenes_batch(spark_df, batch_size=10):
    """
    Procesa imágenes en batches en el driver (evita OOM en executors).
    
    Args:
        spark_df: DataFrame de Spark con columnas 'image_name' y 'content'
        batch_size: Tamaño del batch (reducir si hay OOM)
    
    Returns:
        DataFrame de Spark con predicciones
    """
    # Recolectar datos al driver
    rows = spark_df.select("image_name", "content").collect()
    
    resultados = []
    total = len(rows)
    
    print(f"Procesando {total} imágenes en batches de {batch_size}...")
    
    for i in range(0, total, batch_size):
        batch = rows[i:i+batch_size]
        batch_df = pd.DataFrame({
            "image": [bytes(row["content"]) for row in batch]
        })
        
        # Predecir batch
        predicciones = modelo.predict(batch_df)
        
        # Convertir resultados (puede ser dict o lista de dicts)
        if isinstance(predicciones, dict):
            predicciones = [predicciones]
        
        # Agregar image_name a cada resultado
        for j, pred in enumerate(predicciones):
            resultados.append({
                "image_name": batch[j]["image_name"],
                "damage_label": pred["label"],
                "damage_score": float(pred["score"]),
                "damage_all_scores": pred["all_scores"]
            })
        
        print(f"  Procesados {min(i+batch_size, total)}/{total}")
    
    # Convertir a Spark DataFrame
    schema = StructType([
        StructField("image_name", StringType(), False),
        StructField("damage_label", StringType(), True),
        StructField("damage_score", DoubleType(), True),
        StructField("damage_all_scores", MapType(StringType(), DoubleType()), True)
    ])
    
    return spark.createDataFrame(resultados, schema=schema)

print("✅ Función de predicción lista")

# ------------------------------------------------------------
# PASO D: Prueba con tabla Silver
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("PASO D: PRUEBA DE LA FUNCIÓN")
print("=" * 60)

df_prueba = (
    spark.table(f"{CATALOG}.{SILVER_SCHEMA}.training_images")
    .filter(F.col("content").isNotNull())
    .select("image_name", "label", "content")
    .limit(5)
)

# Predecir
df_resultado = predecir_imagenes_batch(df_prueba, batch_size=5)

# Unir con labels originales para comparar
df_resultado_completo = (
    df_resultado
    .join(
        df_prueba.select("image_name", "label"),
        on="image_name",
        how="inner"
    )
    .withColumn(
        "correcto",
        F.col("label") == F.col("damage_label")
    )
    .withColumn(
        "confianza",
        F.when(F.col("damage_score") >= 0.8, "alta")
         .when(F.col("damage_score") >= 0.5, "media")
         .otherwise("baja")
    )
)

print("\nResultados de prueba:")
df_resultado_completo.select(
    "image_name", "label", "damage_label", "damage_score", "correcto"
).show(truncate=False)

# Accuracy en la muestra de prueba
total_prueba = df_resultado_completo.count()
correctas_prueba = df_resultado_completo.filter(F.col("correcto")).count()
print(f"\nAccuracy en prueba: {correctas_prueba}/{total_prueba} "
      f"({correctas_prueba/total_prueba:.0%})")

# ------------------------------------------------------------
# PASO E: Aplicar predicción sobre claim_images
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("PASO E: PREDICCIÓN SOBRE CLAIM IMAGES")
print("=" * 60)

# Verificar que claim_images tiene contenido binario
claim_images = spark.table(f"{CATALOG}.{SILVER_SCHEMA}.claim_images")
tiene_content = "content" in claim_images.columns

if not tiene_content:
    print("⚠️  La tabla claim_images no tiene columna 'content'.")
    print("   Se necesita unir con la metadata para obtener el binario.")
    
    # Unir con metadata para obtener el contenido
    claim_images_con_content = (
        claim_images
        .join(
            spark.table(f"{CATALOG}.{SILVER_SCHEMA}.claim_images_metadata_clean"),
            on="image_name",
            how="left"
        )
    )
else:
    claim_images_con_content = claim_images

# Filtrar solo imágenes con content
df_para_predecir = claim_images_con_content.filter(F.col("content").isNotNull())
total_imagenes = df_para_predecir.count()
print(f"Total imágenes a procesar: {total_imagenes:,}")

if total_imagenes == 0:
    print("⚠️  No hay imágenes con content. Revisa la carga en Bronze/Silver.")
    df_claims_predicho = spark.createDataFrame([], schema=StructType([
        StructField("image_name", StringType(), False),
        StructField("damage_label", StringType(), True),
        StructField("damage_score", DoubleType(), True),
        StructField("damage_all_scores", MapType(StringType(), DoubleType()), True),
        StructField("confianza", StringType(), True)
    ]))
else:
    # Aplicar predicción en batches
    df_claims_predicho = (
        predecir_imagenes_batch(df_para_predecir, batch_size=20)
        .withColumn(
            "confianza",
            F.when(F.col("damage_score") >= 0.8, "alta")
             .when(F.col("damage_score") >= 0.5, "media")
             .otherwise("baja")
        )
    )
    
    print("\nPredicciones sobre claim_images:")
    df_claims_predicho.show(10, truncate=False)
    
    # Distribución de daños predichos
    print("\nDistribución de tipos de daño predichos:")
    (
        df_claims_predicho
        .groupBy("damage_label", "confianza")
        .count()
        .orderBy("damage_label", "confianza")
        .show()
    )

# ------------------------------------------------------------
# PASO F: Guardar predicciones en Gold
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("PASO F: GUARDANDO PREDICCIONES EN GOLD")
print("=" * 60)

if total_imagenes > 0:
    df_claims_predicho.write \
        .format("delta") \
        .mode("overwrite") \
        .saveAsTable(f"{CATALOG}.{GOLD_SCHEMA}.claim_images_predictions")
    
    total_predicciones = df_claims_predicho.count()
    print(f"✅ gold.claim_images_predictions guardada.")
    print(f"   Total predicciones: {total_predicciones:,}")
    
    # Muestra de la tabla guardada
    print("\nMuestra de la tabla guardada:")
    spark.table(f"{CATALOG}.{GOLD_SCHEMA}.claim_images_predictions") \
         .show(5, truncate=False)
else:
    print("⚠️  No se guardaron predicciones (no había imágenes para procesar).")

# ------------------------------------------------------------
# Resumen
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("RESUMEN PASO 16")
print("=" * 60)
print(f"""
  FUNCIÓN CREADA: predecir_imagenes_batch()
  ─────────────────────────────────────────────
  Tipo:      Procesamiento en Driver (batch)
  Entrada:   Spark DataFrame con 'image_name' y 'content'
  Salida:    Spark DataFrame con predicciones
  Modelo:    {UC_MODEL_NAME}@{ALIAS} (Unity Catalog)
  
  VENTAJAS DE ESTE ENFOQUE:
  • Evita OOM en executors (modelo ViT es muy pesado)
  • Procesamiento controlado en batches
  • Fácil de ajustar el tamaño del batch
  
  USO:
  ────
  df_predicciones = predecir_imagenes_batch(
      df_con_imagenes,
      batch_size=20  # Ajustar según memoria disponible
  )

  TABLA GENERADA
  ──────────────
  {CATALOG}.{GOLD_SCHEMA}.claim_images_predictions
  → Columnas: image_name, damage_label, damage_score,
              damage_all_scores, confianza
""")

print("✅ Paso 16 completo.")
print("   El modelo de Unity Catalog procesó las imágenes")
print("   exitosamente en el driver para evitar OOM.")

# COMMAND ----------

# DBTITLE 1,DEBUG: Test model outside UDF
# Test the model directly on driver (outside UDF)
import mlflow.pyfunc
import pandas as pd
from pyspark.sql import functions as F

CATALOG       = "proyecto_smart_claims"
SILVER_SCHEMA = "silver"
MODEL_SCHEMA  = "models"
NOMBRE_MODELO = "smart_claims_clasificador_danios"
ALIAS         = "prod"
UC_MODEL_NAME = f"{CATALOG}.{MODEL_SCHEMA}.{NOMBRE_MODELO}"

print("Loading model from UC...")
modelo = mlflow.pyfunc.load_model(f"models:/{UC_MODEL_NAME}@{ALIAS}")
print(f"✅ Model loaded: {type(modelo)}")

# Get a test image
print("\nFetching test image...")
test_row = (
    spark.table(f"{CATALOG}.{SILVER_SCHEMA}.training_images")
    .filter(F.col("content").isNotNull())
    .limit(1)
    .collect()[0]
)

image_bytes = bytes(test_row["content"])
print(f"Image: {test_row['image_name']}")
print(f"Label: {test_row['label']}")
print(f"Bytes length: {len(image_bytes)}")

# Test prediction
print("\nTesting prediction...")
try:
    input_df = pd.DataFrame({"image": [image_bytes]})
    print(f"Input DataFrame shape: {input_df.shape}")
    print(f"Input DataFrame columns: {input_df.columns.tolist()}")
    print(f"Input type: {type(input_df['image'].iloc[0])}")
    
    resultado = modelo.predict(input_df)
    print(f"\n✅ Prediction successful!")
    print(f"Result type: {type(resultado)}")
    print(f"Result: {resultado}")
except Exception as e:
    print(f"\n❌ Prediction failed!")
    print(f"Error type: {type(e).__name__}")
    print(f"Error message: {str(e)}")
    import traceback
    print(f"\nFull traceback:")
    traceback.print_exc()

# COMMAND ----------

# DBTITLE 1,PASO 17: Predecir sobre imágenes reales de reclamos — CORREGIDO
# ============================================================
# PASO 17: Predecir sobre imágenes reales de reclamos — CORREGIDO
# ============================================================

from pyspark.sql import functions as F

CATALOG       = "proyecto_smart_claims"
SILVER_SCHEMA = "silver"
GOLD_SCHEMA   = "gold"

# ------------------------------------------------------------
# PASO A: Verificar qué tiene claim_images
#
# CORRECCIÓN: claim_images solo tiene path e image_name,
# NO tiene content binario. El binario está en Bronze
# cargado con binaryFile. Hay que obtenerlo desde ahí.
# ------------------------------------------------------------
print("=" * 60)
print("PASO A: VERIFICANDO CLAIM IMAGES")
print("=" * 60)

claim_images = spark.table(f"{CATALOG}.{SILVER_SCHEMA}.claim_images")

print("Columnas disponibles:")
print(f"  {claim_images.columns}")
print(f"Total imágenes: {claim_images.count():,}")

tiene_content = "content" in claim_images.columns
print(f"Tiene 'content': {'✅ Sí' if tiene_content else '⚠️  No — se obtendrá desde Bronze'}")

# ------------------------------------------------------------
# PASO B: Obtener el contenido binario desde Bronze
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("PASO B: OBTENIENDO CONTENIDO BINARIO DESDE BRONZE")
print("=" * 60)

if not tiene_content:
    try:
        bronze_claim = spark.table(f"{CATALOG}.bronze.claim_images")

        if "content" in bronze_claim.columns:
            # Unir Silver con Bronze para obtener el binario
            df_con_content = (
                claim_images
                .join(
                    bronze_claim.select("image_name", "content"),
                    on  = "image_name",
                    how = "inner"           # inner: solo las que tienen binario
                )
            )
            total_con_content = df_con_content.count()
            print(f"✅ Contenido binario obtenido desde Bronze.")
            print(f"   Imágenes con content: {total_con_content:,}")
        else:
            print("❌ Bronze tampoco tiene 'content'.")
            print("   Verifica que claim_images se cargó con binaryFile.")
            df_con_content = claim_images.filter(F.lit(False))   # DataFrame vacío

    except Exception as e:
        print(f"❌ No se pudo acceder a Bronze: {e}")
        df_con_content = claim_images.filter(F.lit(False))
else:
    df_con_content = claim_images
    print("✅ claim_images ya tiene content binario.")

# ------------------------------------------------------------
# PASO C: Aplicar predicción
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("PASO C: APLICANDO MODELO")
print("=" * 60)

df_para_predecir = df_con_content.filter(F.col("content").isNotNull())
total_para_predecir = df_para_predecir.count()
print(f"⏳ Prediciendo {total_para_predecir:,} imágenes...")

if total_para_predecir == 0:
    print("⚠️  No hay imágenes con content. Revisa la carga en Bronze.")
else:
    df_predicciones = (
        df_para_predecir
        .withColumn("prediccion", predecir_danos(F.col("content")))
        .select(
            "image_name",
            F.col("prediccion.label").alias("damage_label"),
            F.col("prediccion.score").alias("damage_score"),
            F.col("prediccion.all_scores").alias("damage_all_scores"),
            F.current_timestamp().alias("predicted_at"),
        )
        .withColumn(
            "confidence_level",
            F.when(F.col("damage_score") >= 0.80, "alta")
             .when(F.col("damage_score") >= 0.50, "media")
             .otherwise("baja")
        )
        .withColumn("is_reliable",       F.col("damage_score") >= 0.5)
        .withColumn("is_null_prediction", F.col("damage_label").isNull())
    )

    # ── Estadísticas ─────────────────────────────────────────
    total_pred         = df_predicciones.count()
    pred_confiables    = df_predicciones.filter(F.col("is_reliable")).count()
    pred_nulas         = df_predicciones.filter(F.col("is_null_prediction")).count()

    print(f"\n✅ Predicciones generadas: {total_pred:,}")
    print(f"   Confiables (>= 0.5):    {pred_confiables:,}")
    print(f"   Nulas:                  {pred_nulas:,}")

    print("\nDistribución de daños predichos:")
    (
        df_predicciones
        .groupBy("damage_label", "confidence_level")
        .agg(
            F.count("*").alias("cantidad"),
            F.round(F.avg("damage_score"), 4).alias("score_promedio"),
        )
        .orderBy("damage_label")
        .show(truncate=False)
    )

    # ── Guardar en Gold ───────────────────────────────────────
    print("\n" + "=" * 60)
    print("GUARDANDO EN GOLD")
    print("=" * 60)

    (
        df_predicciones
        .write
        .format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(f"{CATALOG}.{GOLD_SCHEMA}.claim_images_predictions")
    )

    print(f"✅ gold.claim_images_predictions guardada con {total_pred:,} filas.")

    print("\nMuestra:")
    (
        spark.table(f"{CATALOG}.{GOLD_SCHEMA}.claim_images_predictions")
        .select("image_name", "damage_label",
                F.round("damage_score", 4).alias("score"),
                "confidence_level")
        .show(10, truncate=False)
    )

print("\n✅ Paso 17 completo.")
print("   Siguiente paso: cruzar con claim_images_metadata_clean")
print("   para asociar cada predicción con su claim_no.")

# COMMAND ----------

# COMMAND ----------

# DBTITLE 1,PASO 18: Unir predicciones con metadata de imágenes
# ============================================================
# PASO 18: Unir predicciones con metadata de imágenes
# ============================================================
#
# ¿POR QUÉ ES ESTE PASO NECESARIO?
# ─────────────────────────────────────────────────────────────
# El modelo trabajó únicamente con bytes de imagen e image_name.
# No sabe nada del reclamo al que pertenece esa imagen.
#
# gold.claim_images_predictions tiene:
#   → image_name, damage_label, damage_score, confidence_level
#   → NO tiene claim_no ni chassis_no
#
# silver.claim_images_metadata_clean tiene:
#   → image_name, claim_no, chassis_no y otros atributos
#   → NO tiene la predicción del modelo
#
# Sin unir ambas tablas, una predicción como:
#   "frontal_damage, score=0.91" no tiene ningún valor operativo
#   porque no sabemos a qué reclamo corresponde.
#
# Con la unión obtenemos:
#   claim_no + chassis_no + damage_label + damage_score
#   → La predicción queda vinculada al reclamo real.
#   → Esta tabla es la entrada directa para el área de siniestros.
# ============================================================

from pyspark.sql import functions as F

CATALOG       = "proyecto_smart_claims"
SILVER_SCHEMA = "silver"
GOLD_SCHEMA   = "gold"

# ------------------------------------------------------------
# PASO A: Cargar las dos tablas que se van a unir
# ------------------------------------------------------------
print("=" * 60)
print("PASO A: CARGANDO TABLAS FUENTE")
print("=" * 60)

# Predicciones generadas por el modelo en el Paso 17
df_predicciones = spark.table(f"{CATALOG}.{GOLD_SCHEMA}.claim_images_predictions")

# Metadata de imágenes con claim_no y chassis_no
df_metadata = spark.table(f"{CATALOG}.{SILVER_SCHEMA}.claim_images_metadata_clean")

print(f"gold.claim_images_predictions:        {df_predicciones.count():,} filas")
print(f"  Columnas: {df_predicciones.columns}")

print(f"\nsilver.claim_images_metadata_clean:   {df_metadata.count():,} filas")
print(f"  Columnas: {df_metadata.columns}")

# Verificar que la columna de unión existe en ambas tablas
for tabla_nombre, tabla_df in [
    ("gold.claim_images_predictions",      df_predicciones),
    ("silver.claim_images_metadata_clean", df_metadata),
]:
    if "image_name" in tabla_df.columns:
        print(f"\n✅ 'image_name' presente en {tabla_nombre}.")
    else:
        raise ValueError(
            f"❌ La columna 'image_name' no existe en {tabla_nombre}. "
            f"Revisa la carga antes de continuar."
        )

# ------------------------------------------------------------
# PASO B: Verificar la clave de unión — image_name
#
# Antes de hacer el join es importante entender cuántas
# imágenes en predicciones tienen correspondencia en metadata.
# Un número bajo de matches indica un problema en la carga
# de alguna de las dos capas.
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("PASO B: DIAGNÓSTICO DE LA CLAVE DE UNIÓN (image_name)")
print("=" * 60)

total_predicciones = df_predicciones.count()
total_metadata     = df_metadata.count()

# Cuántas predicciones tienen metadata
pred_con_metadata = (
    df_predicciones
    .join(df_metadata.select("image_name"), on="image_name", how="inner")
    .count()
)

# Cuántas predicciones NO tienen metadata (huérfanas)
pred_sin_metadata = total_predicciones - pred_con_metadata

print(f"Predicciones totales:             {total_predicciones:,}")
print(f"Metadata disponible:              {total_metadata:,}")
print(f"Predicciones con metadata:        {pred_con_metadata:,}  ({pred_con_metadata / total_predicciones:.1%})")
print(f"Predicciones sin metadata:        {pred_sin_metadata:,}  ({pred_sin_metadata / total_predicciones:.1%})")

if pred_sin_metadata > 0:
    print(f"\n⚠️  Hay {pred_sin_metadata:,} imagen(es) sin metadata.")
    print("   Pueden ser imágenes que no pasaron la validación en Silver.")
    print("   Se incluirán en la tabla Gold con claim_no = null.")
    print("\n   Muestra de imágenes sin metadata:")
    (
        df_predicciones
        .join(df_metadata.select("image_name"), on="image_name", how="left_anti")
        .select("image_name", "damage_label", "damage_score")
        .show(5, truncate=False)
    )
else:
    print("\n✅ Todas las predicciones tienen correspondencia en metadata.")

# ------------------------------------------------------------
# PASO C: Realizar la unión
#
# Se usa LEFT JOIN desde predicciones → metadata:
#   → Conserva TODAS las predicciones (incluso sin metadata)
#   → Las imágenes sin metadata quedan con claim_no = null
#   → No se pierden predicciones por ausencia de metadata
#
# Alternativa descartada — INNER JOIN:
#   → Solo conservaría imágenes con metadata
#   → Si hay imágenes huérfanas, sus predicciones se perderían
#   → El área de siniestros no podría saber qué imágenes faltan
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("PASO C: EJECUTANDO LA UNIÓN")
print("=" * 60)

# Columnas de metadata que queremos incorporar
# Se seleccionan explícitamente para evitar duplicados
cols_metadata = [
    c for c in df_metadata.columns
    if c != "image_name"   # image_name ya viene de predicciones
]

print(f"Columnas que se agregan desde metadata: {cols_metadata}")

df_predictions_enriched = (
    df_predicciones
    .join(
        df_metadata.select("image_name", *cols_metadata),
        on  = "image_name",
        how = "left",        # LEFT para no perder predicciones sin metadata
    )
)

total_enriquecido = df_predictions_enriched.count()
print(f"\n✅ Unión completada.")
print(f"   Filas en resultado:   {total_enriquecido:,}")
print(f"   Columnas resultantes: {df_predictions_enriched.columns}")

# Verificar que no se duplicaron filas (puede pasar si metadata
# tiene más de un registro por image_name)
if total_enriquecido > total_predicciones:
    duplicados = total_enriquecido - total_predicciones
    print(f"\n⚠️  El resultado tiene {duplicados:,} filas extra.")
    print("   Posible causa: metadata tiene image_name duplicados.")
    print("   Verificando...")
    (
        df_metadata
        .groupBy("image_name")
        .count()
        .filter(F.col("count") > 1)
        .orderBy(F.col("count").desc())
        .show(5, truncate=False)
    )
elif total_enriquecido == total_predicciones:
    print("✅ Sin duplicados. Relación 1:1 entre predicciones y metadata.")

# ------------------------------------------------------------
# PASO D: Validar que claim_no y chassis_no se recuperaron
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("PASO D: VALIDANDO CAMPOS CLAVE RECUPERADOS")
print("=" * 60)

for campo in ["claim_no", "chassis_no"]:
    if campo in df_predictions_enriched.columns:
        nulos = df_predictions_enriched.filter(F.col(campo).isNull()).count()
        presentes = total_enriquecido - nulos
        print(f"  {campo}:")
        print(f"    Con valor:  {presentes:,}  ({presentes / total_enriquecido:.1%})")
        print(f"    Nulos:      {nulos:,}  ({nulos / total_enriquecido:.1%})")
        if nulos == 0:
            print(f"    ✅ Todos los registros tienen {campo}.")
        else:
            print(f"    ⚠️  Hay registros sin {campo}. Corresponden a imágenes sin metadata.")
    else:
        print(f"  ⚠️  La columna '{campo}' no está en metadata. Revisa silver.")

# ------------------------------------------------------------
# PASO E: Guardar en Gold
#
# La tabla resultante es la salida operativa del pipeline:
# une la inteligencia del modelo con la identidad del reclamo.
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("PASO E: GUARDANDO EN GOLD")
print("=" * 60)

(
    df_predictions_enriched
    .write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(f"{CATALOG}.{GOLD_SCHEMA}.claim_damage_predictions")
)

print(f"✅ gold.claim_damage_predictions guardada con {total_enriquecido:,} filas.")

# ------------------------------------------------------------
# PASO F: Verificación final con muestra
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("PASO F: MUESTRA DE LA TABLA FINAL")
print("=" * 60)

cols_muestra = [
    c for c in ["claim_no", "chassis_no", "image_name",
                "damage_label", "damage_score", "confidence_level"]
    if c in df_predictions_enriched.columns
]

(
    spark.table(f"{CATALOG}.{GOLD_SCHEMA}.claim_damage_predictions")
    .select(*cols_muestra)
    .orderBy("claim_no", "damage_score", ascending=[True, False])
    .show(15, truncate=False)
)

# Resumen por claim_no: cuántas imágenes tiene cada reclamo
# y cuál es el daño predominante
print("\n" + "=" * 60)
print("RESUMEN: DAÑO PREDOMINANTE POR RECLAMO")
print("=" * 60)

if "claim_no" in df_predictions_enriched.columns:
    (
        spark.table(f"{CATALOG}.{GOLD_SCHEMA}.claim_damage_predictions")
        .groupBy("claim_no")
        .agg(
            F.count("image_name").alias("num_imagenes"),
            F.first(
                "damage_label",
                ignorenulls=True
            ).alias("damage_label_top"),
            F.round(F.max("damage_score"), 4).alias("max_score"),
            F.round(F.avg("damage_score"), 4).alias("avg_score"),
            F.collect_set("confidence_level").alias("niveles_confianza"),
        )
        .orderBy(F.col("max_score").desc())
        .show(15, truncate=False)
    )

# ------------------------------------------------------------
# RESUMEN CONCEPTUAL DEL PASO
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("RESUMEN PASO 18")
print("=" * 60)
print(f"""
  QUÉ SE HIZO
  ────────────────────────────────────────────────────────────
  Se unieron dos tablas usando image_name como llave:

    gold.claim_images_predictions        (predicciones del modelo)
    silver.claim_images_metadata_clean   (identidad del reclamo)
                     ↓ LEFT JOIN por image_name
    gold.claim_damage_predictions        (tabla enriquecida)

  QUÉ CONTIENE LA TABLA RESULTANTE
  ────────────────────────────────────────────────────────────
  → claim_no:         identificador del reclamo
  → chassis_no:       número de chasis del vehículo
  → image_name:       nombre del archivo de imagen
  → damage_label:     tipo de daño predicho por el modelo
  → damage_score:     confianza de la predicción (0 a 1)
  → confidence_level: alta / media / baja

  POR QUÉ ES EL PASO MÁS IMPORTANTE
  ────────────────────────────────────────────────────────────
  Sin esta unión, el modelo sabe que una imagen muestra
  "frontal_damage" con 91% de confianza, pero no sabe
  a qué reclamo ni a qué vehículo pertenece esa imagen.

  Con esta unión, el área de siniestros puede consultar:
    "El reclamo CLM-00482 tiene 3 imágenes.
     La predicción dominante es total_loss con score 0.88."

  SIGUIENTE PASO SUGERIDO
  ────────────────────────────────────────────────────────────
  Cruzar gold.claim_damage_predictions con la tabla de
  reclamos principal para incorporar monto asegurado,
  fecha del siniestro y datos del asegurado.
""")

print("✅ Paso 18 completo.")
print(f"   gold.claim_damage_predictions lista con {total_enriquecido:,} registros.")
print("   Cada predicción está ahora vinculada a su claim_no y chassis_no.")

# COMMAND ----------

# COMMAND ----------

# DBTITLE 1,PASO 19: Guardar la tabla final de predicciones
# ============================================================
# PASO 19: Guardar la tabla final de predicciones
# ============================================================
#
# ¿QUÉ SE HACE EN ESTE PASO?
# ─────────────────────────────────────────────────────────────
# Se construye y persiste la tabla Gold definitiva del pipeline
# de Machine Learning: gold.claim_predictions_final
#
# Esta tabla es el producto entregable del modelo.
# A diferencia de las tablas intermedias de pasos anteriores:
#
#   gold.claim_images_predictions   → solo predicción + image_name
#   gold.claim_damage_predictions   → predicción + metadata raw
#   gold.claim_predictions_final    → tabla curada, tipada,
#                                     lista para reglas de negocio
#
# ¿QUÉ TIENE QUE TENER ESTA TABLA?
# ─────────────────────────────────────────────────────────────
#   → image_name:       identificador único de la imagen
#   → damage_label:     tipo de daño predicho por el modelo
#   → damage_score:     confianza de la predicción (0.0 – 1.0)
#   → confidence_level: alta / media / baja
#   → claim_no:         número del reclamo (del área de siniestros)
#   → chassis_no:       número de chasis del vehículo asegurado
#   → is_reliable:      flag booleano — score >= 0.5
#   → predicted_at:     timestamp de la predicción
#   → pipeline_version: trazabilidad del modelo que generó la pred.
#
# ¿POR QUÉ SE CURAN LOS DATOS ANTES DE GUARDAR?
# ─────────────────────────────────────────────────────────────
# Las tablas intermedias acumulan columnas de diagnóstico,
# flags temporales y campos de depuración útiles durante el
# desarrollo pero innecesarios para el consumidor de datos.
# La tabla final solo expone lo que el área de negocio necesita.
# ============================================================

from pyspark.sql import functions as F
from pyspark.sql import types as T

CATALOG       = "proyecto_smart_claims"
SILVER_SCHEMA = "silver"
GOLD_SCHEMA   = "gold"
MODEL_SCHEMA  = "models"
NOMBRE_MODELO = "smart_claims_clasificador_danios"
UC_MODEL_NAME = f"{CATALOG}.{MODEL_SCHEMA}.{NOMBRE_MODELO}"
ALIAS         = "prod"

# Nombre de la tabla final — el producto del pipeline ML
TABLA_FINAL   = f"{CATALOG}.{GOLD_SCHEMA}.claim_predictions_final"

# ------------------------------------------------------------
# PASO A: Cargar la tabla enriquecida del Paso 18
# ------------------------------------------------------------
print("=" * 60)
print("PASO A: CARGANDO TABLA ENRIQUECIDA DEL PASO 18")
print("=" * 60)

df_enriquecido = spark.table(f"{CATALOG}.{GOLD_SCHEMA}.claim_damage_predictions")

total_entrada = df_enriquecido.count()
print(f"Filas recibidas desde Paso 18:  {total_entrada:,}")
print(f"Columnas disponibles:           {df_enriquecido.columns}")

# ------------------------------------------------------------
# PASO B: Seleccionar y tipar las columnas finales
#
# Se seleccionan únicamente las columnas necesarias para el
# área de negocio. Se aplica tipado explícito para garantizar
# que el schema de la tabla Delta sea estable y predecible.
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("PASO B: SELECCIÓN Y TIPADO DE COLUMNAS FINALES")
print("=" * 60)

# Verificar que todas las columnas requeridas existen
columnas_requeridas = [
    "image_name", "damage_label", "damage_score",
    "confidence_level", "claim_no", "chassis_no",
    "is_reliable", "predicted_at",
]

faltantes = [c for c in columnas_requeridas if c not in df_enriquecido.columns]

if faltantes:
    print(f"⚠️  Columnas faltantes en la tabla enriquecida: {faltantes}")
    print("   Revisa que el Paso 18 se ejecutó correctamente.")
else:
    print("✅ Todas las columnas requeridas están presentes.")

# Construir la tabla final con schema controlado
df_final = (
    df_enriquecido
    .select(
        # ── Identidad del reclamo ─────────────────────────────
        F.col("claim_no")
         .cast(T.StringType())
         .alias("claim_no"),

        F.col("chassis_no")
         .cast(T.StringType())
         .alias("chassis_no"),

        # ── Identidad de la imagen ────────────────────────────
        F.col("image_name")
         .cast(T.StringType())
         .alias("image_name"),

        # ── Predicción del modelo ─────────────────────────────
        F.col("damage_label")
         .cast(T.StringType())
         .alias("damage_label"),

        F.round(F.col("damage_score").cast(T.DoubleType()), 4)
         .alias("damage_score"),

        F.col("confidence_level")
         .cast(T.StringType())
         .alias("confidence_level"),

        # ── Flags operativos ──────────────────────────────────
        F.col("is_reliable")
         .cast(T.BooleanType())
         .alias("is_reliable"),

        # ── Trazabilidad ──────────────────────────────────────
        F.col("predicted_at")
         .cast(T.TimestampType())
         .alias("predicted_at"),

        F.lit(f"{UC_MODEL_NAME}@{ALIAS}")
         .cast(T.StringType())
         .alias("pipeline_version"),
    )
)

print("\nSchema de la tabla final:")
df_final.printSchema()

# ------------------------------------------------------------
# PASO C: Validaciones de calidad antes de escribir
#
# Se aplican tres validaciones críticas:
#   1. Sin filas duplicadas por (claim_no, image_name)
#   2. Sin damage_label nulo en registros confiables
#   3. Distribución final de etiquetas
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("PASO C: VALIDACIONES DE CALIDAD")
print("=" * 60)

total_final = df_final.count()

# 1. Duplicados por clave natural
duplicados = (
    df_final
    .groupBy("claim_no", "image_name")
    .count()
    .filter(F.col("count") > 1)
    .count()
)

if duplicados == 0:
    print("✅ Sin duplicados por (claim_no, image_name).")
else:
    print(f"⚠️  Hay {duplicados:,} combinaciones (claim_no, image_name) duplicadas.")
    print("   Revisa si la metadata tiene registros repetidos.")

# 2. Predicciones nulas en registros confiables
nulos_label_confiable = (
    df_final
    .filter(F.col("is_reliable") & F.col("damage_label").isNull())
    .count()
)

if nulos_label_confiable == 0:
    print("✅ Sin damage_label nulo en registros marcados como confiables.")
else:
    print(f"⚠️  {nulos_label_confiable:,} registros confiables tienen damage_label nulo.")

# 3. Registros sin claim_no (huérfanos)
sin_claim = df_final.filter(F.col("claim_no").isNull()).count()
con_claim = total_final - sin_claim

print(f"\nCobertura de claim_no:")
print(f"  Con claim_no:    {con_claim:,}  ({con_claim / total_final:.1%})")
print(f"  Sin claim_no:    {sin_claim:,}  ({sin_claim / total_final:.1%})")

if sin_claim > 0:
    print("  ⚠️  Los registros sin claim_no no podrán usarse en reglas de negocio.")
    print("     Considera filtrarlos o investigar su origen.")

# 4. Distribución de etiquetas de daño
print("\nDistribución de damage_label:")
(
    df_final
    .groupBy("damage_label", "confidence_level")
    .agg(
        F.count("*").alias("cantidad"),
        F.round(F.avg("damage_score"), 4).alias("score_promedio"),
        F.round(F.count("*") / total_final * 100, 2).alias("pct_%"),
    )
    .orderBy("damage_label", "confidence_level")
    .show(truncate=False)
)

# ------------------------------------------------------------
# PASO D: Persistir en Gold como tabla Delta
#
# Se usa mode("overwrite") + overwriteSchema para garantizar
# que el schema de la tabla siempre refleja la definición
# actual, sin que versiones anteriores de columnas interfieran.
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("PASO D: ESCRIBIENDO EN GOLD")
print("=" * 60)

(
    df_final
    .write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(TABLA_FINAL)
)

print(f"✅ Tabla guardada: {TABLA_FINAL}")
print(f"   Total de filas: {total_final:,}")

# ------------------------------------------------------------
# PASO E: Verificación post-escritura
#
# Se recarga la tabla desde el catálogo para confirmar que
# la escritura fue exitosa y el schema quedó correcto.
# No se confía en el DataFrame en memoria — se lee desde disco.
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("PASO E: VERIFICACIÓN POST-ESCRITURA")
print("=" * 60)

df_verificacion = spark.table(TABLA_FINAL)
filas_escritas  = df_verificacion.count()

print(f"Filas escritas (verificado desde catálogo): {filas_escritas:,}")

if filas_escritas == total_final:
    print("✅ Conteo coincide. Escritura correcta.")
else:
    print(f"⚠️  Discrepancia: se esperaban {total_final:,} filas, "
          f"se encontraron {filas_escritas:,}.")

print("\nSchema en catálogo:")
df_verificacion.printSchema()

print("\nMuestra de la tabla final:")
(
    df_verificacion
    .select(
        "claim_no", "chassis_no", "image_name",
        "damage_label",
        F.round("damage_score", 4).alias("score"),
        "confidence_level", "is_reliable",
    )
    .orderBy("claim_no", F.col("damage_score").desc())
    .show(15, truncate=False)
)

# ------------------------------------------------------------
# PASO F: Estadísticas finales del pipeline completo
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("PASO F: ESTADÍSTICAS FINALES DEL PIPELINE")
print("=" * 60)

stats = df_verificacion.agg(
    F.count("*")                                    .alias("total_registros"),
    F.countDistinct("claim_no")                     .alias("claims_cubiertos"),
    F.countDistinct("chassis_no")                   .alias("chassis_distintos"),
    F.countDistinct("image_name")                   .alias("imagenes_procesadas"),
    F.sum(F.col("is_reliable").cast("int"))         .alias("predicciones_confiables"),
    F.round(F.avg("damage_score"), 4)               .alias("score_promedio"),
    F.round(F.min("damage_score"), 4)               .alias("score_minimo"),
    F.round(F.max("damage_score"), 4)               .alias("score_maximo"),
    F.min("predicted_at")                           .alias("primera_prediccion"),
    F.max("predicted_at")                           .alias("ultima_prediccion"),
).collect()[0]

print(f"  Total registros:              {stats['total_registros']:,}")
print(f"  Claims cubiertos:             {stats['claims_cubiertos']:,}")
print(f"  Chassis distintos:            {stats['chassis_distintos']:,}")
print(f"  Imágenes procesadas:          {stats['imagenes_procesadas']:,}")

# Handle None values from aggregations
predicciones_confiables = stats['predicciones_confiables'] or 0
total_registros = stats['total_registros']
pct_confiables = (predicciones_confiables / total_registros * 100) if total_registros > 0 else 0

print(f"  Predicciones confiables:      {predicciones_confiables:,}  ({pct_confiables:.1f}%)")
print(f"  Score promedio:               {stats['score_promedio'] or 'N/A'}")
print(f"  Score mínimo / máximo:        {stats['score_minimo'] or 'N/A'} / {stats['score_maximo'] or 'N/A'}")
print(f"  Rango temporal predicciones:  {stats['primera_prediccion']}  →  {stats['ultima_prediccion']}")

# ------------------------------------------------------------
# RESUMEN CONCEPTUAL DEL PASO
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("RESUMEN PASO 19")
print("=" * 60)
print(f"""
  QUÉ SE PRODUJO
  ────────────────────────────────────────────────────────────
  {TABLA_FINAL}

  SCHEMA FINAL
  ────────────────────────────────────────────────────────────
  claim_no          STRING     → número del reclamo
  chassis_no        STRING     → número de chasis del vehículo
  image_name        STRING     → nombre del archivo de imagen
  damage_label      STRING     → tipo de daño predicho
  damage_score      DOUBLE     → confianza del modelo (0.0–1.0)
  confidence_level  STRING     → alta / media / baja
  is_reliable       BOOLEAN    → score >= 0.5
  predicted_at      TIMESTAMP  → momento de la predicción
  pipeline_version  STRING     → modelo UC que generó la pred.

  LINAJE DEL PIPELINE
  ────────────────────────────────────────────────────────────
  bronze.claim_images          (binario de imágenes)
       ↓
  silver.claim_images          (imágenes validadas)
  silver.claim_images_metadata_clean  (identidad del reclamo)
       ↓
  gold.claim_images_predictions       (Paso 17 — pred. cruda)
  gold.claim_damage_predictions       (Paso 18 — enriquecida)
       ↓
  gold.claim_predictions_final        (Paso 19 — tabla final)

  LISTO PARA
  ────────────────────────────────────────────────────────────
  → Reglas de negocio (Fase siguiente del proyecto)
  → Consultas del área de siniestros
  → Dashboards de supervisión de reclamos
  → Auditoría y trazabilidad del modelo
""")

print("✅ Paso 19 completo.")
print(f"   {TABLA_FINAL}")
print(f"   {filas_escritas:,} predicciones listas para la fase de reglas de negocio.")

