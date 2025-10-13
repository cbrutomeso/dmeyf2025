from datetime import datetime
import time
import os
import polars as pl
import numpy as np
import pandas as pd
import yaml
import random
import lightgbm as lgb
import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import gc
import logging

import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="X does not have valid feature names")



# ## Optimizacion Hiperparámetros

# aqui debe cargar SU semilla primigenia
# <br>recuerde cambiar el numero de experimento en cada corrida nueva

# In[7]:


PARAM = {}
PARAM["experimento"] = 4966
PARAM["semilla_primigenia"] = 100129
PARAM["semillas_ensemble"] = [
    100129, 200357, 456791, 400429, 500459,  # originales
    670373, 890173, 104729, 1299709, 15485863  # 5 nuevos primos grandes
]

# Configure logging
log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
log_path = os.path.join(log_dir, "experimento_{}.log".format(PARAM["experimento"]))
if os.path.exists(log_path):
    os.remove(log_path)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_path, mode="a"),  # se asegura que cada corrida arranque de cero
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
logger.info(f"Iniciando experimento {PARAM['experimento']} con semilla {PARAM['semilla_primigenia']}")

# -------- Fijar semillas --------
random.seed(PARAM["experimento"])
np.random.seed(PARAM["experimento"])
os.environ["PYTHONHASHSEED"] = str(PARAM["experimento"])

# In[8]:


# training and future
PARAM["train"] = [202102]
PARAM["train_final"] = [202104]
PARAM["future"] = [202104]
PARAM["kaggle"] = [202106]
PARAM["semilla_kaggle"] = 314159
PARAM["cortes"] = list(range(6000, 19001, 500))


# In[9]:


# un undersampling de 0.1  toma solo el 10% de los CONTINUA
# undersampling de 1.0  implica tomar TODOS los datos

PARAM["trainingstrategy"] = {}
PARAM["trainingstrategy"]["undersampling"] = 0.5


# ### Definicion de Parametros

# In[10]:


# Parámetros de LightGBM

PARAM["hyperparametertuning"] = {}
PARAM["hyperparametertuning"]["validation_split"] = 0.2  # 20% para validation, 80% para train

# Parámetros fijos de LightGBM que serán sobreescritos por la parte variable de la BO
PARAM["lgbm"] = {}
PARAM["lgbm"]["param_fijos"] = {
    "boosting": "gbdt",  # puede ir 'dart', no probar 'random_forest'
    "objective": "binary",
    "metric": "auc",
    "first_metric_only": False,
    "boost_from_average": True,
    "feature_pre_filter": False,
    "force_row_wise": True,  # para reducir warnings
    "verbosity": -100,

    "seed": PARAM["semilla_primigenia"],

    "max_depth": -1,  # -1 significa no limitar, por ahora lo dejo fijo
    "min_gain_to_split": 0.0,  # min_gain_to_split >= 0.0
    "min_sum_hessian_in_leaf": 0.001,  # min_sum_hessian_in_leaf >= 0.0
    "lambda_l1": 0.0,  # lambda_l1 >= 0.0
    "lambda_l2": 0.0,  # lambda_l2 >= 0.0
    "max_bin": 31,  # lo debo dejar fijo, no participa de la BO

    "bagging_fraction": 1.0,  # 0.0 < bagging_fraction <= 1.0
    "pos_bagging_fraction": 1.0,  # 0.0 < pos_bagging_fraction <= 1.0
    "neg_bagging_fraction": 1.0,  # 0.0 < neg_bagging_fraction <= 1.0
    "is_unbalance": False,
    "scale_pos_weight": 1.0,

    "drop_rate": 0.1,  # 0.0 < drop_rate <= 1.0
    "max_drop": 50,  # <=0 means no limit
    "skip_drop": 0.5,  # 0.0 <= skip_drop <= 1.0

    "extra_trees": False
}


# Aqui se definen los hiperparámetros de LightGBM que participan de la Bayesian Optimization

# In[11]:


# Aquí se definen los límites de los hiperparámetros para la Optimización Bayesiana
PARAM["hyperparametertuning"]["hs"] = {
    "num_iterations":   {"type": "int",   "bounds": (8, 2048)},
    "learning_rate":    {"type": "float", "bounds": (0.01, 0.3)},
    "feature_fraction": {"type": "float", "bounds": (0.1, 1.0)},
    "num_leaves":       {"type": "int",   "bounds": (8, 2048)},
    "min_data_in_leaf": {"type": "int",   "bounds": (1, 8000)}
}


# A mayor cantidad de hiperparámetros, se debe aumentar las iteraciones de la Bayesian Optimization
# <br> 30 es un valor muy tacaño, pero corre rápido
# <br> deberia partir de 50, alcanzando los 100 si se dispone de tiempo

# In[12]:


# Definir la cantidad de iteraciones para la optimización bayesiana
PARAM["hyperparametertuning"]["iteraciones"] = 50  # iteraciones bayesianas


# In[13]:


# particionar agrega una columna llamada fold a un dataset
#   que consiste en una particion estratificada segun agrupa
# particionar( data=dataset, division=c(70,30),
#  agrupa=clase_ternaria, seed=semilla)   crea una particion 70, 30

def particionar(data: pl.DataFrame, division, agrupa="", campo="fold", start=1, seed=None):
    """
    Replica la función R particionar() usando Polars.

    Args:
        data: DataFrame de Polars
        division: Lista con las proporciones de cada fold (ej: [70, 30])
        agrupa: Nombre de la columna para estratificación (opcional)
        campo: Nombre de la columna resultante (default: "fold")
        start: Número inicial para los folds (default: 1)
        seed: Semilla para reproducibilidad (opcional)

    Returns:
        DataFrame con la columna de folds agregada
    """
    if seed is not None:
        np.random.seed(seed)

    # Crear los bloques repetidos según la división
    bloques = np.concatenate([
        np.repeat(i, d) for i, d in enumerate(division, start=start)
    ])

    def asignar_fold(df):
        n = len(df)
        # Mezclar los bloques y repetir hasta cubrir todas las filas
        folds = np.resize(np.random.permutation(bloques), n)
        return df.with_columns(pl.Series(campo, folds))

    if agrupa and agrupa != "":
        # Estratificación por grupo
        result = (
            data.group_by(agrupa, maintain_order=True)
            .map_groups(asignar_fold)
        )
    else:
        # Sin estratificación
        n = data.height
        folds = np.resize(np.random.permutation(bloques), n)
        result = data.with_columns(pl.Series(campo, folds))

    return result


# In[15]:


def realidad_inicializar(pfuture: pl.DataFrame, pparam: dict) -> pl.DataFrame:
    """
    Inicializa el dataset de realidad para medir la ganancia.
    Replica la función R realidad_inicializar().

    Args:
        pfuture: DataFrame con los datos del futuro
        pparam: Diccionario con los parámetros (debe contener 'semilla_kaggle')

    Returns:
        DataFrame con las columnas necesarias y la partición aplicada
    """
    # Seleccionar solo las columnas necesarias (equivalente a pfuture[, list(numero_de_cliente, foto_mes, clase_ternaria)])
    drealidad = pfuture.select(["numero_de_cliente", "foto_mes", "clase_ternaria"])

    # Aplicar la partición usando la función particionar que creamos antes
    drealidad = particionar(
        data=drealidad,
        division=[3, 7],  # 30% y 70%
        agrupa="clase_ternaria",
        seed=pparam["semilla_kaggle"]
    )

    return drealidad


# In[17]:


# Evalúa la ganancia en los datos de la realidad usando polars

def realidad_evaluar(prealidad: pl.DataFrame, pprediccion: pl.DataFrame) -> dict:
    """
    Evalúa la ganancia en los datos de la realidad.
    Replica la función R realidad_evaluar().

    Args:
        prealidad: DataFrame con la realidad (debe tener columnas: numero_de_cliente, foto_mes, fold, clase_ternaria)
        pprediccion: DataFrame con las predicciones (debe tener columnas: numero_de_cliente, foto_mes, Predicted)

    Returns:
        dict: Diccionario con las ganancias 'public', 'private' y 'total'
    """
    # Join de la realidad con las predicciones (equivalente a prealidad[pprediccion, on=...])
    df_joined = prealidad.join(
        pprediccion.select(["numero_de_cliente", "foto_mes", "Predicted"]),
        on=["numero_de_cliente", "foto_mes"],
        how="left"
    )

    # Agrupar por fold, predicted, clase_ternaria y contar (equivalente a tbl <- prealidad[, list("qty"=.N), list(fold, predicted, clase_ternaria)])
    tbl = (
        df_joined
        .group_by(["fold", "Predicted", "clase_ternaria"])
        .agg(pl.len().alias("qty"))
    )

    # Calcular ganancia por registro
    tbl = tbl.with_columns(
        pl.when(pl.col("clase_ternaria") == "BAJA+2")
        .then(pl.col("qty") * 780000)
        .otherwise(pl.col("qty") * -20000)
        .alias("ganancia")
    )

    # Calcular métricas
    res = {}

    # Ganancia pública: fold==1 y predicted==1, dividido por 0.3
    public_ganancia = (
        tbl.filter((pl.col("fold") == 1) & (pl.col("Predicted") == 1))
        .select(pl.col("ganancia").sum())
        .item()
    )
    res["public"] = public_ganancia / 0.3 if public_ganancia is not None else 0

    # Ganancia privada: fold==2 y predicted==1, dividido por 0.7
    private_ganancia = (
        tbl.filter((pl.col("fold") == 2) & (pl.col("Predicted") == 1))
        .select(pl.col("ganancia").sum())
        .item()
    )
    res["private"] = private_ganancia / 0.7 if private_ganancia is not None else 0

    # Ganancia total: predicted==1
    total_ganancia = (
        tbl.filter(pl.col("Predicted") == 1)
        .select(pl.col("ganancia").sum())
        .item()
    )
    res["total"] = total_ganancia if total_ganancia is not None else 0

    return res


# ### Preprocesamiento

# In[19]:


# carpeta de trabajo

# Definir la carpeta de trabajo y crearla si no existe
base_path = "exp"
experimento_folder = f"HT{PARAM['experimento']}"
experimento_path = os.path.join(base_path, experimento_folder)

# Crear la carpeta del experimento si no existe
os.makedirs(experimento_path, exist_ok=True)


# In[20]:


# Lectura del dataset
dataset = pl.read_csv("datasets/competencia_01.csv.gz", infer_schema_length=100000)
dataset.head()


# In[22]:


dataset.shape

# Ordenar primero
dataset = dataset.sort(["numero_de_cliente", "foto_mes"])

# columnas numéricas (excluyendo id/tiempo/target)
excluir = {"numero_de_cliente", "foto_mes", "clase_ternaria"}
num_cols = [c for c, dt in zip(dataset.columns, dataset.dtypes)
            if dt in pl.NUMERIC_DTYPES and c not in excluir]

# lags 1 y 2 + deltas 1 y 2 por cliente
exprs = []
for c in num_cols:
    exprs += [
        pl.col(c).shift(1).over("numero_de_cliente").alias(f"{c}_lag1"),
        pl.col(c).shift(2).over("numero_de_cliente").alias(f"{c}_lag2"),
        (pl.col(c) - pl.col(c).shift(1).over("numero_de_cliente")).alias(f"{c}_delta1"),
        (pl.col(c) - pl.col(c).shift(2).over("numero_de_cliente")).alias(f"{c}_delta2"),
    ]

dataset = dataset.with_columns(exprs)

# Intentamos identificar quien recibe aguinaldo
# Creo un criterio de identificación de aguinaldo, y la modifico con un factor de proporcion

for var in num_cols:
    mediana_hist = dataset.filter(pl.col("foto_mes") < 202106)[var].median()
    mediana_junio = dataset.filter(pl.col("foto_mes") == 202106)[var].median()

    # Evitar división por cero
    if mediana_junio == 0 or mediana_junio is None:
        factor = 1.0
    else:
        factor = mediana_hist / mediana_junio

    # Corrige solo en junio 202106 para la columna correspondiente
    dataset = dataset.with_columns(
        pl.when(pl.col("foto_mes") == 202106)
        .then(pl.col(var) * factor)
        .otherwise(pl.col(var))
        .alias(var)
    )

# In[23]:


dataset_train = dataset.filter(pl.col("foto_mes").is_in(PARAM["train"]))
dataset_train.head()


# In[24]:


dataset_train['foto_mes'].value_counts()


# In[25]:


# paso la clase a binaria que tome valores {0,1} enteros
# BAJA+1 y BAJA+2 son 1, CONTINUA es 0
# a partir de ahora ya NO puedo cortar por prob(BAJA+2) > 1/40

dataset_train = dataset_train.with_columns(
    (pl.col("clase_ternaria").is_in(["BAJA+2", "BAJA+1"]).cast(pl.Int8)).alias("clase01")
)
dataset_train.head()


# In[26]:


dataset_train['clase01'].value_counts()


# In[27]:


# defino los datos que forma parte del training
# aqui se hace el undersampling de los CONTINUA
# notar que para esto utilizo la SEGUNDA semilla

np.random.seed(PARAM["semilla_primigenia"])

# Generar columna aleatoria "azar"
dataset_train = dataset_train.with_columns(
    pl.Series("azar", np.random.uniform(0, 1, dataset_train.height))
)

# Inicializar columna "training" en 0
dataset_train = dataset_train.with_columns(
    pl.lit(0).cast(pl.Int8).alias("training")
)

# Aplicar lógica de undersampling y selección de entrenamiento
mask = (
    pl.col("foto_mes").is_in(PARAM["train"])
    & (
        (pl.col("azar") <= PARAM["trainingstrategy"]["undersampling"])
        | (pl.col("clase_ternaria").is_in(["BAJA+1", "BAJA+2"]))
    )
)

dataset_train = dataset_train.with_columns(
    pl.when(mask)
      .then(1)
      .otherwise(pl.col("training"))
      .alias("training")
)
dataset_train.head()


# In[28]:


dataset_train['training'].value_counts()


# In[29]:


# los campos que se van a utilizar

excluir = {"clase_ternaria","clase01","azar","training","periodo0","periodo1","periodo2","numero_de_cliente"}
campos_buenos = [c for c in dataset_train.columns if c not in excluir]
logger.info(f"Campos para entrenamiento: {campos_buenos}")
logger.info(f"Total de campos: {len(campos_buenos)}")


# In[30]:


# dejo los datos en el formato que necesita LightGBM

# Filtramos las filas que entran al entrenamiento
train_mask = dataset_train["training"] == 1

# Extraemos features y labels en formato NumPy
X_train = dataset_train.filter(train_mask).select(campos_buenos).to_numpy()
y_train = dataset_train.filter(train_mask)["clase01"].to_numpy()

# Creamos el Dataset de LightGBM
dtrain = lgb.Dataset(
    data=X_train,
    label=y_train,
    free_raw_data=False
)

# Verificamos dimensiones
logger.info(f"Dimensiones del dataset de entrenamiento - nrow: {X_train.shape[0]}, ncol: {X_train.shape[1]}")


# In[31]:


# Ver value counts del dataset original filtrado
logger.info(f"Distribución por foto_mes en entrenamiento:\n{dataset_train.filter(train_mask)['foto_mes'].value_counts()}")


# ### Configuracion y Corrida Bayesian Optimization

# In[32]:


# -------------------------------
# 1. Función objetivo (equivalente a EstimarGanancia_AUC_lightgbm)
# -------------------------------
def objective(trial):
    # Extraer bounds desde PARAM
    hs = PARAM["hyperparametertuning"]["hs"]
    
    # Hiperparámetros variables (los que la BO va a optimizar)
    params_variables = {
        "learning_rate": trial.suggest_float("learning_rate", *hs["learning_rate"]["bounds"]),
        "feature_fraction": trial.suggest_float("feature_fraction", *hs["feature_fraction"]["bounds"]),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", *hs["min_data_in_leaf"]["bounds"]),
        "num_leaves": trial.suggest_int("num_leaves", *hs["num_leaves"]["bounds"]),
        "num_iterations": trial.suggest_int("num_iterations", *hs["num_iterations"]["bounds"]),
    }
    
    # Combinar parámetros fijos con variables
    params = {
        **PARAM["lgbm"]["param_fijos"],
        **params_variables,
        "deterministic": True,
    }
    
    # Ajustar nombre de parámetro para lgb.train API
    if "boosting" in params:
        params["boosting_type"] = params.pop("boosting")
    
    # Dividir datos en train y validation estratificado
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train, 
        y_train,
        test_size=PARAM["hyperparametertuning"]["validation_split"],
        stratify=y_train,  # mantener proporción de clases
        random_state=PARAM["semilla_primigenia"]
    )
    
    # Crear datasets de LightGBM
    dtrain_split = lgb.Dataset(
        data=X_train_split,
        label=y_train_split,
        free_raw_data=False
    )
    
    dval_split = lgb.Dataset(
        data=X_val_split,
        label=y_val_split,
        reference=dtrain_split,  # importante para mantener consistencia
        free_raw_data=False
    )
    
    # Entrenar modelo
    model = lgb.train(
        params=params,
        train_set=dtrain_split,
        valid_sets=[dval_split],
        valid_names=['validation'],
        callbacks=[lgb.log_evaluation(period=0)]  # desactiva logging por iteración
    )
    
    # Predecir en validation set
    y_pred = model.predict(X_val_split)
    
    # Calcular AUC en validation
    auc_val = roc_auc_score(y_val_split, y_pred)
    
    logger.info(f"[Trial {trial.number:03d}] AUC = {auc_val:.5f}")
    
    return auc_val


# In[33]:


# -------------------------------
# 2. Configuración de la búsqueda bayesiana
# -------------------------------

# --- crear el estudio Optuna ---
sampler = optuna.samplers.TPESampler(seed=PARAM["semilla_primigenia"])

study = optuna.create_study(
    direction="maximize",
    study_name=f"lightgbm_auc_HT{PARAM['experimento']}", 
    sampler=sampler
)

# -------------------------------
# 3. Ejecución de la optimización
# -------------------------------

n_trials_total = PARAM["hyperparametertuning"]["iteraciones"]

# Ejecuto directamente todos los trials de una vez,
# ya que no hay persistencia y los logs parciales no tienen sentido aquí
logger.info(f"Iniciando optimización de {n_trials_total} trials...")
study.optimize(
    objective,
    n_trials=PARAM["hyperparametertuning"]["iteraciones"],       # equivalente a PARAM$hyperparametertuning$iteraciones
    gc_after_trial=True
)
logger.info(f"Optimización completada con {len(study.trials)} trials.")


# In[35]:


# -------------------------------
# 4. Resultado final
# -------------------------------
logger.info("Mejores hiperparámetros:")
logger.info(study.best_params)
logger.info(f"Mejor AUC promedio: {study.best_value:.5f}")


# In[36]:


# Obtener los resultados de la optimización
df_bayesiana = study.trials_dataframe()

# Mostrar las columnas disponibles (hiperparámetros + AUC)
logger.info(f"Columnas disponibles en resultados BO: {df_bayesiana.columns.tolist()}")

# Ver las primeras filas
logger.info(f"Primeras filas de resultados BO:\n{df_bayesiana.head()}")


# In[37]:


# -------------------------------
# Guardado de resultados Bayesiana
# -------------------------------

# Agregar columna iter (número de iteración)
df_bayesiana["iter"] = range(1, len(df_bayesiana) + 1)

# Ordenar por AUC (valor) descendente
df_bayesiana = df_bayesiana.sort_values(by="value", ascending=False)

# Guardar log de resultados en el experimento
bo_log_path = os.path.join(experimento_path, "BO_log.txt")
df_bayesiana.to_csv(bo_log_path, sep="\t", index=False)
logger.info(f"Guardado log de la Bayesian Optimization en: {bo_log_path}")


# In[38]:


# Extraer los mejores hiperparámetros
best_params = study.best_params
best_auc = study.best_value

# Guardar mejores hiperparámetros y AUC en YAML
param_yml_path = os.path.join(experimento_path, "PARAM.yml")
PARAM_out = {
    "out": {
        "lgbm": {
            "mejores_hiperparametros": best_params,
            "y": float(best_auc)
        }
    }
}

with open(param_yml_path, "w") as f:
    yaml.safe_dump(PARAM_out, f, sort_keys=False)

logger.info(f"Mejor AUC: {best_auc:.5f}")
logger.info(f"Mejores hiperparámetros: {best_params}")
logger.info(f"Resultados guardados en: {param_yml_path}")


# ============================================================
# Entrenamiento para Evaluación sobre el Test/Futuro (No FINAL)
# ============================================================
#
# En esta sección, se entrena un ensemble de modelos LightGBM
# utilizando los mejores hiperparámetros encontrados y TODO el dataset
# original de training (NO incluye el "test"). Esto no es el entrenamiento
# "final" para submit, sino una instancia para evaluar el desempeño sobre el holdout
# (test real/futuro). El verdadero entrenamiento FINAL será el que se realice
# posteriormente incluyendo también el test.
#
# Comentarios y nombres de variables reflejan esta aclaración.
# ============================================================

# Variable binaria target
dataset = dataset.with_columns(
    (pl.col("clase_ternaria").is_in(["BAJA+2", "BAJA+1"]).cast(pl.Int8)).alias("clase01")
)

# --- Seleccionar Datos de Entrenamiento segun PARAM["train"] ---
dataset_train = dataset.filter(pl.col("foto_mes").is_in(PARAM["train"]))

resumen = (
    dataset_train
    .group_by("clase_ternaria")
    .len()
    .rename({"len": "N"})
    .sort("clase_ternaria")
)

logger.info(f"Resumen del dataset de entrenamiento para evaluación sobre test:\n{resumen}")

# --- Preparar datos para LightGBM ---
X_train_eval = (
    dataset_train
    .select(campos_buenos)
    # .with_columns(pl.all().cast(pl.Float32))  # opcional: forzar float
    .to_numpy()
)
y_train_eval = dataset_train["clase01"].to_numpy()

dtrain_eval = lgb.Dataset(
    data=X_train_eval,
    label=y_train_eval,
    free_raw_data=False
)

logger.info(f"Dimensiones para training de evaluación - nrow: {X_train_eval.shape[0]}, ncol: {X_train_eval.shape[1]}")

# --- Hiperparámetros combinando fijos + mejores encontrados ---
param_eval = {**PARAM["lgbm"]["param_fijos"], **best_params}

logger.info("Parámetros de LightGBM para evaluación sobre test (ensemble):")
for k, v in param_eval.items():
    logger.info(f" {k}: {v}")

# --- Ensemble Training (para test) ---
ensemble_seeds = PARAM["semillas_ensemble"]
modelos_ensemble = []

param_eval_normalizado = param_eval.copy()
param_eval_normalizado["min_data_in_leaf"] = round(
    param_eval["min_data_in_leaf"] / PARAM["trainingstrategy"]["undersampling"]
)

logger.info("Parámetros normalizados para ensemble (min_data_in_leaf ajustado):")
logger.info(f"min_data_in_leaf ajustado: {param_eval_normalizado['min_data_in_leaf']}")

for idx, seed in enumerate(ensemble_seeds):
    param_ens = param_eval_normalizado.copy()
    param_ens["seed"] = seed
    logger.info(f"Entrenando modelo {idx+1}/{len(ensemble_seeds)} para ensemble (evaluación test), seed={seed}")
    modelo = lgb.train(
        params=param_ens,
        train_set=dtrain_eval
    )
    modelos_ensemble.append(modelo)

# ============================================================
# Scoring sobre Test/Futuro
# ============================================================

# Preparo datos a predecir (conjunto "futuro" = test/horizonte holdout)
dfuture = dataset.filter(pl.col("foto_mes").is_in(PARAM["future"]))
X_predict_eval = dfuture.select(campos_buenos).to_numpy()

# Ensemble: promedio de predicciones
predicciones_ensemble = []
for modelo in modelos_ensemble:
    preds = modelo.predict(X_predict_eval)
    predicciones_ensemble.append(preds)
prediccion = np.mean(predicciones_ensemble, axis=0)

# Inicializar dataset de "realidad" para cálculo de métricas
drealidad = realidad_inicializar(dfuture, PARAM)
drealidad.shape

# Generar tabla de predicción para análisis y evaluación
tb_prediccion = (
    dfuture
    .select(["numero_de_cliente", "foto_mes"])
    .with_columns(pl.Series("prob", prediccion))
)

# ============================================================
# Simulación de Submit/Kaggle sobre el test/holdout (no real)
# ============================================================

logger.info(f"Cortes a evaluar: {PARAM['cortes']}")

tb_prediccion_sorted = tb_prediccion.sort("prob", descending=True)

for envios in PARAM["cortes"]:
    pred = (
        tb_prediccion_sorted
        .with_row_index("rn")
        .with_columns(
            pl.when(pl.col("rn") < envios).then(1).otherwise(0).alias("Predicted")
        )
        .drop("rn")
    )
    res = realidad_evaluar(drealidad, pred)
    logger.info(f"Envios={envios}\t TOTAL={res['total']}  Public={res['public']} Private={res['private']}")




# ## Production
# #### Final Training Dataset
# 
# Aqui esta la gran decision de en qué meses hago el Final Training
# <br> debo utilizar los mejores hiperparámetros que encontré en la  optimización bayesiana

# In[40]:

# clase01
dataset = dataset.with_columns(
    (pl.col("clase_ternaria").is_in(["BAJA+2", "BAJA+1"]).cast(pl.Int8)).alias("clase01")
)

# In[41]:

# --- Filtrar por los meses de entrenamiento final ---
dataset_train = dataset.filter(pl.col("foto_mes").is_in(PARAM["train_final"]))

resumen = (
    dataset_train
    .group_by("clase_ternaria")
    .len()
    .rename({"len": "N"})
    .sort("clase_ternaria")
)

logger.info(f"Resumen del dataset de entrenamiento final:\n{resumen}")

# In[42]:

# dejo los datos en el formato que necesita LightGBM
X_final = (
    dataset_train
    .select(campos_buenos)
    .to_numpy()
)
y_final = dataset_train["clase01"].to_numpy()

dtrain_final = lgb.Dataset(
    data=X_final,
    label=y_final,
    free_raw_data=False
)

logger.info(f"Dimensiones finales - nrow_final: {X_final.shape[0]}, ncol_final: {X_final.shape[1]}")

# #### Final Training Hyperparameters

# In[43]:

# Fusionar parámetros fijos con los óptimos encontrados
param_final = {**PARAM["lgbm"]["param_fijos"], **best_params}

logger.info("Parámetros finales de LightGBM:")
for k, v in param_final.items():
    logger.info(f" {k}: {v}")

# #### Training
# Genero el modelo final, siempre sobre TODOS los datos de final_train, sin hacer ningun tipo de undersampling de la clase mayoritaria y mucho menos cross validation.

# In[44]:

param_normalizado = param_final.copy()
param_normalizado["min_data_in_leaf"] = round(
    param_final["min_data_in_leaf"] / PARAM["trainingstrategy"]["undersampling"]
)
logger.info(" Parámetros normalizados:")
logger.info(f"min_data_in_leaf ajustado: {param_normalizado['min_data_in_leaf']}")

# Entrenar modelos de ensemble, uno por cada semilla
modelos_ensemble_final = []
for seed in PARAM["semillas_ensemble"]:
    param_seed = param_normalizado.copy()
    param_seed["seed"] = seed
    modelo = lgb.train(
        params=param_seed,
        train_set=dtrain_final
    )
    modelos_ensemble_final.append(modelo)
logger.info(f"Entrenados {len(modelos_ensemble_final)} modelos para el ensemble final")

# Toma el primer modelo para calcular importancias y guardar modelo de ejemplo
modelo_final = modelos_ensemble_final[0]

gain = modelo_final.feature_importance(importance_type="gain")
split = modelo_final.feature_importance(importance_type="split")
feat = modelo_final.feature_name()

tb_importancia = pl.DataFrame({
    "Feature": feat,
    "Gain": gain,
    "Split": split
})

tb_importancia = tb_importancia.with_columns(
    pl.Series("Feature", campos_buenos)
)

experimento_folder = f"exp{PARAM['experimento']}"
experimento_path = os.path.join(base_path, experimento_folder)
os.makedirs(experimento_path, exist_ok=True)
archivo_importancia = os.path.join(experimento_path, "impo.txt")
tb_importancia.write_csv(archivo_importancia, separator="\t")

# Guardar el primer modelo de ensemble (de referencia)
modelo_path = os.path.join(experimento_path, "modelo.txt")
modelo_final.save_model(modelo_path)
logger.info(f"Primer modelo del ensemble guardado en: {modelo_path}")

param_path = os.path.join(experimento_path, "PARAM.yml")
with open(param_path, "w", encoding="utf-8") as f:
    yaml.safe_dump(PARAM, f, sort_keys=False, allow_unicode=True)

logger.info(f"Experimento finalizado: {datetime.now().strftime('%a %b %d %X %Y')}")

# ## Kaggle Submission - ahora con ensemble

# In[56]:

# datos sin clase (kaggle, junio)
dfuture = dataset.filter(pl.col("foto_mes").is_in(PARAM["kaggle"]))

# Aplicar ensemble de modelos a los datos nuevos
X_future = dfuture.select(campos_buenos).to_numpy()
predicciones_ensemble = []
for modelo in modelos_ensemble_final:
    preds = modelo.predict(X_future)
    predicciones_ensemble.append(preds)
prediccion = np.mean(predicciones_ensemble, axis=0)

# dimensiones y tabla de foto_mes
logger.info(f"Dimensiones del dataset Kaggle: {dfuture.shape}")
logger.info(f"Distribución por foto_mes en Kaggle:\n{dfuture.group_by('foto_mes').len().rename({'len': 'N'}).sort('foto_mes')}")

# In[57]:

# inicilizo el dataset drealidad
drealidad = realidad_inicializar(dfuture, PARAM)
logger.info(f"Dimensiones del dataset de realidad: {drealidad.shape}")

# In[58]:

# tabla de predicción
tb_prediccion = (
    dfuture
    .select(["numero_de_cliente", "foto_mes"])
    .with_columns(pl.Series("prob", prediccion))
)

archivo_prediccion = os.path.join(experimento_path, "prediccion.txt")
tb_prediccion.write_csv(archivo_prediccion, separator="\t")
logger.info(f"Predicciones guardadas en: {archivo_prediccion}")

# In[59]:

# ordenar por probabilidad descendente
tb_prediccion_sorted = tb_prediccion.sort("prob", descending=True)

# crear carpeta
kaggle_path = os.path.join(experimento_path, "kaggle")
os.makedirs(kaggle_path, exist_ok=True)

for envios in PARAM["cortes"]:
    # Predicted = 1 para los primeros 'envios', 0 para el resto
    pred = (
        tb_prediccion_sorted
        .with_row_index("rn")
        .with_columns(
            pl.when(pl.col("rn") < envios).then(1).otherwise(0).alias("Predicted")
        )
        .select(["numero_de_cliente", "Predicted"])
    )

    archivo_kaggle = os.path.join(kaggle_path, f"KA{PARAM['experimento']}_{envios}.csv")
    pred.write_csv(archivo_kaggle)

logger.info(f"Archivos de Kaggle generados en: {kaggle_path}")