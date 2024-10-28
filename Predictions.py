from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler, MinMaxScaler
from pyspark.ml.classification import RandomForestClassificationModel
from pyspark.ml.evaluation import BinaryClassificationEvaluator


# Créer une session Spark
spark = SparkSession.builder \
    .appName("MongoDBToSpark") \
    .config('spark.jars.packages', 
            'org.mongodb.spark:mongo-spark-connector_2.12:10.3.0') \
    .config("spark.mongodb.read.connection.uri", "mongodb://localhost:27017/essaie.test") \
    .config("spark.mongodb.write.connection.uri", "mongodb://localhost:27017/essaie.predictions") \
    .getOrCreate()

# Configurer le niveau de log pour afficher uniquement les erreurs
spark.sparkContext.setLogLevel("ERROR")

# Lire les données depuis MongoDB
test_data = spark.read \
    .format("mongodb") \
    .option("spark.mongodb.input.uri", "mongodb://localhost:27017/essaie.test") \
    .load()

test_data.printSchema()
test_data.show(5)


def pretraitement(df):
  # Convertir la colonne "Churn" en StringType
  df = df.withColumn("Churn", df["Churn"].cast("string"))
  # Encoder la variable cible "Churn"
  indexer = StringIndexer(inputCol="Churn", outputCol="Label")
  df = indexer.fit(df).transform(df)
  
  # Sélectionner les colonnes pour les caractéristiques
  feature_cols = ['State', 'Account length', 'Area code', 'International plan', 'Voice mail plan',
                  'Number vmail messages', 'Total day minutes', 'Total day calls', 'Total day charge',
                  'Total eve minutes', 'Total eve calls', 'Total eve charge',
                  'Total night minutes', 'Total night calls', 'Total night charge',
                  'Total intl minutes', 'Total intl calls', 'Total intl charge',
                  'Customer service calls']
  
  # Supprimer les valeurs manquantes
  df = df.na.drop()
  
  # Encoder les variables catégoriques en numériques
  indexers = [StringIndexer(inputCol=col, outputCol=col+"_index").fit(df) for col in ['State', 'International plan', 'Voice mail plan']]
  
  pipeline = Pipeline(stages=indexers)
  
  df = pipeline.fit(df).transform(df)
  
  assembler = VectorAssembler(inputCols=[col+"_index" if col in ['State', 'International plan', 'Voice mail plan'] else col for col in feature_cols], outputCol="features")
  
  df = assembler.transform(df)
  
  # Initialiser le MinMaxScaler
  scaler = MinMaxScaler(inputCol="features", outputCol="scaled_features")
  
  # Calculer les statistiques de résumé et normaliser les caractéristiques
  scaler_model = scaler.fit(df)
  df = scaler_model.transform(df)
  return df

load_model = RandomForestClassificationModel.load('random_forest')
test = pretraitement(test_data)
test_predictions = load_model.transform(test)
evaluator_rf = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction", labelCol="Label")

# Évaluation sur les données de test
test_accuracy = evaluator_rf.evaluate(test_predictions)
print(f"Accuracy on test set: {test_accuracy}")

# Afficher les résultats de prédiction
final_df = test_predictions.select(
    col("State"),
    col("Account length"),
    col("Area code"),
    col("International plan"),
    col("Voice mail plan"),
    col("Number vmail messages"),
    col("Total day minutes"),
    col("Total day calls"),
    col("Total day charge"),
    col("Total eve minutes"),
    col("Total eve calls"),
    col("Total eve charge"),
    col("Total night minutes"),
    col("Total night calls"),
    col("Total night charge"),
    col("Total intl minutes"),
    col("Total intl calls"),
    col("Total intl charge"),
    col("Customer service calls"),
    col("Label"),
    col("Prediction")
)

final_df.show(10, truncate=False)  


final_df.write \
    .format("mongodb") \
    .option("spark.mongodb.output.uri", "mongodb://localhost:27017/essaie.predictions") \
    .mode("overwrite") \
    .save()
  
# Arrêter la session Spark
spark.stop()