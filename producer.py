import pandas as pd
from kafka import KafkaProducer
import json
import time

# Chemins vers les fichiers CSV
test_file = 'churn-bigml-20.csv'

# Configuration du producteur Kafka
producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    key_serializer=lambda v: str(v).encode('utf-8'),
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

def on_send_success(record_metadata):
    print(f'Successfully sent message to {record_metadata.topic} partition {record_metadata.partition} offset {record_metadata.offset}')

def on_send_error(excp):
    print(f'Error sending message: {excp}')

def send_csv_to_kafka(file_path, topic):
    df = pd.read_csv(file_path)
    print("Colonnes disponibles dans le CSV:", df.columns)  # Afficher les colonnes disponibles
    for _, row in df.iterrows():
        # Utiliser une colonne existante comme clé de partitionnement
        key = row['id'] if 'id' in df.columns else str(_)  # Utiliser 'id' ou l'index de la ligne
        value = row.to_dict()
        future = producer.send(topic, key=key, value=value)
        future.add_callback(on_send_success).add_errback(on_send_error)
        time.sleep(0.1)  # Simuler un flux de données

# Envoyer les données d'entraînement et de test à Kafka
send_csv_to_kafka(test_file, 'test_topic')

# Assurez-vous que tous les messages sont envoyés avant de terminer
producer.flush()
