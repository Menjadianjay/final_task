from pymongo import MongoClient
import pandas as pd

def connect_mongo(db_name, collection_name):
    client = MongoClient("mongodb://localhost:27017/")
    db = client[db_name]
    return db[collection_name]

def enterSampleData(source_file, db_name, collection_name, chunk_size=500):
    total_inserted = 0
    try:
        client = MongoClient("mongodb://localhost:27017/")
        db = client[db_name]
        collection = db[collection_name]

        for chunk in pd.read_csv(source_file, chunksize=chunk_size):
            # Strip whitespace from column names
            chunk.columns = chunk.columns.str.strip()
            chunk.dropna(inplace=True)
            
            # Dynamically build astype dict from actual CSV columns
            dynamic_types = {col: str for col in chunk.columns}
            records = chunk.astype(dynamic_types).to_dict(orient='records')

            result = collection.insert_many(records, ordered=False)
            total_inserted += len(result.inserted_ids)

        print(f"Successfully inserted {total_inserted} records.")

    except FileNotFoundError:
        print(f"Error: File '{source_file}' not found.")
    except Exception as e:
        print(f"Insertion failed: {e}")