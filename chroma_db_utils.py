import pandas as pd
import chromadb
import os
import shutil
from dotenv import load_dotenv

load_dotenv()

def get_next_id():
    """Get the next ID by finding the maximum existing ID and adding 1"""
    try:
        # Get existing collection to find max ID
        collection = get_or_create_collection()
        # Get all documents to find max ID
        results = collection.get()
        if results['ids']:
            max_id = max(int(id) for id in results['ids'])
            return max_id + 1
        else:
            return 1
    except Exception as e:
        print(f"Error getting next ID: {e}")
        return 1

def get_or_create_collection():
    """Get existing collection or create a new one - single persistent collection"""
    client = chromadb.PersistentClient(path=os.getenv('CHROMA_DB_PATH'))
    
    try:
        collection = client.get_collection("customer_service_kb")
        print("Using existing ChromaDB collection...")
        return collection
    except Exception as e:
        print("Creating new ChromaDB collection...")
        collection = client.create_collection(
            name="customer_service_kb",
            metadata={"description": "Customer Service Knowledge Base"}
        )
        return collection

def add_to_chroma_only(case_id, topic_name, description, sentiment, solution):
    """Add a single case to ChromaDB only (not CSV)"""
    try:
        collection = get_or_create_collection()
        
        document_text = f"Topic: {topic_name}. Query: {description}. Solution: {solution}"
        
        collection.add(
            documents=[document_text],
            metadatas=[{
                "id": str(case_id),
                "topic_name": topic_name,
                "description": description,
                "sentiment": sentiment,
                "solution": solution,
                "source": "human_approved"
            }],
            ids=[str(case_id)]
        )
        print(f"Successfully added case {case_id} to ChromaDB")
        return True
    except Exception as e:
        print(f"Error adding to ChromaDB: {e}")
        return False

def load_csv_to_chroma(csv_file_path, batch_size=100):
    """Load data from CSV file into ChromaDB using batch processing"""
    try:
        # Check if CSV file exists
        if not os.path.exists(csv_file_path):
            print(f"CSV file not found: {csv_file_path}")
            return False
        
        # Read CSV file
        df = pd.read_csv(csv_file_path)
        print(f"Loaded {len(df)} records from CSV")
        
        # Check required columns
        required_columns = ['id', 'topic_name', 'description', 'overall_sentiment', 'solution']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Missing required columns in CSV: {missing_columns}")
            return False
        
        collection = get_or_create_collection()
        total_added = 0
        
        # Process in batches for better performance
        for i in range(0, len(df), batch_size):
            batch = df.iloc[i:i + batch_size]
            documents = []
            metadatas = []
            ids = []
            
            for _, row in batch.iterrows():
                try:
                    document_text = f"Topic: {row['topic_name']}. Query: {row['description']}. Solution: {row['solution']}"
                    
                    documents.append(document_text)
                    metadatas.append({
                        "id": str(row['id']),
                        "topic_name": row['topic_name'],
                        "description": row['description'],
                        "sentiment": row['overall_sentiment'],
                        "solution": row['solution'],
                        "source": "csv_import"
                    })
                    ids.append(str(row['id']))
                    
                except Exception as e:
                    print(f"Error preparing record ID {row['id']}: {e}")
                    continue
            
            # Add the entire batch at once
            if documents:
                try:
                    collection.add(
                        documents=documents,
                        metadatas=metadatas,
                        ids=ids
                    )
                    total_added += len(documents)
                    print(f"Added batch {i//batch_size + 1}: {len(documents)} records (Total: {total_added})")
                except Exception as e:
                    print(f"Error adding batch {i//batch_size + 1}: {e}")
                    # Fallback: try adding records individually if batch fails
                    for j, (doc, meta, id_val) in enumerate(zip(documents, metadatas, ids)):
                        try:
                            collection.add(
                                documents=[doc],
                                metadatas=[meta],
                                ids=[id_val]
                            )
                            total_added += 1
                        except Exception as e2:
                            print(f"Error adding individual record ID {meta['id']}: {e2}")
        
        print(f"Successfully loaded {total_added} records from CSV to ChromaDB")
        return total_added > 0
        
    except Exception as e:
        print(f"Error loading CSV to ChromaDB: {e}")
        return False

def is_chroma_empty():
    """Check if ChromaDB collection is empty"""
    try:
        collection = get_or_create_collection()
        results = collection.get()
        return len(results['ids']) == 0
    except Exception as e:
        print(f"Error checking ChromaDB: {e}")
        return True

def cleanup_old_chroma_data():
    """Clean up old ChromaDB data to prevent accumulation - now uses single collection"""
    chroma_path = os.getenv('CHROMA_DB_PATH')
    if os.path.exists(chroma_path):
        for item in os.listdir(chroma_path):
            item_path = os.path.join(chroma_path, item)
            # Remove any temporary or backup files, but keep the main collection
            if item.endswith('.bak') or item.startswith('temp_') or item.startswith('chroma-'):
                try:
                    if os.path.isfile(item_path):
                        os.remove(item_path)
                    elif os.path.isdir(item_path):
                        shutil.rmtree(item_path)
                    print(f"Cleaned up: {item_path}")
                except Exception as e:
                    print(f"Error cleaning up {item_path}: {e}")

def get_all_records():
    """Get all records from ChromaDB for display"""
    try:
        collection = get_or_create_collection()
        results = collection.get()
        return results
    except Exception as e:
        print(f"Error getting all records: {e}")
        return None

if __name__ == "__main__":
    # Define your CSV file path
    CSV_FILE_PATH = "data/customer_service_data.csv"
    
    # Initialize the collection
    collection = get_or_create_collection()
    print("ChromaDB collection initialized successfully")
    
    # Check if ChromaDB is empty and load CSV data if it is
    if is_chroma_empty():
        print("ChromaDB collection is empty. Loading data from CSV...")
        start_time = pd.Timestamp.now()
        if load_csv_to_chroma(CSV_FILE_PATH, batch_size=100):  # Adjust batch_size as needed
            end_time = pd.Timestamp.now()
            duration = (end_time - start_time).total_seconds()
            print(f"CSV data successfully loaded into ChromaDB in {duration:.2f} seconds")
        else:
            print("Failed to load CSV data into ChromaDB")
    else:
        print("ChromaDB already contains data. Skipping CSV import.")
    
    # Perform cleanup
    cleanup_old_chroma_data()