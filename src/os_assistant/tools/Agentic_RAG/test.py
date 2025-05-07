import sqlite3
import json
import os
import argparse
from tabulate import tabulate

def connect_to_db(db_path):
    """Connect to the database and return connection and cursor."""
    if not os.path.exists(db_path):
        print(f"Database file not found at: {db_path}")
        print("Please run 'python main.py init --log-file users_logs.json' first to create the database.")
        exit(1)
    
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row  
    cursor = conn.cursor()
    return conn, cursor

def show_tables(cursor):
    """Show all tables in the database."""
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    
    if not tables:
        print("No tables found in the database.")
        return
    
    print("\n=== Tables in Database ===")
    for i, table in enumerate(tables):
        print(f"{i+1}. {table['name']}")
    return [t['name'] for t in tables]

def show_table_schema(cursor, table_name):
    """Show the schema of a specific table."""
    try:
        cursor.execute(f"PRAGMA table_info({table_name});")
        columns = cursor.fetchall()
        
        if not columns:
            print(f"No columns found for table '{table_name}'")
            return
        
        print(f"\n=== Schema for Table '{table_name}' ===")
        headers = ["Column ID", "Name", "Type", "NotNull", "Default Value", "Primary Key"]
        rows = [[col['cid'], col['name'], col['type'], col['notnull'], col['dflt_value'], col['pk']] for col in columns]
        print(tabulate(rows, headers=headers, tablefmt="pretty"))
    except sqlite3.OperationalError as e:
        print(f"Error: {e}")

def show_log_chunks(cursor, limit=10, show_embeddings=False, log_number=None):
    """Show log chunks from the database."""
    try:
        query = "SELECT id, log_number, chunk_number, chunk_text, timestamp, embedding FROM log_chunks"
        params = []
        
        if log_number is not None:
            query += " WHERE log_number = ?"
            params.append(log_number)
        
        query += " ORDER BY log_number DESC, chunk_number ASC"
        
        if limit:
            query += f" LIMIT {limit}"
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        if not rows:
            print("No log chunks found in the database.")
            return
        
        print(f"\n=== Log Chunks {f'for Log #{log_number}' if log_number else ''} ===")
        
        if show_embeddings:
            headers = ["ID", "Log #", "Chunk #", "Timestamp", "Text", "Embedding"]
            table_rows = []
            for row in rows:
                embedding = json.loads(row['embedding']) if row['embedding'] else []
                embedding_preview = str(embedding[:3]) + "..." if embedding else "None"
                table_rows.append([
                    row['id'], 
                    row['log_number'], 
                    row['chunk_number'], 
                    row['timestamp'], 
                    row['chunk_text'], 
                    embedding_preview
                ])
        else:
            headers = ["ID", "Log #", "Chunk #", "Timestamp", "Text"]
            table_rows = []
            for row in rows:
                table_rows.append([
                    row['id'], 
                    row['log_number'], 
                    row['chunk_number'], 
                    row['timestamp'], 
                    row['chunk_text']
                ])
        
        print(tabulate(table_rows, headers=headers, tablefmt="pretty"))
        print(f"\nShowing {len(rows)} of potentially more rows" if limit else "")
        
    except sqlite3.OperationalError as e:
        print(f"Error: {e}")

def main():
    parser = argparse.ArgumentParser(
        description="SQLite Database Viewer for Agentic RAG",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--db-path", 
        type=str, 
        metavar="PATH",
        help="Path to SQLite database"
    )
    parser.add_argument(
        "--schema", 
        action="store_true", 
        help="Show table schema details"
    )
    parser.add_argument(
        "--log", 
        type=int,
        metavar="INT",
        help="Show chunks for specific log number only"
    )
    parser.add_argument(
        "--limit", 
        type=int, 
        default=10,
        metavar="INT",
        help="Maximum number of rows to display (use 0 for all)"
    )
    parser.add_argument(
        "--show-embeddings", 
        action="store_true", 
        help="Include embedding vector previews"
    )
    
    args = parser.parse_args()
    print(args.db_path)
    conn, cursor = connect_to_db(args.db_path)
    
    try:
        tables = show_tables(cursor)
        
        if args.schema and tables:
            for table in tables:
                show_table_schema(cursor, table)
        
        limit = args.limit if args.limit > 0 else None
        
        if 'log_chunks' in tables:
            show_log_chunks(cursor, limit, args.show_embeddings, args.log)
    finally:
        conn.close()

if __name__ == "__main__":
    main()
