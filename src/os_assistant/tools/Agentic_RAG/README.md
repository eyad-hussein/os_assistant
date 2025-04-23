#### 1. Initialize the Database

```bash
python main.py init --log-file users_logs.json
```

**Required Parameters:**
- `--log-file PATH`: Path to the log JSON file

**Optional Parameters:**
- `--chunk-size INT`: Text chunk size in characters (default: 200)
- `--overlap FLOAT`: Chunk overlap as fraction between 0.0-1.0 (default: 0.2)
- `--force`: Force rebuild the database from scratch, even if already initialized


#### 2. Search for Logs

```bash
python main.py search "password"
```

**Required Parameters:**
- `QUERY`: Search query text (positional argument)

**Optional Parameters:**
- `--top-k INT`: Number of top results to return (default: 5)

#### 3. View Database Contents

```bash
python test.py
```

**Optional Parameters:**
- `--db-path PATH`: Path to SQLite database (default: from config)
- `--schema`: Show database schema details
- `--log INT`: Show chunks for specific log number only
- `--limit INT`: Maximum number of rows to display (default: 10, use 0 for all)
- `--show-embeddings`: Include embedding vector previews

 
