import sqlite3
import os

db_path = "e:/repos/crawl/.gitnexus/lbug"
if os.path.exists(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()
    print("Tables:", tables)
    for t in tables:
        cursor.execute(f"SELECT COUNT(*) FROM {t[0]}")
        print(f"  {t[0]}: {cursor.fetchone()[0]} rows")
    conn.close()
else:
    print("Database not found")