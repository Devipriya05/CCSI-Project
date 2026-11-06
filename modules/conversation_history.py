import os
import sqlite3
from contextlib import closing

DB_PATH = "database/conversations.db"


def initialize_database():
    os.makedirs(os.path.dirname(DB_PATH) or ".", exist_ok=True)
    with closing(sqlite3.connect(DB_PATH)) as conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id TEXT, customer_id TEXT, message_id TEXT,
            sequence_number INTEGER, speaker TEXT, message TEXT, timestamp TEXT
        )""")
        conn.commit()


def save_conversation(messages, customer_id):
    if not messages:
        return
    initialize_database()
    rows = [(
        m.get("conversation_id"), customer_id, m.get("message_id"),
        m.get("sequence_number"), m.get("speaker"), m.get("message"), m.get("timestamp")
    ) for m in messages]
    with closing(sqlite3.connect(DB_PATH)) as conn:
        conn.executemany("""INSERT INTO conversations
            (conversation_id, customer_id, message_id, sequence_number, speaker, message, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?)""", rows)
        conn.commit()


def get_customer_history(customer_id):
    initialize_database()
    with closing(sqlite3.connect(DB_PATH)) as conn:
        return conn.execute("""SELECT conversation_id, message_id, sequence_number,
            speaker, message, timestamp FROM conversations
            WHERE customer_id = ? ORDER BY id ASC""", (customer_id,)).fetchall()


def get_contact_statistics(customer_id):
    initialize_database()
    with closing(sqlite3.connect(DB_PATH)) as conn:
        total, customer_total = conn.execute("""SELECT
            COUNT(DISTINCT conversation_id),
            COUNT(DISTINCT CASE WHEN speaker = 'Customer' THEN conversation_id END)
            FROM conversations WHERE customer_id = ?""", (customer_id,)).fetchone()
    return {"total_conversations": total or 0, "customer_contacted_conversations": customer_total or 0}
