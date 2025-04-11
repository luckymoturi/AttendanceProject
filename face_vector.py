import psycopg2
import numpy as np
from typing import List, Dict, Optional
from psycopg2.extras import execute_values
from datetime import datetime

class FaceEmbeddingDB:
    def __init__(self, db_params: Dict[str, str]):
        self.db_params = db_params
        self.conn = None
        self.connect()
        self.create_tables()

    def connect(self):
        try:
            self.conn = psycopg2.connect(**self.db_params)
            print("Successfully connected to the database")
        except Exception as e:
            print(f"Error connecting to database: {e}")
            raise

    def create_tables(self):
        create_tables_query = """
        CREATE TABLE IF NOT EXISTS face_embeddings (
            id SERIAL PRIMARY KEY,
            name VARCHAR(255) NOT NULL UNIQUE,
            embedding vector(128) NOT NULL,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS attendance (
            id SERIAL PRIMARY KEY,
            user_name VARCHAR(255) NOT NULL,
            event_type VARCHAR(10) NOT NULL,
            event_time TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            latitude FLOAT NOT NULL,
            longitude FLOAT NOT NULL,
            FOREIGN KEY (user_name) REFERENCES face_embeddings (name) ON DELETE CASCADE
        );
        """
        with self.conn.cursor() as cur:
            cur.execute(create_tables_query)
            self.conn.commit()

    def store_embedding(self, name: str, embedding: np.ndarray) -> bool:
        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO face_embeddings (name, embedding)
                    VALUES (%s, %s)
                    ON CONFLICT (name) DO UPDATE
                    SET embedding = EXCLUDED.embedding
                """, (name, embedding.tolist()))
                self.conn.commit()
                return True
        except Exception as e:
            print(f"Error storing embedding: {e}")
            self.conn.rollback()
            return False

    def store_multiple_embeddings(self, embeddings_data: List[Dict[str, any]]) -> bool:
        try:
            with self.conn.cursor() as cur:
                embeddings_values = [
                    (data['name'], data['embedding'].tolist())
                    for data in embeddings_data
                ]
                execute_values(cur, """
                    INSERT INTO face_embeddings (name, embedding)
                    VALUES %s
                    ON CONFLICT (name) DO UPDATE
                    SET embedding = EXCLUDED.embedding
                """, embeddings_values)
                self.conn.commit()
                return True
        except Exception as e:
            print(f"Error storing multiple embeddings: {e}")
            self.conn.rollback()
            return False

    def retrieve_all_data(self) -> List[Dict[str, any]]:
        try:
            with self.conn.cursor() as cur:
                query = """
                    SELECT
                        fe.id,
                        fe.name,
                        fe.embedding,
                        fe.created_at,
                        (
                            SELECT event_time
                            FROM attendance
                            WHERE user_name = fe.name AND event_type = 'checkin'
                            ORDER BY event_time DESC
                            LIMIT 1
                        ) AS latest_checkin,
                        (
                            SELECT event_time
                            FROM attendance
                            WHERE user_name = fe.name AND event_type = 'checkout'
                            ORDER BY event_time DESC
                            LIMIT 1
                        ) AS latest_checkout
                    FROM face_embeddings fe;
                """
                cur.execute(query)
                results = cur.fetchall()
                return [
                    {
                        "id": row[0],
                        "name": row[1],
                        "embedding": row[2],
                        "created_at": row[3],
                        "latest_checkin": row[4],
                        "latest_checkout": row[5]
                    }
                    for row in results
                ]
        except Exception as e:
            print(f"Error retrieving data with check-in/check-out: {e}")
            return []

    def has_checked_in_today(self, user_name: str) -> bool:
        today = datetime.now().date()
        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    SELECT 1
                    FROM attendance
                    WHERE user_name = %s
                      AND event_type = 'checkin'
                      AND DATE(event_time) = %s
                """, (user_name, today))
                return cur.fetchone() is not None
        except Exception as e:
            print(f"Error checking if user has checked in today: {e}")
            return False

    def has_checked_out_today(self, user_name: str) -> bool:
        today = datetime.now().date()
        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    SELECT 1
                    FROM attendance
                    WHERE user_name = %s
                      AND event_type = 'checkout'
                      AND DATE(event_time) = %s
                """, (user_name, today))
                return cur.fetchone() is not None
        except Exception as e:
            print(f"Error checking if user has checked out today: {e}")
            return False

    def delete_tables(self):
        delete_tables_query = "DROP TABLE IF EXISTS face_embeddings, attendance CASCADE;"
        try:
            with self.conn.cursor() as cur:
                cur.execute(delete_tables_query)
                self.conn.commit()
                print("Successfully deleted the tables")
        except Exception as e:
            print(f"Error deleting tables: {e}")
            self.conn.rollback()

    def get_user_attendance_report(self, user_name: str) -> List[Dict[str, any]]:
        try:
            with self.conn.cursor() as cur:
                query = """
                    WITH daily_attendance AS (
                        SELECT
                            DATE(event_time) as attendance_date,
                            MAX(CASE WHEN event_type = 'checkin' THEN event_time END) as checkin_time,
                            MAX(CASE WHEN event_type = 'checkout' THEN event_time END) as checkout_time
                        FROM attendance
                        WHERE user_name = %s
                        GROUP BY DATE(event_time)
                    )
                    SELECT
                        attendance_date,
                        checkin_time,
                        checkout_time
                    FROM daily_attendance
                    ORDER BY attendance_date DESC;
                """
                cur.execute(query, (user_name,))
                results = cur.fetchall()

                return [
                    {
                        "date": row[0],
                        "checkin_time": row[1],
                        "checkout_time": row[2]
                    }
                    for row in results
                ]
        except Exception as e:
            print(f"Error retrieving user attendance report: {e}")
            return []

    def embedding_exists(self, embedding: np.ndarray) -> bool:
        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    SELECT 1
                    FROM face_embeddings
                    WHERE embedding = %s
                """, (embedding.tolist(),))
                return cur.fetchone() is not None
        except Exception as e:
            print(f"Error checking if embedding exists: {e}")
            return False

    def delete_user(self, name: str) -> bool:
        try:
            with self.conn.cursor() as cur:
                cur.execute("DELETE FROM face_embeddings WHERE name = %s", (name,))
                self.conn.commit()
                if cur.rowcount > 0:
                    print(f"User '{name}' deleted successfully.")
                    return True
                else:
                    print(f"User '{name}' not found.")
                    return False
        except Exception as e:
            print(f"Error deleting user '{name}': {e}")
            self.conn.rollback()
            return False

    def vector_search(self, encoding: np.ndarray) -> List[Dict[str, any]]:
        try:
            with self.conn.cursor() as cur:
                query = """
                    SELECT id, name, embedding, created_at,
                    1 - (embedding <=> %s::vector) as similarity
                    FROM face_embeddings
                    WHERE 1 - (embedding <=> %s::vector) > %s
                    ORDER BY embedding <=> %s::vector
                    LIMIT 5;
                """
                cur.execute(query, (encoding.tolist(), encoding.tolist(), 0.9, encoding.tolist()))
                results = cur.fetchall()
                return [
                    {
                        "id": result[0],
                        "name": result[1],
                        "embedding": result[2],
                        "created_at": result[3],
                        "similarity": result[4]
                    }
                    for result in results
                ]
        except Exception as e:
            print(f"Error performing vector search: {e}")
            return []

    def log_attendance(self, user_name: str, event_type: str, latitude: float, longitude: float) -> bool:
        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO attendance (user_name, event_type, latitude, longitude)
                    VALUES (%s, %s, %s, %s)
                """, (user_name, event_type, latitude, longitude))
                self.conn.commit()
                return True
        except Exception as e:
            print(f"Error logging attendance: {e}")
            self.conn.rollback()
            return False

    def retrieve_attendance(self, user_name: str) -> List[Dict[str, any]]:
        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    SELECT event_type, event_time
                    FROM attendance
                    WHERE user_name = %s
                    ORDER BY event_time DESC
                """, (user_name,))
                attendance_records = cur.fetchall()
                return [
                    {"event_type": row[0], "event_time": row[1]}
                    for row in attendance_records
                ]
        except Exception as e:
            print(f"Error retrieving attendance: {e}")
            return []

    def close(self):
        if self.conn:
            self.conn.close()
