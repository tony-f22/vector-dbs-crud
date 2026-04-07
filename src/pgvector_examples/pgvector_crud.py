import psycopg2
from pgvector.psycopg2 import register_vector

# from pg_embedding_util import generate_embeddings
from sentence_transformers import SentenceTransformer


class PgVectorCRUD:
    """
    Class to handle CRUD operations on a PostgreSQL database using pgvector.
    """

    def __init__(self, user: str, password: str, host: str, port: int, database: str):
        """
        Initialize the PgVectorCRUD object with database credentials.

        Args:
            user (str): The PostgreSQL username.
            password (str): The PostgreSQL password.
            host (str): The PostgreSQL host.
            port (int): The PostgreSQL port.
            database (str): The name of the PostgreSQL database.
        """
        self.user = user
        self.password = password
        self.host = host
        self.port = port
        self.database = database

    def connect_db(self):
        """
        Establish a connection to the PostgreSQL database.

        Returns:
            conn: A connection object to interact with the PostgreSQL database.
            cur: A cursor object to execute SQL commands.
        """
        conn = psycopg2.connect(
            user=self.user,
            password=self.password,
            host=self.host,
            port=self.port,
            database=self.database,
        )
        # Register the vector type with psycopg2
        register_vector(conn)
        cur = conn.cursor()
        return conn, cur

    # ================================
    # CREATE Operation
    # ================================
    def create_items(self, model: SentenceTransformer, sentences: list[str]) -> None:
        """
        Insert new sentences with their embeddings into the items table.

        Args:
            sentences (List[str]): A list of sentences to be inserted into the database.
        """
        conn, cur = self.connect_db()
        try:
            for sentence in sentences:
                embedding = model.encode(sentence)
                cur.execute("INSERT INTO items (content, embedding) VALUES (%s, %s)", (sentence, embedding))

            # Commit the transaction to save the changes
            conn.commit()
            print("Sentences inserted successfully")
        except Exception as e:
            print("Error during insertion:", str(e))
        finally:
            cur.close()
            conn.close()

    # ================================
    # READ Operation (Search)
    # ================================
    def read_similar_items(self, model: SentenceTransformer, query: str, limit: int) -> None:
        """
        Perform a cosine similarity search for the query and return similar items.

        Args:
            query (str): The query text to search for similar items.
            limit (int): The number of top results to return.
        """
        conn, cur = self.connect_db()
        try:
            query_embedding = model.encode(query)

            # Perform a cosine similarity search
            cur.execute(
                """SELECT id, content, 1 - (embedding <=> %s) AS cosine_similarity
                   FROM items
                   ORDER BY cosine_similarity DESC LIMIT %s""",
                (query_embedding, limit),
            )

            # Fetch and print the result
            print(f"Query: '{query}'")
            print("Most similar sentences:")
            for row in cur.fetchall():
                print(f"ID: {row[0]}, CONTENT: {row[1]}, Cosine Similarity: {row[2]}")
        except Exception as e:
            print("Error during read query:", str(e))
        finally:
            cur.close()
            conn.close()

    # ================================
    # UPDATE Operation
    # ================================
    def update_item(self, model: SentenceTransformer, item_id: int, new_content: str) -> None:
        """
        Update the content and embedding of an item in the database by its ID.

        Args:
            item_id (int): The ID of the item to be updated.
            new_content (str): The new content to update.
        """
        conn, cur = self.connect_db()
        try:
            new_embedding = model.encode(new_content)

            # Update the item's content and embedding in the table
            cur.execute(
                "UPDATE items SET content = %s, embedding = %s WHERE id = %s",
                (new_content, new_embedding, item_id),
            )

            # Commit the transaction to save the changes
            conn.commit()
            print(f"Item with ID {item_id} updated successfully.")
        except Exception as e:
            print("Error during update:", str(e))
        finally:
            cur.close()
            conn.close()

    # ================================
    # DELETE Operation
    # ================================
    def delete_item(self, item_id: int) -> None:
        """
        Delete an item from the database by its ID.

        Args:
            item_id (int): The ID of the item to be deleted.
        """
        conn, cur = self.connect_db()
        try:
            cur.execute("DELETE FROM items WHERE id = %s", (item_id,))

            # Commit the transaction to save the changes
            conn.commit()
            print(f"Item with ID {item_id} deleted successfully.")
        except Exception as e:
            print("Error during deletion:", str(e))
        finally:
            cur.close()
            conn.close()

    # ================================
    # Index Creation
    # ================================
    def create_index(self, index_type: str = "hnsw", distance_op: str = "cosine_distance") -> None:
        """
        Create an index on the embedding column.
        
        Args:
            index_type: Either "hnsw" or "ivfflat"
            distance_op: Either "cosine", "l2", or "ip" (inner product)
        """
        conn, cur = self.connect_db()
        try:
            ops_map = {
                "cosine_distance": "vector_cosine_ops",
                "l2": "vector_l2_ops",
                "inner_product": "vector_ip_ops"
            }

            ops = ops_map.get(distance_op, "vector_cosine_ops")

            if index_type == "hnsw":
                cur.execute(f"CREATE INDEX IF NOT EXISTS items_embedding_idx ON items USING hnsw (embedding {ops})")
            elif index_type == "ivfflat":
                cur.execute(
                    f"""
                        CREATE INDEX IF NOT EXISTS items_embedding_idx ON items USING ivfflat (embedding {ops}) WITH (lists = 100)
                    """
                )

            conn.commit()
            print(f"{index_type.upper()} index created successfully")
        except Exception as e:
            print("Error creating index:", str(e))
        finally:
            cur.close()
            conn.close()



# This check ensures that the functions are only run when the script is executed directly, not when it's imported as a module.
if __name__ == "__main__":
    pg_crud = PgVectorCRUD(user="myuser", password="mypassword", host="localhost", port=5433, database="mydb")

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    sentences = [
        "A group of vibrant parrots chatter loudly, sharing stories of their tropical adventures.",
        "The mathematician found solace in numbers, deciphering the hidden patterns of the universe.",
        "The robot, with its intricate circuitry and precise movements, assembles the devices swiftly.",
        "The chef, with a sprinkle of spices and a dash of love, creates culinary masterpieces.",
        "The ancient tree, with its gnarled branches and deep roots, whispers secrets of the past.",
        "The detective, with keen observation and logical reasoning, unravels the intricate web of clues.",
        "The sunset paints the sky with shades of orange, pink, and purple, reflecting on the calm sea.",
        "In the dense forest, the howl of a lone wolf echoes, blending with the symphony of the night.",
        "The dancer, with graceful moves and expressive gestures, tells a story without uttering a word.",
        "In the quantum realm, particles flicker in and out of existence, dancing to the tunes of probability.",
    ]

    # Example of CRUD operations
    # pg_crud.create_items(model=model, sentences=sentences)
    pg_crud.create_index(index_type="hnsw", distance_op="l2")
    pg_crud.read_similar_items(model=model, query="Give me some content about the ocean", limit=5)
    # pg_crud.update_item(model=model, item_id=1, new_content="Updated content about tropical birds.")
    # pg_crud.update_item(item_id=2, new_content="Updated content about Mathematician.")
    # pg_crud.delete_item(item_id=3)
