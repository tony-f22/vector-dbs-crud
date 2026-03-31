import psycopg2
from pg_embedding_util import generate_embeddings


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
        cur = conn.cursor()
        return conn, cur

    # ================================
    # CREATE Operation
    # ================================
    def create_items(self, sentences: list[str]) -> None:
        """
        Insert new sentences with their embeddings into the items table.

        Args:
            sentences (List[str]): A list of sentences to be inserted into the database.
        """
        conn, cur = self.connect_db()
        try:
            for sentence in sentences:
                embedding = generate_embeddings(sentence)
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
    def read_similar_items(self, query: str, limit: int) -> None:
        """
        Perform a cosine similarity search for the query and return similar items.

        Args:
            query (str): The query text to search for similar items.
            limit (int): The number of top results to return.
        """
        conn, cur = self.connect_db()
        try:
            query_embedding = generate_embeddings(query)

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
    def update_item(self, item_id: int, new_content: str) -> None:
        """
        Update the content and embedding of an item in the database by its ID.

        Args:
            item_id (int): The ID of the item to be updated.
            new_content (str): The new content to update.
        """
        conn, cur = self.connect_db()
        try:
            new_embedding = generate_embeddings(new_content)

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


# This check ensures that the functions are only run when the script is executed directly, not when it's imported as a module.
if __name__ == "__main__":
    pg_crud = PgVectorCRUD(user="myuser", password="mypassword", host="localhost", port=5433, database="mydb")

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
    # pg_crud.create_items(sentences)
    pg_crud.read_similar_items(query="Give me some content about the ocean", limit=5)
    # pg_crud.update_item(item_id=1, new_content="Updated content about tropical birds.")
    # pg_crud.update_item(item_id=2, new_content="Updated content about Mathematician.")
    # pg_crud.delete_item(item_id=3)
