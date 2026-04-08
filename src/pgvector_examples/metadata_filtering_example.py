import psycopg2
import psycopg2.extras
from pgvector.psycopg2 import register_vector
from sentence_transformers import SentenceTransformer


class PgVectorMetadataFiltering:
    """
    Strategy 1: Single Collection with Metadata Filtering
    
    Demonstrates the fundamental pattern: Vector + Metadata
    - Single table with consistent dimensionality
    - Rich metadata for filtering
    - Combines vector similarity with structured queries
    """

    def __init__(self, user: str, password: str, host: str, port: int, database: str):
        self.user = user
        self.password = password
        self.host = host
        self.port = port
        self.database = database

    def connect_db(self):
        """Establish a connection to the PostgreSQL database."""
        conn = psycopg2.connect(
            user=self.user,
            password=self.password,
            host=self.host,
            port=self.port,
            database=self.database,
        )
        register_vector(conn)
        cur = conn.cursor()
        return conn, cur

    def create_table_with_metadata(self) -> None:
        """Create a table with vector embeddings and JSONB metadata."""
        conn, cur = self.connect_db()
        try:
            cur.execute("""
                DROP TABLE IF EXISTS documents;
                
                CREATE TABLE documents (
                    id bigserial PRIMARY KEY,
                    content TEXT NOT NULL,
                    embedding vector(384),
                    metadata JSONB
                );
                
                -- Create GIN index for efficient metadata filtering
                CREATE INDEX documents_metadata_idx ON documents USING GIN (metadata);
                
                -- Create HNSW index for vector similarity search
                CREATE INDEX documents_embedding_idx ON documents 
                USING hnsw (embedding vector_cosine_ops);
            """)
            conn.commit()
            print("Table 'documents' created with metadata support")
        except Exception as e:
            print("Error creating table:", str(e))
        finally:
            cur.close()
            conn.close()

    def insert_documents_with_metadata(
        self, model: SentenceTransformer, documents: list[dict]
    ) -> None:
        """
        Insert documents with embeddings and metadata.
        
        Args:
            model: SentenceTransformer model for generating embeddings
            documents: List of dicts with 'content' and 'metadata' keys
        """
        conn, cur = self.connect_db()
        try:
            for doc in documents:
                content = doc["content"]
                metadata = doc["metadata"]
                embedding = model.encode(content)

                cur.execute(
                    """INSERT INTO documents (content, embedding, metadata) 
                       VALUES (%s, %s, %s)""",
                    (content, embedding, psycopg2.extras.Json(metadata)),
                )

            conn.commit()
            print(f"{len(documents)} documents inserted with metadata")
        except Exception as e:
            print("Error during insertion:", str(e))
        finally:
            cur.close()
            conn.close()

    def search_with_metadata_filter(
        self,
        model: SentenceTransformer,
        query: str,
        metadata_filter: dict = None,
        limit: int = 5
    ) -> None:
        """
        Search for similar documents with optional metadata filtering.
        
        Args:
            model: SentenceTransformer model
            query: Search query text
            metadata_filter: Dict of metadata conditions (e.g., {"category": "Technology"})
            limit: Number of results to return
        """
        conn, cur = self.connect_db()
        try:
            query_embedding = model.encode(query)

            if metadata_filter:
                # Build WHERE clause for metadata filtering
                conditions = []
                for key, value in metadata_filter.items():
                    conditions.append(f"metadata->'{key}' = %s")

                where_clause = " AND ".join(conditions)
                values = [psycopg2.extras.Json(v) for v in metadata_filter.values()]

                cur.execute(
                    f"""SELECT id, content, metadata, 
                              1 - (embedding <=> %s) AS cosine_similarity
                       FROM documents
                       WHERE {where_clause}
                       ORDER BY cosine_similarity DESC 
                       LIMIT %s""",
                    [query_embedding] + values + [limit],
                )
            else:
                # No metadata filter - pure vector search
                cur.execute(
                    """SELECT id, content, metadata, 
                              1 - (embedding <=> %s) AS cosine_similarity
                       FROM documents
                       ORDER BY cosine_similarity DESC 
                       LIMIT %s""",
                    (query_embedding, limit),
                )

            print(f"\nQuery: '{query}'")
            if metadata_filter:
                print(f"Metadata Filter: {metadata_filter}")
            print("Results:")

            for row in cur.fetchall():
                print(f"\nID: {row[0]}")
                print(f"Content: {row[1]}")
                print(f"Metadata: {row[2]}")
                print(f"Similarity: {row[3]:.4f}")

        except Exception as e:
            print("Error during search:", str(e))
        finally:
            cur.close()
            conn.close()

    def search_by_metadata_array(
        self,
        model: SentenceTransformer,
        query: str,
        tag: str,
        limit: int = 5
    ) -> None:
        """
        Search documents that contain a specific tag in metadata array.
        
        Args:
            model: SentenceTransformer model
            query: Search query text
            tag: Tag to filter by (e.g., "AI")
            limit: Number of results to return
        """
        conn, cur = self.connect_db()
        try:
            query_embedding = model.encode(query)

            cur.execute(
                """SELECT id, content, metadata, 
                          1 - (embedding <=> %s) AS cosine_similarity
                   FROM documents
                   WHERE metadata->'tags' ? %s
                   ORDER BY cosine_similarity DESC 
                   LIMIT %s""",
                (query_embedding, tag, limit),
            )

            print(f"\nQuery: '{query}'")
            print(f"Tag Filter: '{tag}'")
            print("Results:")

            for row in cur.fetchall():
                print(f"\nID: {row[0]}")
                print(f"Content: {row[1]}")
                print(f"Metadata: {row[2]}")
                print(f"Similarity: {row[3]:.4f}")

        except Exception as e:
            print("Error during search:", str(e))
        finally:
            cur.close()
            conn.close()


if __name__ == "__main__":
    # Initialize the class
    pg_metadata = PgVectorMetadataFiltering(
        user="myuser",
        password="mypassword",
        host="localhost",
        port=5433,
        database="mydb"
    )

    # Load the embedding model
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    # Create table with metadata support
    pg_metadata.create_table_with_metadata()

    # Sample documents with metadata
    documents = [
        {
            "content": "Vector databases are specialized systems for storing and querying high-dimensional vectors efficiently.",
            "metadata": {
                "title": "Vector Databases Guide",
                "category": "Technology",
                "tags": ["AI", "databases", "vectors"],
                "author": "Tech Team",
                "date": "2024-01-15"
            }
        },
        {
            "content": "Machine learning models convert text into numerical representations called embeddings.",
            "metadata": {
                "title": "Understanding Embeddings",
                "category": "Technology",
                "tags": ["AI", "machine-learning", "embeddings"],
                "author": "Data Science Team",
                "date": "2024-01-20"
            }
        },
        {
            "content": "The sunset paints the sky with shades of orange, pink, and purple, reflecting on the calm sea.",
            "metadata": {
                "title": "Beautiful Sunset",
                "category": "Nature",
                "tags": ["sunset", "ocean", "scenery"],
                "author": "Nature Writer",
                "date": "2024-02-01"
            }
        },
        {
            "content": "HNSW and IVFFlat are popular indexing algorithms for approximate nearest neighbor search.",
            "metadata": {
                "title": "Vector Indexing Algorithms",
                "category": "Technology",
                "tags": ["AI", "algorithms", "indexing"],
                "author": "Tech Team",
                "date": "2024-02-10"
            }
        },
        {
            "content": "The ancient tree, with its gnarled branches and deep roots, whispers secrets of the past.",
            "metadata": {
                "title": "Ancient Wisdom",
                "category": "Nature",
                "tags": ["trees", "nature", "history"],
                "author": "Nature Writer",
                "date": "2024-02-15"
            }
        },
    ]

    # Insert documents with metadata
    pg_metadata.insert_documents_with_metadata(model, documents)

    # Example 1: Pure vector search (no metadata filter)
    print("\n" + "="*80)
    print("Example 1: Pure Vector Search")
    print("="*80)
    pg_metadata.search_with_metadata_filter(
        model=model,
        query="Tell me about AI and databases",
        limit=3
    )

    # Example 2: Vector search with category filter
    print("\n" + "="*80)
    print("Example 2: Vector Search + Category Filter")
    print("="*80)
    pg_metadata.search_with_metadata_filter(
        model=model,
        query="Tell me about AI and databases",
        metadata_filter={"category": "Technology"},
        limit=3
    )

    # Example 3: Vector search with tag filter
    print("\n" + "="*80)
    print("Example 3: Vector Search + Tag Filter")
    print("="*80)
    pg_metadata.search_by_metadata_array(
        model=model,
        query="machine learning concepts",
        tag="AI",
        limit=3
    )

    # Example 4: Search nature content
    print("\n" + "="*80)
    print("Example 4: Nature Category Search")
    print("="*80)
    pg_metadata.search_with_metadata_filter(
        model=model,
        query="beautiful scenery",
        metadata_filter={"category": "Nature"},
        limit=3
    )

