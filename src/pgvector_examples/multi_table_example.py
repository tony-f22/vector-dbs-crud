import psycopg2
from pgvector.psycopg2 import register_vector
from sentence_transformers import SentenceTransformer


class PgVectorMultiTable:
    """
    Strategy 2: Multiple Tables (Organized)
    
    Demonstrates using different tables for different data types:
    - product_descriptions (768 dims, all-mpnet-base-v2)
    - product_images (512 dims, simulated CLIP-like embeddings)
    - user_profiles (384 dims, all-MiniLM-L6-v2)
    
    Benefits:
    ✅ Different embedding models per table
    ✅ Optimized indexes per data type
    ✅ Clear separation of concerns
    
    Trade-offs:
    ❌ Can't search across tables in a single query
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

    def create_all_tables(self) -> None:
        """Create all specialized tables with different vector dimensions."""
        conn, cur = self.connect_db()
        try:
            # Table 1: Product Descriptions (768 dimensions)
            cur.execute("""
                DROP TABLE IF EXISTS product_descriptions CASCADE;
                
                CREATE TABLE product_descriptions (
                    id bigserial PRIMARY KEY,
                    product_id VARCHAR(50) UNIQUE NOT NULL,
                    title TEXT NOT NULL,
                    description TEXT NOT NULL,
                    embedding vector(768),
                    category VARCHAR(100),
                    price DECIMAL(10, 2),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                -- HNSW index for cosine similarity
                CREATE INDEX product_desc_embedding_idx ON product_descriptions 
                USING hnsw (embedding vector_cosine_ops);
                
                -- Regular indexes for filtering
                CREATE INDEX product_desc_category_idx ON product_descriptions(category);
            """)

            # Table 2: Product Images
            cur.execute("""
                DROP TABLE IF EXISTS product_images CASCADE;
                
                CREATE TABLE product_images (
                    id bigserial PRIMARY KEY,
                    product_id VARCHAR(50) NOT NULL,
                    image_url TEXT NOT NULL,
                    alt_text TEXT,
                    embedding vector(512),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                -- HNSW index for image similarity
                CREATE INDEX product_img_embedding_idx ON product_images 
                USING hnsw (embedding vector_cosine_ops);
            """)

            # Table 3: User Profiles (384 dimensions)
            cur.execute("""
                DROP TABLE IF EXISTS user_profiles CASCADE;
                
                CREATE TABLE user_profiles (
                    id bigserial PRIMARY KEY,
                    user_id VARCHAR(50) UNIQUE NOT NULL,
                    bio TEXT,
                    interests TEXT,
                    embedding vector(384),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                -- HNSW index for user similarity
                CREATE INDEX user_profile_embedding_idx ON user_profiles 
                USING hnsw (embedding vector_cosine_ops);
            """)

            conn.commit()
            print("All tables created successfully:")
            print("  - product_descriptions (768 dims)")
            print("  - product_images (512 dims)")
            print("  - user_profiles (384 dims)")
        except Exception as e:
            print("Error creating tables:", str(e))
        finally:
            cur.close()
            conn.close()

    def insert_product_descriptions(
        self, model: SentenceTransformer, products: list[dict]
    ) -> None:
        """
        Insert product descriptions with 768-dimensional embeddings.
        
        Args:
            model: SentenceTransformer model (should output 768 dims)
            products: List of product dicts
        """
        conn, cur = self.connect_db()
        try:
            for product in products:
                # Combine title and description for embedding
                text = f"{product['title']}. {product['description']}"
                embedding = model.encode(text)

                cur.execute(
                    """INSERT INTO product_descriptions 
                       (product_id, title, description, embedding, category, price) 
                       VALUES (%s, %s, %s, %s, %s, %s)""",
                    (
                        product["product_id"],
                        product["title"],
                        product["description"],
                        embedding.tolist(),
                        product["category"],
                        product["price"],
                    ),
                )

            conn.commit()
            print(f"{len(products)} product descriptions inserted")
        except Exception as e:
            print("Error inserting product descriptions:", str(e))
        finally:
            cur.close()
            conn.close()

    def insert_product_images(
        self, model: SentenceTransformer, images: list[dict]
    ) -> None:
        """
        Insert product images with embeddings.
        
        Note: In production, you'd use a vision model like CLIP.
        Here we simulate by embedding the alt_text with a smaller model.
        
        Args:
            model: SentenceTransformer model
            images: List of image dicts
        """
        conn, cur = self.connect_db()
        try:
            for image in images:
                # Simulate image embedding using alt_text
                # In production: use CLIP or similar vision model
                embedding = model.encode(image["alt_text"])

                # Pad or truncate to 512 dimensions (simulation)
                embedding_list = embedding.tolist()
                if len(embedding_list) < 512:
                    embedding_list.extend([0.0] * (512 - len(embedding_list)))
                else:
                    embedding_list = embedding_list[:512]

                cur.execute(
                    """INSERT INTO product_images 
                       (product_id, image_url, alt_text, embedding) 
                       VALUES (%s, %s, %s, %s)""",
                    (
                        image["product_id"],
                        image["image_url"],
                        image["alt_text"],
                        embedding_list,
                    ),
                )

            conn.commit()
            print(f"{len(images)} product images inserted")
        except Exception as e:
            print("Error inserting product images:", str(e))
        finally:
            cur.close()
            conn.close()

    def insert_user_profiles(
        self, model: SentenceTransformer, users: list[dict]
    ) -> None:
        """
        Insert user profiles with 384-dimensional embeddings.
        
        Args:
            model: SentenceTransformer model (should output 384 dims)
            users: List of user profile dicts
        """
        conn, cur = self.connect_db()
        try:
            for user in users:
                # Combine bio and interests for embedding
                text = f"{user['bio']}. Interests: {user['interests']}"
                embedding = model.encode(text)

                cur.execute(
                    """INSERT INTO user_profiles 
                       (user_id, bio, interests, embedding) 
                       VALUES (%s, %s, %s, %s)""",
                    (
                        user["user_id"],
                        user["bio"],
                        user["interests"],
                        embedding.tolist(),
                    ),
                )

            conn.commit()
            print(f"{len(users)} user profiles inserted")
        except Exception as e:
            print("Error inserting user profiles:", str(e))
        finally:
            cur.close()
            conn.close()

    def search_product_descriptions(
        self, model: SentenceTransformer, query: str, category: str = None, limit: int = 5
    ) -> None:
        """Search product descriptions by semantic similarity."""
        conn, cur = self.connect_db()
        try:
            query_embedding = model.encode(query)

            if category:
                cur.execute(
                    """SELECT product_id, title, description, category, price,
                              1 - (embedding <=> %s) AS similarity
                       FROM product_descriptions
                       WHERE category = %s
                       ORDER BY similarity DESC 
                       LIMIT %s""",
                    (query_embedding, category, limit),
                )
            else:
                cur.execute(
                    """SELECT product_id, title, description, category, price,
                              1 - (embedding <=> %s) AS similarity
                       FROM product_descriptions
                       ORDER BY similarity DESC 
                       LIMIT %s""",
                    (query_embedding, limit),
                )

            print(f"\nProduct Search: '{query}'")
            if category:
                print(f"Category Filter: {category}")
            print("Results:")

            for row in cur.fetchall():
                print(f"\n  Product ID: {row[0]}")
                print(f"  Title: {row[1]}")
                print(f"  Description: {row[2]}")
                print(f"  Category: {row[3]}")
                print(f"  Price: ${row[4]}")
                print(f"  Similarity: {row[5]:.4f}")

        except Exception as e:
            print("Error during search:", str(e))
        finally:
            cur.close()
            conn.close()

    def search_similar_users(
        self, model: SentenceTransformer, query: str, limit: int = 5
    ) -> None:
        """Find users with similar interests/profiles."""
        conn, cur = self.connect_db()
        try:
            query_embedding = model.encode(query)

            cur.execute(
                """SELECT user_id, bio, interests,
                          1 - (embedding <=> %s) AS similarity
                   FROM user_profiles
                   ORDER BY similarity DESC 
                   LIMIT %s""",
                (query_embedding, limit),
            )

            print(f"\nUser Profile Search: '{query}'")
            print("Results:")

            for row in cur.fetchall():
                print(f"\n  User ID: {row[0]}")
                print(f"  Bio: {row[1]}")
                print(f"  Interests: {row[2]}")
                print(f"  Similarity: {row[3]:.4f}")

        except Exception as e:
            print("Error during search:", str(e))
        finally:
            cur.close()
            conn.close()


if __name__ == "__main__":
    # Initialize the class
    pg_multi = PgVectorMultiTable(
        user="myuser",
        password="mypassword",
        host="localhost",
        port=5433,
        database="mydb"
    )

    # Create all tables
    pg_multi.create_all_tables()

    # Load different models for different tables
    # Model 1: 768 dimensions for product descriptions
    model_768 = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

    # Model 2: 384 dimensions for user profiles (and simulated images)
    model_384 = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    # Sample product descriptions
    products = [
        {
            "product_id": "LAPTOP-001",
            "title": "High-Performance Gaming Laptop",
            "description": "Powerful laptop with RTX 4080 GPU, 32GB RAM, perfect for gaming and AI development",
            "category": "Electronics",
            "price": 2499.99
        },
        {
            "product_id": "PHONE-001",
            "title": "Smartphone Pro Max",
            "description": "Latest flagship smartphone with advanced AI camera and 5G connectivity",
            "category": "Electronics",
            "price": 1199.99
        },
        {
            "product_id": "BOOK-001",
            "title": "Machine Learning Fundamentals",
            "description": "Comprehensive guide to machine learning algorithms and neural networks",
            "category": "Books",
            "price": 49.99
        },
        {
            "product_id": "DESK-001",
            "title": "Ergonomic Standing Desk",
            "description": "Adjustable height desk for comfortable work from home setup",
            "category": "Furniture",
            "price": 599.99
        },
    ]

    # Sample product images
    images = [
        {
            "product_id": "LAPTOP-001",
            "image_url": "https://example.com/laptop.jpg",
            "alt_text": "Gaming laptop with RGB keyboard and large display"
        },
        {
            "product_id": "PHONE-001",
            "image_url": "https://example.com/phone.jpg",
            "alt_text": "Sleek smartphone with triple camera system"
        },
    ]

    # Sample user profiles
    users = [
        {
            "user_id": "USER-001",
            "bio": "Software engineer passionate about AI and machine learning",
            "interests": "programming, deep learning, gaming, technology"
        },
        {
            "user_id": "USER-002",
            "bio": "Data scientist working on NLP and computer vision projects",
            "interests": "natural language processing, computer vision, research"
        },
        {
            "user_id": "USER-003",
            "bio": "Home office enthusiast and productivity blogger",
            "interests": "productivity, ergonomics, remote work, furniture"
        },
    ]

    # Insert data into different tables
    print("\n" + "="*80)
    print("Inserting Data into Multiple Tables")
    print("="*80)
    pg_multi.insert_product_descriptions(model_768, products)
    pg_multi.insert_product_images(model_384, images)
    pg_multi.insert_user_profiles(model_384, users)

    # Example 1: Search products
    print("\n" + "="*80)
    print("Example 1: Search All Products")
    print("="*80)
    pg_multi.search_product_descriptions(
        model=model_768,
        query="computer for AI development",
        limit=3
    )

    # Example 2: Search products by category
    print("\n" + "="*80)
    print("Example 2: Search Electronics Only")
    print("="*80)
    pg_multi.search_product_descriptions(
        model=model_768,
        query="device with good camera",
        category="Electronics",
        limit=3
    )

    # Example 3: Find similar users
    print("\n" + "="*80)
    print("Example 3: Find Users Interested in AI")
    print("="*80)
    pg_multi.search_similar_users(
        model=model_384,
        query="interested in artificial intelligence and neural networks",
        limit=3
    )

    # Example 4: Find users interested in productivity
    print("\n" + "="*80)
    print("Example 4: Find Users Interested in Productivity")
    print("="*80)
    pg_multi.search_similar_users(
        model=model_384,
        query="work from home and office setup",
        limit=3
    )
