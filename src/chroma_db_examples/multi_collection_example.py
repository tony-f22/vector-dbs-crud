from chromadb import HttpClient
from chromadb.api.models.Collection import Collection
from sentence_transformers import SentenceTransformer


class ChromaMultiCollection:
    """
    Strategy 2: Multiple Collections (Organized)
    
    Demonstrates using different collections for different data types:
    - product_descriptions (768 dims, all-mpnet-base-v2)
    - product_images (512 dims, simulated CLIP-like)
    - user_profiles (384 dims, all-MiniLM-L6-v2)
    
    Benefits:
    ✅ Different embedding models per collection
    ✅ Optimized for different data types
    ✅ Clear separation of concerns
    
    Trade-offs:
    ❌ Can't search across collections in a single query
    """

    def __init__(self, host: str = "localhost", port: int = 8000):
        self.client = HttpClient(host=host, port=port)

    def create_product_descriptions_collection(
        self, model: SentenceTransformer, products: list[dict]
    ) -> Collection:
        """
        Create collection for product descriptions with 768-dim embeddings.
        
        Args:
            model: SentenceTransformer model (768 dims)
            products: List of product dicts
            
        Returns:
            Collection object
        """
        collection_name = "product_descriptions"

        # Delete if exists
        try:
            self.client.delete_collection(name=collection_name)
        except Exception:
            pass

        collection = self.client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

        # Prepare data
        ids = [p["product_id"] for p in products]
        documents = [f"{p['title']}. {p['description']}" for p in products]
        metadatas = [
            {
                "title": p["title"],
                "category": p["category"],
                "price": p["price"]
            }
            for p in products
        ]

        # Generate embeddings manually for 768-dim model
        embeddings = [model.encode(doc).tolist() for doc in documents]

        # Add to collection
        collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings
        )

        print(f"Collection '{collection_name}' created with {len(products)} products (768 dims)")
        return collection

    def create_product_images_collection(
        self, model: SentenceTransformer, images: list[dict]
    ) -> Collection:
        """
        Create collection for product images with 512-dim embeddings.
        
        Note: In production, use CLIP or similar vision model.
        Here we simulate by embedding alt_text and padding to 512 dims.
        
        Args:
            model: SentenceTransformer model
            images: List of image dicts
            
        Returns:
            Collection object
        """
        collection_name = "product_images"

        # Delete if exists
        try:
            self.client.delete_collection(name=collection_name)
        except Exception:
            pass

        collection = self.client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

        # Prepare data
        ids = [f"img_{i}" for i in range(len(images))]
        documents = [img["alt_text"] for img in images]
        metadatas = [
            {
                "product_id": img["product_id"],
                "image_url": img["image_url"]
            }
            for img in images
        ]

        # Generate embeddings and pad/truncate to 512 dims
        embeddings = []
        for doc in documents:
            emb = model.encode(doc).tolist()
            if len(emb) < 512:
                emb.extend([0.0] * (512 - len(emb)))
            else:
                emb = emb[:512]
            embeddings.append(emb)

        # Add to collection
        collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings
        )

        print(f"Collection '{collection_name}' created with {len(images)} images (512 dims)")
        return collection

    def create_user_profiles_collection(
        self, model: SentenceTransformer, users: list[dict]
    ) -> Collection:
        """
        Create collection for user profiles with 384-dim embeddings.
        
        Args:
            model: SentenceTransformer model (384 dims)
            users: List of user profile dicts
            
        Returns:
            Collection object
        """
        collection_name = "user_profiles"

        # Delete if exists
        try:
            self.client.delete_collection(name=collection_name)
        except Exception:
            pass

        collection = self.client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

        # Prepare data
        ids = [u["user_id"] for u in users]
        documents = [f"{u['bio']}. Interests: {u['interests']}" for u in users]
        metadatas = [
            {
                "bio": u["bio"],
                "interests": u["interests"]
            }
            for u in users
        ]

        # Add to collection (ChromaDB will auto-generate embeddings)
        collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )

        print(f"Collection '{collection_name}' created with {len(users)} users (384 dims)")
        return collection

    def search_products(
        self, collection: Collection, model: SentenceTransformer, query: str, category: str = None, n_results: int = 5
    ) -> None:
        """Search product descriptions."""
        where_clause = {"category": category} if category else None

        # Generate query embedding using the same model (768 dims)
        query_embedding = model.encode(query)

        result = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where_clause,
            include=["documents", "metadatas", "distances"]
        )

        print(f"\nProduct Search: '{query}'")
        if category:
            print(f"Category Filter: {category}")
        print("Results:")

        for i, (prod_id, document, metadata, distance) in enumerate(
            zip(
                result["ids"][0],
                result["documents"][0],
                result["metadatas"][0],
                result["distances"][0],
                strict=True
            )
        ):
            similarity = 1 - distance
            print(f"\n  {i+1}. Product ID: {prod_id}")
            print(f"     Description: {document}")
            print(f"     Category: {metadata['category']}")
            print(f"     Price: ${metadata['price']}")
            print(f"     Similarity: {similarity:.4f}")

    def search_images(
        self, collection: Collection, model: SentenceTransformer, query: str, n_results: int = 5
    ) -> None:
        """Search product images by description."""

        # Generate and pad to 512 dims (simulating CLIP)
        emb = model.encode(query).tolist()
        if len(emb) < 512:
            emb.extend([0.0] * (512 - len(emb)))
        else:
            emb = emb[:512]
            
        result = collection.query(
            query_embeddings=[emb],
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )

        print(f"\nImage Search: '{query}'")
        print("Results:")

        for i, (img_id, document, metadata, distance) in enumerate(
            zip(
                result["ids"][0],
                result["documents"][0],
                result["metadatas"][0],
                result["distances"][0],
                strict=True
            )
        ):
            similarity = 1 - distance
            print(f"\n  {i+1}. Image ID: {img_id}")
            print(f"     Alt Text: {document}")
            print(f"     Product ID: {metadata['product_id']}")
            print(f"     URL: {metadata['image_url']}")
            print(f"     Similarity: {similarity:.4f}")

    def search_users(
        self, collection: Collection, query: str, n_results: int = 5
    ) -> None:
        """Search user profiles."""
        result = collection.query(
            query_texts=[query],
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )

        print(f"\nUser Profile Search: '{query}'")
        print("Results:")

        for i, (user_id, document, metadata, distance) in enumerate(
            zip(
                result["ids"][0],
                result["documents"][0],
                result["metadatas"][0],
                result["distances"][0],
                strict=True
            )
        ):
            similarity = 1 - distance
            print(f"\n  {i+1}. User ID: {user_id}")
            print(f"     Profile: {document}")
            print(f"     Similarity: {similarity:.4f}")

    def list_all_collections(self) -> None:
        """List all collections in ChromaDB."""
        collections = self.client.list_collections()
        print("\nAll Collections:")
        for coll in collections:
            count = coll.count()
            print(f"  - {coll.name}: {count} items")


if __name__ == "__main__":
    # Initialize ChromaDB client
    chroma_multi = ChromaMultiCollection(host="localhost", port=8000)

    # Load different models for different collections
    model_768 = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
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
        {
            "product_id": "MONITOR-001",
            "title": "4K Ultra HD Monitor",
            "description": "27-inch 4K monitor with HDR support, perfect for content creation",
            "category": "Electronics",
            "price": 499.99
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
        {
            "product_id": "MONITOR-001",
            "image_url": "https://example.com/monitor.jpg",
            "alt_text": "Ultra-wide 4K monitor with thin bezels"
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
        {
            "user_id": "USER-004",
            "bio": "Content creator specializing in tech reviews and photography",
            "interests": "photography, video editing, tech reviews, monitors"
        },
    ]

    # Create all collections
    print("="*80)
    print("Creating Multiple Collections")
    print("="*80)

    product_collection = chroma_multi.create_product_descriptions_collection(
        model=model_768,
        products=products
    )

    image_collection = chroma_multi.create_product_images_collection(
        model=model_384,
        images=images
    )

    user_collection = chroma_multi.create_user_profiles_collection(
        model=model_384,
        users=users
    )

    # List all collections
    print("\n" + "="*80)
    print("All Collections in Database")
    print("="*80)
    chroma_multi.list_all_collections()

    # Example 1: Search products
    print("\n" + "="*80)
    print("Example 1: Search All Products")
    print("="*80)
    chroma_multi.search_products(
        collection=product_collection,
        model=model_768,
        query="computer for AI development",
        n_results=3
    )

    # Example 2: Search products by category
    print("\n" + "="*80)
    print("Example 2: Search Electronics Only")
    print("="*80)
    chroma_multi.search_products(
        collection=product_collection,
        model=model_768,
        query="device with good camera",
        category="Electronics",
        n_results=3
    )

    # Example 3: Search images
    print("\n" + "="*80)
    print("Example 3: Search Product Images")
    print("="*80)
    chroma_multi.search_images(
        collection=image_collection,
        model=model_384,
        query="laptop with colorful keyboard",
        n_results=3
    )

    # Example 4: Find similar users
    print("\n" + "="*80)
    print("Example 4: Find Users Interested in AI")
    print("="*80)
    chroma_multi.search_users(
        collection=user_collection,
        query="interested in artificial intelligence and neural networks",
        n_results=3
    )

    # Example 5: Find users interested in productivity
    print("\n" + "="*80)
    print("Example 5: Find Users Interested in Content Creation")
    print("="*80)
    chroma_multi.search_users(
        collection=user_collection,
        query="video editing and photography",
        n_results=3
    )

    # Example 6: Cross-collection workflow
    print("\n" + "="*80)
    print("Example 6: Cross-Collection Workflow")
    print("="*80)
    print("\nStep 1: Find users interested in gaming")
    chroma_multi.search_users(
        collection=user_collection,
        query="gaming and technology",
        n_results=2
    )

    print("\nStep 2: Find products for those users")
    chroma_multi.search_products(
        collection=product_collection,
        model=model_768,
        query="gaming laptop with powerful GPU",
        n_results=2
    )
