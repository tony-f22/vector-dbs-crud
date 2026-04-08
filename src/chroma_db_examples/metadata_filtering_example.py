
from chromadb import HttpClient
from chromadb.api.models.Collection import Collection
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction


class ChromaMetadataFiltering:
    """
    Strategy 1: Single Collection with Metadata Filtering
    
    Demonstrates the fundamental pattern: Vector + Metadata
    - Single collection with consistent dimensionality
    - Rich metadata for filtering
    - Combines vector similarity with structured queries
    """

    def __init__(self, host: str = "localhost", port: int = 8000):
        self.client = HttpClient(host=host, port=port)

    def create_collection_with_metadata(
        self, collection_name: str, documents: list[dict], ef: SentenceTransformerEmbeddingFunction
    ) -> Collection:
        """
        Create a collection and add documents with metadata.
        
        Args:
            collection_name: Name of the collection
            documents: List of dicts with 'content' and 'metadata' keys
            
        Returns:
            Collection object
        """
        # Delete if exists (for clean demo)
        try:
            self.client.delete_collection(name=collection_name)
        except Exception:
            pass

        collection = self.client.create_collection(name=collection_name, embedding_function=ef)

        # Prepare data for insertion
        ids = [f"doc_{i}" for i in range(len(documents))]
        contents = [doc["content"] for doc in documents]
        metadatas = [doc["metadata"] for doc in documents]

        # Add documents with metadata
        collection.add(
            documents=contents,
            metadatas=metadatas,
            ids=ids
        )

        print(f"Collection '{collection_name}' created with {len(documents)} documents")
        return collection

    def search_with_metadata_filter(
        self,
        collection: Collection,
        query: str,
        metadata_filter: dict = None,
        n_results: int = 5
    ) -> None:
        """
        Search with optional metadata filtering.
        
        Args:
            collection: ChromaDB collection
            query: Search query text
            metadata_filter: Dict of metadata conditions
            n_results: Number of results to return
        """
        # Build where clause for metadata filtering
        where_clause = metadata_filter if metadata_filter else None

        result = collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where_clause,
            include=["documents", "metadatas", "distances"]
        )

        print(f"\nQuery: '{query}'")
        if metadata_filter:
            print(f"Metadata Filter: {metadata_filter}")
        print("Results:")

        for i, (doc_id, document, metadata, distance) in enumerate(
            zip(
                result["ids"][0],
                result["documents"][0],
                result["metadatas"][0],
                result["distances"][0],
                strict=True
            )
        ):
            similarity = 1 - distance  # Convert distance to similarity
            print(f"\n  {i+1}. ID: {doc_id}")
            print(f"     Content: {document}")
            print(f"     Metadata: {metadata}")
            print(f"     Similarity: {similarity:.4f}")

    def search_with_complex_filter(
        self,
        collection: Collection,
        query: str,
        where_filter: dict,
        n_results: int = 5
    ) -> None:
        """
        Search with complex metadata filtering using operators.
        
        ChromaDB supports operators like:
        - $eq, $ne (equals, not equals)
        - $gt, $gte, $lt, $lte (greater than, less than)
        - $in, $nin (in array, not in array)
        - $and, $or (logical operators)
        
        Args:
            collection: ChromaDB collection
            query: Search query text
            where_filter: Complex filter dict with operators
            n_results: Number of results
        """
        result = collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where_filter,
            include=["documents", "metadatas", "distances"]
        )

        print(f"\nQuery: '{query}'")
        print(f"Complex Filter: {where_filter}")
        print("Results:")

        for i, (doc_id, document, metadata, distance) in enumerate(
            zip(
                result["ids"][0],
                result["documents"][0],
                result["metadatas"][0],
                result["distances"][0],
                strict=True
            )
        ):
            similarity = 1 - distance
            print(f"\n  {i+1}. ID: {doc_id}")
            print(f"     Content: {document}")
            print(f"     Metadata: {metadata}")
            print(f"     Similarity: {similarity:.4f}")

    def search_with_document_filter(
        self,
        collection: Collection,
        query: str,
        where_document: dict,
        n_results: int = 5
    ) -> None:
        """
        Search with document content filtering.
        
        Args:
            collection: ChromaDB collection
            query: Search query text
            where_document: Filter on document content (e.g., {"$contains": "AI"})
            n_results: Number of results
        """
        result = collection.query(
            query_texts=[query],
            n_results=n_results,
            where_document=where_document,
            include=["documents", "metadatas", "distances"]
        )

        print(f"\nQuery: '{query}'")
        print(f"Document Filter: {where_document}")
        print("Results:")

        for i, (doc_id, document, metadata, distance) in enumerate(
            zip(
                result["ids"][0],
                result["documents"][0],
                result["metadatas"][0],
                result["distances"][0],
                strict=True
            )
        ):
            similarity = 1 - distance
            print(f"\n  {i+1}. ID: {doc_id}")
            print(f"     Content: {document}")
            print(f"     Metadata: {metadata}")
            print(f"     Similarity: {similarity:.4f}")


if __name__ == "__main__":
    # Initialize ChromaDB client
    chroma_metadata = ChromaMetadataFiltering(host="localhost", port=8000)

    # Sample documents with rich metadata (matching presentation slide 212)
    documents = [
        {
            "content": "Vector databases are specialized systems for storing and querying high-dimensional vectors efficiently.",
            "metadata": {
                "title": "Vector Databases Guide",
                "category": "Technology",
                "tags": "AI,databases,vectors",
                "author": "Tech Team",
                "year": 2024,
                "views": 1500
            }
        },
        {
            "content": "Machine learning models convert text into numerical representations called embeddings.",
            "metadata": {
                "title": "Understanding Embeddings",
                "category": "Technology",
                "tags": "AI,machine-learning,embeddings",
                "author": "Data Science Team",
                "year": 2024,
                "views": 2300
            }
        },
        {
            "content": "The sunset paints the sky with shades of orange, pink, and purple, reflecting on the calm sea.",
            "metadata": {
                "title": "Beautiful Sunset",
                "category": "Nature",
                "tags": "sunset,ocean,scenery",
                "author": "Nature Writer",
                "year": 2024,
                "views": 890
            }
        },
        {
            "content": "HNSW and IVFFlat are popular indexing algorithms for approximate nearest neighbor search.",
            "metadata": {
                "title": "Vector Indexing Algorithms",
                "category": "Technology",
                "tags": "AI,algorithms,indexing",
                "author": "Tech Team",
                "year": 2024,
                "views": 3200
            }
        },
        {
            "content": "The ancient tree, with its gnarled branches and deep roots, whispers secrets of the past.",
            "metadata": {
                "title": "Ancient Wisdom",
                "category": "Nature",
                "tags": "trees,nature,history",
                "author": "Nature Writer",
                "year": 2023,
                "views": 650
            }
        },
        {
            "content": "Deep learning neural networks have revolutionized computer vision and natural language processing.",
            "metadata": {
                "title": "Deep Learning Revolution",
                "category": "Technology",
                "tags": "AI,deep-learning,neural-networks",
                "author": "AI Researcher",
                "year": 2023,
                "views": 4100
            }
        },
    ]

    sentence_transformer_ef = SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2",
    )


    # Create collection with metadata
    collection = chroma_metadata.create_collection_with_metadata(
        collection_name="documents_with_metadata",
        documents=documents,
        ef=sentence_transformer_ef
    )

    # Example 1: Pure vector search (no filter)
    print("\n" + "="*80)
    print("Example 1: Pure Vector Search")
    print("="*80)
    chroma_metadata.search_with_metadata_filter(
        collection=collection,
        query="Tell me about AI and databases",
        n_results=3
    )

    # Example 2: Vector search with simple category filter
    print("\n" + "="*80)
    print("Example 2: Vector Search + Category Filter")
    print("="*80)
    chroma_metadata.search_with_metadata_filter(
        collection=collection,
        query="Tell me about AI and databases",
        metadata_filter={"category": "Technology"},
        n_results=3
    )

    # Example 3: Vector search with author filter
    print("\n" + "="*80)
    print("Example 3: Vector Search + Author Filter")
    print("="*80)
    chroma_metadata.search_with_metadata_filter(
        collection=collection,
        query="interesting content",
        metadata_filter={"author": "Tech Team"},
        n_results=3
    )

    # Example 4: Complex filter - Technology category AND high views
    print("\n" + "="*80)
    print("Example 4: Complex Filter - Technology with High Views")
    print("="*80)
    chroma_metadata.search_with_complex_filter(
        collection=collection,
        query="artificial intelligence",
        where_filter={
            "$and": [
                {"category": {"$eq": "Technology"}},
                {"views": {"$gt": 2000}}
            ]
        },
        n_results=3
    )

    # Example 5: Filter by year
    print("\n" + "="*80)
    print("Example 5: Filter by Year (2024 only)")
    print("="*80)
    chroma_metadata.search_with_complex_filter(
        collection=collection,
        query="latest technology trends",
        where_filter={"year": {"$eq": 2024}},
        n_results=3
    )

    # Example 6: Document content filter
    print("\n" + "="*80)
    print("Example 6: Document Content Filter (contains 'neural')")
    print("="*80)
    chroma_metadata.search_with_document_filter(
        collection=collection,
        query="machine learning",
        where_document={"$contains": "neural"},
        n_results=3
    )

    # Example 7: OR filter - Either Nature OR high views
    print("\n" + "="*80)
    print("Example 7: OR Filter - Nature OR High Views")
    print("="*80)
    chroma_metadata.search_with_complex_filter(
        collection=collection,
        query="interesting content",
        where_filter={
            "$or": [
                {"category": {"$eq": "Nature"}},
                {"views": {"$gt": 3000}}
            ]
        },
        n_results=4
    )
