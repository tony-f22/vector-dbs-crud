from chromadb import HttpClient
from chromadb.api.models.Collection import Collection


class ChromaMultiTenant:
    """
    Strategy 3: Namespaces/Partitions (Multi-tenant)
    
    Demonstrates tenant isolation using collection naming conventions:
    - Each tenant gets their own prefixed collections
    - Same structure across all tenants
    - Data isolation between tenants
    - Easy tenant management
    
    Benefits:
    ✅ Multi-tenancy with data isolation
    ✅ Per-tenant management
    ✅ Easy tenant onboarding/offboarding
    ✅ Consistent structure across tenants
    
    Trade-offs:
    ❌ Collection naming convention must be enforced
    ❌ No native namespace support in ChromaDB
    
    Note: ChromaDB doesn't have native namespace support like PostgreSQL schemas,
    so we use a naming convention: {tenant_id}_{collection_type}
    """

    def __init__(self, host: str = "localhost", port: int = 8000):
        self.client = HttpClient(host=host, port=port)

    def get_tenant_collection_name(self, tenant_id: str, collection_type: str) -> str:
        """
        Generate a tenant-specific collection name.
        
        Args:
            tenant_id: Unique tenant identifier
            collection_type: Type of collection (e.g., "documents", "images")
            
        Returns:
            Namespaced collection name
        """
        return f"tenant_{tenant_id}_{collection_type}"

    def create_tenant_collection(
        self, tenant_id: str, collection_type: str = "documents"
    ) -> Collection:
        """
        Create a collection for a specific tenant.
        
        Args:
            tenant_id: Tenant identifier
            collection_type: Type of collection
            
        Returns:
            Collection object
        """
        collection_name = self.get_tenant_collection_name(tenant_id, collection_type)

        # Delete if exists (for clean demo)
        try:
            self.client.delete_collection(name=collection_name)
        except Exception:
            pass

        collection = self.client.create_collection(name=collection_name)
        print(f"Collection created for tenant '{tenant_id}': {collection_name}")
        return collection

    def add_tenant_documents(
        self,
        tenant_id: str,
        documents: list[dict],
        collection_type: str = "documents"
    ) -> None:
        """
        Add documents to a tenant's collection.
        
        Args:
            tenant_id: Tenant identifier
            documents: List of document dicts with 'content' and 'metadata'
            collection_type: Type of collection
        """
        collection_name = self.get_tenant_collection_name(tenant_id, collection_type)
        collection = self.client.get_collection(name=collection_name)

        # Prepare data
        ids = [f"{tenant_id}_doc_{i}" for i in range(len(documents))]
        contents = [doc["content"] for doc in documents]
        metadatas = [doc["metadata"] for doc in documents]

        # Add documents
        collection.add(
            ids=ids,
            documents=contents,
            metadatas=metadatas
        )

        print(f"Added {len(documents)} documents to tenant '{tenant_id}'")

    def search_tenant_documents(
        self,
        tenant_id: str,
        query: str,
        collection_type: str = "documents",
        metadata_filter: dict = None,
        n_results: int = 5
    ) -> None:
        """
        Search documents within a specific tenant's collection.
        
        Args:
            tenant_id: Tenant identifier
            query: Search query
            collection_type: Type of collection
            metadata_filter: Optional metadata filter
            n_results: Number of results
        """
        collection_name = self.get_tenant_collection_name(tenant_id, collection_type)

        try:
            collection = self.client.get_collection(name=collection_name)

            result = collection.query(
                query_texts=[query],
                n_results=n_results,
                where=metadata_filter,
                include=["documents", "metadatas", "distances"]
            )

            print(f"\nTenant: {tenant_id}")
            print(f"Query: '{query}'")
            if metadata_filter:
                print(f"Filter: {metadata_filter}")
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

        except Exception as e:
            print(f"Error searching tenant '{tenant_id}': {str(e)}")

    def get_tenant_stats(self, tenant_id: str, collection_type: str = "documents") -> None:
        """
        Get statistics for a tenant's collection.
        
        Args:
            tenant_id: Tenant identifier
            collection_type: Type of collection
        """
        collection_name = self.get_tenant_collection_name(tenant_id, collection_type)

        try:
            collection = self.client.get_collection(name=collection_name)
            count = collection.count()

            print(f"\nTenant: {tenant_id}")
            print(f"  Collection: {collection_name}")
            print(f"  Documents: {count}")

        except Exception as e:
            print(f"Error getting stats for tenant '{tenant_id}': {str(e)}")

    def delete_tenant_collection(
        self, tenant_id: str, collection_type: str = "documents"
    ) -> None:
        """
        Delete a tenant's collection.
        
        WARNING: This permanently deletes all tenant data!
        
        Args:
            tenant_id: Tenant identifier
            collection_type: Type of collection
        """
        collection_name = self.get_tenant_collection_name(tenant_id, collection_type)

        try:
            self.client.delete_collection(name=collection_name)
            print(f"Deleted collection for tenant '{tenant_id}': {collection_name}")
        except Exception as e:
            print(f"Error deleting collection: {str(e)}")

    def list_all_tenants(self) -> None:
        """List all tenant collections."""
        collections = self.client.list_collections()

        # Group by tenant
        tenants = {}
        for coll in collections:
            if coll.name.startswith("tenant_"):
                parts = coll.name.split("_", 2)
                if len(parts) >= 3:
                    tenant_id = parts[1]
                    collection_type = parts[2]

                    if tenant_id not in tenants:
                        tenants[tenant_id] = []
                    tenants[tenant_id].append({
                        "type": collection_type,
                        "count": coll.count()
                    })

        print("\nAll Tenants:")
        for tenant_id, collections in sorted(tenants.items()):
            print(f"\n  Tenant: {tenant_id}")
            for coll in collections:
                print(f"    - {coll['type']}: {coll['count']} documents")

    def delete_all_tenant_data(self, tenant_id: str) -> None:
        """
        Delete all collections for a tenant.
        
        Args:
            tenant_id: Tenant identifier
        """
        collections = self.client.list_collections()
        prefix = f"tenant_{tenant_id}_"

        deleted_count = 0
        for coll in collections:
            if coll.name.startswith(prefix):
                self.client.delete_collection(name=coll.name)
                deleted_count += 1

        print(f"Deleted {deleted_count} collections for tenant '{tenant_id}'")


if __name__ == "__main__":
    # Initialize multi-tenant manager
    chroma_tenant = ChromaMultiTenant(host="localhost", port=8000)

    # Create collections for multiple tenants
    print("="*80)
    print("Creating Tenant Collections")
    print("="*80)
    chroma_tenant.create_tenant_collection("company_a", "documents")
    chroma_tenant.create_tenant_collection("company_b", "documents")
    chroma_tenant.create_tenant_collection("company_c", "documents")

    # Company A: Tech company documents
    company_a_docs = [
        {
            "content": "Our AI-powered analytics platform helps businesses make data-driven decisions.",
            "metadata": {
                "type": "product",
                "department": "engineering",
                "confidential": True
            }
        },
        {
            "content": "Q4 revenue exceeded expectations with 45% growth in enterprise subscriptions.",
            "metadata": {
                "type": "financial",
                "department": "finance",
                "confidential": True
            }
        },
        {
            "content": "New employee onboarding process includes AI ethics training and security protocols.",
            "metadata": {
                "type": "hr",
                "department": "human_resources",
                "confidential": False
            }
        },
        {
            "content": "Machine learning model deployment pipeline automated with CI/CD integration.",
            "metadata": {
                "type": "technical",
                "department": "engineering",
                "confidential": False
            }
        },
    ]

    # Company B: Healthcare organization documents
    company_b_docs = [
        {
            "content": "Patient care protocols updated to include telemedicine best practices.",
            "metadata": {
                "type": "clinical",
                "department": "medical",
                "confidential": True
            }
        },
        {
            "content": "New medical imaging AI system improves diagnostic accuracy by 30%.",
            "metadata": {
                "type": "technology",
                "department": "radiology",
                "confidential": False
            }
        },
        {
            "content": "HIPAA compliance training mandatory for all staff members.",
            "metadata": {
                "type": "compliance",
                "department": "legal",
                "confidential": False
            }
        },
    ]

    # Company C: E-commerce company documents
    company_c_docs = [
        {
            "content": "Customer satisfaction scores improved after implementing AI chatbot support.",
            "metadata": {
                "type": "customer_service",
                "department": "support",
                "confidential": False
            }
        },
        {
            "content": "Inventory management system optimized using machine learning predictions.",
            "metadata": {
                "type": "operations",
                "department": "logistics",
                "confidential": True
            }
        },
        {
            "content": "Personalized product recommendations increased conversion rate by 25%.",
            "metadata": {
                "type": "marketing",
                "department": "marketing",
                "confidential": False
            }
        },
    ]

    # Add documents for each tenant
    print("\n" + "="*80)
    print("Adding Tenant Documents")
    print("="*80)
    chroma_tenant.add_tenant_documents("company_a", company_a_docs)
    chroma_tenant.add_tenant_documents("company_b", company_b_docs)
    chroma_tenant.add_tenant_documents("company_c", company_c_docs)

    # List all tenants
    print("\n" + "="*80)
    print("All Tenants in Database")
    print("="*80)
    chroma_tenant.list_all_tenants()

    # Get stats for each tenant
    print("\n" + "="*80)
    print("Tenant Statistics")
    print("="*80)
    chroma_tenant.get_tenant_stats("company_a")
    chroma_tenant.get_tenant_stats("company_b")
    chroma_tenant.get_tenant_stats("company_c")

    # Example 1: Search Company A's documents
    print("\n" + "="*80)
    print("Example 1: Search Company A Documents")
    print("="*80)
    chroma_tenant.search_tenant_documents(
        tenant_id="company_a",
        query="artificial intelligence and analytics",
        n_results=3
    )

    # Example 2: Search Company B's documents
    print("\n" + "="*80)
    print("Example 2: Search Company B Documents")
    print("="*80)
    chroma_tenant.search_tenant_documents(
        tenant_id="company_b",
        query="medical technology and diagnostics",
        n_results=3
    )

    # Example 3: Search Company C's documents
    print("\n" + "="*80)
    print("Example 3: Search Company C Documents")
    print("="*80)
    chroma_tenant.search_tenant_documents(
        tenant_id="company_c",
        query="customer experience and AI",
        n_results=3
    )

    # Example 4: Search with metadata filter
    print("\n" + "="*80)
    print("Example 4: Search Company A - Engineering Department Only")
    print("="*80)
    chroma_tenant.search_tenant_documents(
        tenant_id="company_a",
        query="technology and systems",
        metadata_filter={"department": "engineering"},
        n_results=3
    )

    # Example 5: Demonstrate data isolation
    print("\n" + "="*80)
    print("Example 5: Data Isolation - Same Query, Different Tenants")
    print("="*80)
    query = "AI and machine learning"

    print("\nSearching across all tenants with the same query:")
    chroma_tenant.search_tenant_documents("company_a", query, n_results=2)
    chroma_tenant.search_tenant_documents("company_b", query, n_results=2)
    chroma_tenant.search_tenant_documents("company_c", query, n_results=2)

    # Example 6: Search non-confidential documents only
    print("\n" + "="*80)
    print("Example 6: Search Company A - Non-Confidential Only")
    print("="*80)
    chroma_tenant.search_tenant_documents(
        tenant_id="company_a",
        query="company information",
        metadata_filter={"confidential": False},
        n_results=3
    )

    # Cleanup example (commented out for safety)
    # print("\n" + "="*80)
    # print("Cleanup: Delete Tenant Data")
    # print("="*80)
    # chroma_tenant.delete_all_tenant_data("company_c")
    # chroma_tenant.list_all_tenants()
