import psycopg2
from pgvector.psycopg2 import register_vector
from sentence_transformers import SentenceTransformer


class PgVectorMultiTenant:
    """
    Strategy 3: Namespaces/Partitions (Multi-tenant)
    
    Demonstrates tenant isolation using PostgreSQL schemas:
    - Each tenant gets their own schema (namespace)
    - Same table structure across all tenants
    - Complete data isolation between tenants
    - Per-tenant scaling and management
    
    Benefits:
    ✅ Multi-tenancy with data isolation
    ✅ Per-tenant scaling and backup
    ✅ Easy tenant onboarding/offboarding
    ✅ Consistent structure across tenants
    
    Trade-offs:
    ❌ Requires namespace/schema support
    ❌ More complex database management
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

    def create_tenant_schema(self, tenant_id: str) -> None:
        """
        Create a new schema (namespace) for a tenant.
        
        Args:
            tenant_id: Unique identifier for the tenant (e.g., "company_a")
        """
        conn, cur = self.connect_db()
        try:
            schema_name = f"tenant_{tenant_id}"

            # Create schema
            cur.execute(f"CREATE SCHEMA IF NOT EXISTS {schema_name}")

            # Create documents table in tenant schema
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {schema_name}.documents (
                    id bigserial PRIMARY KEY,
                    content TEXT NOT NULL,
                    embedding vector(384),
                    metadata JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                -- Create indexes for the tenant
                CREATE INDEX IF NOT EXISTS documents_embedding_idx 
                ON {schema_name}.documents 
                USING hnsw (embedding vector_cosine_ops);
                
                CREATE INDEX IF NOT EXISTS documents_metadata_idx 
                ON {schema_name}.documents 
                USING GIN (metadata);
            """)

            conn.commit()
            print(f"Schema created for tenant: {tenant_id}")
        except Exception as e:
            print(f"Error creating tenant schema: {str(e)}")
        finally:
            cur.close()
            conn.close()

    def insert_tenant_documents(
        self,
        tenant_id: str,
        model: SentenceTransformer,
        documents: list[dict]
    ) -> None:
        """
        Insert documents for a specific tenant.
        
        Args:
            tenant_id: Tenant identifier
            model: SentenceTransformer model
            documents: List of document dicts with 'content' and 'metadata'
        """
        conn, cur = self.connect_db()
        try:
            schema_name = f"tenant_{tenant_id}"

            for doc in documents:
                content = doc["content"]
                metadata = doc.get("metadata", {})
                embedding = model.encode(content)

                cur.execute(
                    f"""INSERT INTO {schema_name}.documents (content, embedding, metadata) 
                        VALUES (%s, %s, %s)""",
                    (content, embedding.tolist(), psycopg2.extras.Json(metadata)),
                )

            conn.commit()
            print(f"Inserted {len(documents)} documents for tenant: {tenant_id}")
        except Exception as e:
            print(f"Error inserting documents: {str(e)}")
        finally:
            cur.close()
            conn.close()

    def search_tenant_documents(
        self,
        tenant_id: str,
        model: SentenceTransformer,
        query: str,
        limit: int = 5
    ) -> None:
        """
        Search documents within a specific tenant's namespace.
        
        Args:
            tenant_id: Tenant identifier
            model: SentenceTransformer model
            query: Search query
            limit: Number of results
        """
        conn, cur = self.connect_db()
        try:
            schema_name = f"tenant_{tenant_id}"
            query_embedding = model.encode(query)

            cur.execute(
                f"""SELECT id, content, metadata, 
                           1 - (embedding <=> %s) AS similarity
                    FROM {schema_name}.documents
                    ORDER BY similarity DESC 
                    LIMIT %s""",
                (query_embedding.tolist(), limit),
            )

            print(f"\nTenant: {tenant_id}")
            print(f"Query: '{query}'")
            print("Results:")

            for row in cur.fetchall():
                print(f"\n  ID: {row[0]}")
                print(f"  Content: {row[1]}")
                print(f"  Metadata: {row[2]}")
                print(f"  Similarity: {row[3]:.4f}")

        except Exception as e:
            print(f"Error searching documents: {str(e)}")
        finally:
            cur.close()
            conn.close()

    def get_tenant_stats(self, tenant_id: str) -> None:
        """
        Get statistics for a tenant's data.
        
        Args:
            tenant_id: Tenant identifier
        """
        conn, cur = self.connect_db()
        try:
            schema_name = f"tenant_{tenant_id}"

            cur.execute(f"SELECT COUNT(*) FROM {schema_name}.documents")
            doc_count = cur.fetchone()[0]

            cur.execute(
                f"""SELECT pg_size_pretty(pg_total_relation_size('{schema_name}.documents'))"""
            )
            table_size = cur.fetchone()[0]

            print(f"\nTenant: {tenant_id}")
            print(f"  Documents: {doc_count}")
            print(f"  Storage Size: {table_size}")

        except Exception as e:
            print(f"Error getting tenant stats: {str(e)}")
        finally:
            cur.close()
            conn.close()

    def delete_tenant_schema(self, tenant_id: str) -> None:
        """
        Delete a tenant's schema and all associated data.
        
        WARNING: This permanently deletes all tenant data!
        
        Args:
            tenant_id: Tenant identifier
        """
        conn, cur = self.connect_db()
        try:
            schema_name = f"tenant_{tenant_id}"
            cur.execute(f"DROP SCHEMA IF EXISTS {schema_name} CASCADE")
            conn.commit()
            print(f"Schema deleted for tenant: {tenant_id}")
        except Exception as e:
            print(f"Error deleting tenant schema: {str(e)}")
        finally:
            cur.close()
            conn.close()

    def list_all_tenants(self) -> None:
        """List all tenant schemas in the database."""
        conn, cur = self.connect_db()
        try:
            cur.execute("""
                SELECT schema_name 
                FROM information_schema.schemata 
                WHERE schema_name LIKE 'tenant_%'
                ORDER BY schema_name
            """)

            tenants = cur.fetchall()
            print("\nAll Tenants:")
            for tenant in tenants:
                tenant_id = tenant[0].replace("tenant_", "")
                print(f"  - {tenant_id}")

        except Exception as e:
            print(f"Error listing tenants: {str(e)}")
        finally:
            cur.close()
            conn.close()


if __name__ == "__main__":
    # Initialize the multi-tenant manager
    pg_tenant = PgVectorMultiTenant(
        user="myuser",
        password="mypassword",
        host="localhost",
        port=5433,
        database="mydb"
    )

    # Load embedding model
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    # Create schemas for multiple tenants
    print("="*80)
    print("Creating Tenant Schemas")
    print("="*80)
    pg_tenant.create_tenant_schema("company_a")
    pg_tenant.create_tenant_schema("company_b")
    pg_tenant.create_tenant_schema("company_c")

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
    ]

    # Insert documents for each tenant
    print("\n" + "="*80)
    print("Inserting Tenant Documents")
    print("="*80)
    pg_tenant.insert_tenant_documents("company_a", model, company_a_docs)
    pg_tenant.insert_tenant_documents("company_b", model, company_b_docs)
    pg_tenant.insert_tenant_documents("company_c", model, company_c_docs)

    # List all tenants
    print("\n" + "="*80)
    print("All Tenants in Database")
    print("="*80)
    pg_tenant.list_all_tenants()

    # Get stats for each tenant
    print("\n" + "="*80)
    print("Tenant Statistics")
    print("="*80)
    pg_tenant.get_tenant_stats("company_a")
    pg_tenant.get_tenant_stats("company_b")
    pg_tenant.get_tenant_stats("company_c")

    # Example 1: Search Company A's documents
    print("\n" + "="*80)
    print("Example 1: Search Company A Documents")
    print("="*80)
    pg_tenant.search_tenant_documents(
        tenant_id="company_a",
        model=model,
        query="artificial intelligence and analytics",
        limit=3
    )

    # Example 2: Search Company B's documents
    print("\n" + "="*80)
    print("Example 2: Search Company B Documents")
    print("="*80)
    pg_tenant.search_tenant_documents(
        tenant_id="company_b",
        model=model,
        query="medical technology and diagnostics",
        limit=3
    )

    # Example 3: Search Company C's documents
    print("\n" + "="*80)
    print("Example 3: Search Company C Documents")
    print("="*80)
    pg_tenant.search_tenant_documents(
        tenant_id="company_c",
        model=model,
        query="customer experience and AI",
        limit=3
    )

    # Example 4: Demonstrate data isolation
    print("\n" + "="*80)
    print("Example 4: Data Isolation - Same Query, Different Tenants")
    print("="*80)
    query = "AI and machine learning"

    print("\nSearching across all tenants with the same query:")
    pg_tenant.search_tenant_documents("company_a", model, query, limit=2)
    pg_tenant.search_tenant_documents("company_b", model, query, limit=2)
    pg_tenant.search_tenant_documents("company_c", model, query, limit=2)

    # Cleanup example (commented out for safety)
    # print("\n" + "="*80)
    # print("Cleanup: Delete Tenant Schema")
    # print("="*80)
    # pg_tenant.delete_tenant_schema("company_c")
    # pg_tenant.list_all_tenants()
