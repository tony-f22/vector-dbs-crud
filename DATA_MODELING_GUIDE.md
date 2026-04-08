# Vector Database Data Modeling Guide

This guide demonstrates three fundamental collection organization strategies for vector databases, implemented for both **pgVector (PostgreSQL)** and **ChromaDB**.

## 📚 Table of Contents

- [Overview](#overview)
- [Strategy 1: Single Collection with Metadata](#strategy-1-single-collection-with-metadata)
- [Strategy 2: Multiple Collections](#strategy-2-multiple-collections)
- [Strategy 3: Namespaces/Partitions (Multi-tenant)](#strategy-3-namespacespartitions-multi-tenant)
- [Indexing Strategies](#indexing-strategies)
- [Running the Examples](#running-the-examples)
- [Best Practices](#best-practices)

---

## Overview

Based on the presentation slides (207-279), this codebase implements three collection organization strategies:

| Strategy | Use Case | Benefits | Trade-offs |
|----------|----------|----------|------------|
| **Single Collection** | Homogeneous data, same embedding model | Simple, easy to manage | All vectors must have same dimensions |
| **Multiple Collections** | Different data types, different models | Optimized per data type | Can't search across collections |
| **Namespaces/Partitions** | Multi-tenant SaaS applications | Data isolation, per-tenant scaling | More complex management |

---

## Strategy 1: Single Collection with Metadata

### Pattern: Vector + Metadata

The fundamental pattern for vector databases combines embeddings with rich metadata for filtering.

```json
{
  "id": "doc_123",
  "vector": [0.23, -0.45, ..., 0.12],
  "metadata": {
    "title": "Vector Databases Guide",
    "category": "Technology",
    "tags": ["AI", "databases"]
  }
}
```

### Implementation

#### pgVector
- **File**: [`src/pgvector_examples/metadata_filtering_example.py`](src/pgvector_examples/metadata_filtering_example.py)
- **Table**: `documents` with JSONB metadata column
- **Indexes**: 
  - HNSW for vector similarity
  - GIN for JSONB metadata queries

```sql
CREATE TABLE documents (
    id bigserial PRIMARY KEY,
    content TEXT NOT NULL,
    embedding vector(384),
    metadata JSONB
);

CREATE INDEX documents_embedding_idx ON documents 
USING hnsw (embedding vector_cosine_ops);

CREATE INDEX documents_metadata_idx ON documents 
USING GIN (metadata);
```

#### ChromaDB
- **File**: [`src/chroma_db_examples/metadata_filtering_example.py`](src/chroma_db_examples/metadata_filtering_example.py)
- **Collection**: Single collection with metadata
- **Features**: Complex filtering with `$and`, `$or`, `$gt`, `$lt`, etc.

```python
collection.query(
    query_texts=["AI and databases"],
    where={"category": "Technology"},
    n_results=5
)
```

### When to Use

✅ Single embedding model  
✅ Homogeneous data  
✅ Need to filter by metadata  
✅ Simple architecture

---

## Strategy 2: Multiple Collections

### Pattern: Organized by Data Type

Different collections for different data types, each with optimized embedding models and dimensions.

```
collection: "product_descriptions"  (768 dims, text-embedding-3-small)
collection: "product_images"        (512 dims, CLIP)
collection: "user_profiles"         (384 dims, all-MiniLM-L6-v2)
```

### Implementation

#### pgVector
- **File**: [`src/pgvector_examples/multi_table_example.py`](src/pgvector_examples/multi_table_example.py)
- **Tables**: 
  - `product_descriptions` (768 dims)
  - `product_images` (512 dims)
  - `user_profiles` (384 dims)

```sql
CREATE TABLE product_descriptions (
    id bigserial PRIMARY KEY,
    product_id VARCHAR(50),
    title TEXT,
    description TEXT,
    embedding vector(768),
    category VARCHAR(100),
    price DECIMAL(10, 2)
);

CREATE INDEX product_desc_embedding_idx 
ON product_descriptions 
USING hnsw (embedding vector_cosine_ops);
```

#### ChromaDB
- **File**: [`src/chroma_db_examples/multi_collection_example.py`](src/chroma_db_examples/multi_collection_example.py)
- **Collections**: Separate collections per data type
- **Models**: Different SentenceTransformer models per collection

```python
# 768-dim model for product descriptions
model_768 = SentenceTransformer("all-mpnet-base-v2")
product_collection = client.create_collection("product_descriptions")

# 384-dim model for user profiles
model_384 = SentenceTransformer("all-MiniLM-L6-v2")
user_collection = client.create_collection("user_profiles")
```

### When to Use

✅ Multiple data types (text, images, audio)  
✅ Different embedding models  
✅ Optimized indexes per type  
✅ Clear separation of concerns

❌ Can't search across collections in single query

---

## Strategy 3: Namespaces/Partitions (Multi-tenant)

### Pattern: Tenant Isolation

Each tenant gets isolated namespace with identical structure.

```
namespace: "company_A"
  collection: "documents"
namespace: "company_B"
  collection: "documents"
```

### Implementation

#### pgVector
- **File**: [`src/pgvector_examples/multi_tenant_example.py`](src/pgvector_examples/multi_tenant_example.py)
- **Approach**: PostgreSQL schemas for tenant isolation
- **Structure**: Each schema has identical table structure

```sql
CREATE SCHEMA tenant_company_a;
CREATE SCHEMA tenant_company_b;

CREATE TABLE tenant_company_a.documents (
    id bigserial PRIMARY KEY,
    content TEXT,
    embedding vector(384),
    metadata JSONB
);

CREATE TABLE tenant_company_b.documents (
    id bigserial PRIMARY KEY,
    content TEXT,
    embedding vector(384),
    metadata JSONB
);
```

#### ChromaDB
- **File**: [`src/chroma_db_examples/multi_tenant_example.py`](src/chroma_db_examples/multi_tenant_example.py)
- **Approach**: Collection naming convention (`tenant_{id}_{type}`)
- **Note**: ChromaDB doesn't have native namespaces

```python
def get_tenant_collection_name(tenant_id: str, collection_type: str) -> str:
    return f"tenant_{tenant_id}_{collection_type}"

# Create tenant-specific collections
company_a_docs = client.create_collection("tenant_company_a_documents")
company_b_docs = client.create_collection("tenant_company_b_documents")
```

### When to Use

✅ SaaS applications  
✅ Multi-tenant architecture  
✅ Data isolation requirements  
✅ Per-tenant scaling needs

❌ More complex database management  
❌ Requires namespace support (or naming conventions)

---

## Indexing Strategies

### pgVector Indexes

#### HNSW (Hierarchical Navigable Small World)
- **Best for**: Production, read-heavy workloads
- **Speed**: Fastest queries
- **Memory**: High
- **Parameters**:
  - `m`: Connections per layer (default: 16)
  - `ef_construction`: Build quality (default: 64)

```sql
CREATE INDEX items_embedding_idx ON items 
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);
```

#### IVFFlat (Inverted File with Flat Compression)
- **Best for**: Balanced workloads
- **Speed**: Medium
- **Memory**: Medium
- **Parameters**:
  - `lists`: Number of clusters (rows/1000 for <1M rows)

```sql
CREATE INDEX items_embedding_idx ON items 
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);
```

### Distance Operators

| Operator | Distance Metric | Best For | Index Class |
|----------|----------------|----------|-------------|
| `<=>` | Cosine distance | Text embeddings | `vector_cosine_ops` |
| `<->` | L2/Euclidean | Images, spatial data | `vector_l2_ops` |
| `<#>` | Inner product | Recommendations | `vector_ip_ops` |

**Important**: Index operator class must match query operator!

```sql
-- Create index with cosine operator
CREATE INDEX idx ON items USING hnsw (embedding vector_cosine_ops);

-- This query WILL use the index ✅
SELECT * FROM items ORDER BY embedding <=> '[...]' LIMIT 5;

-- This query will NOT use the index ❌
SELECT * FROM items ORDER BY embedding <-> '[...]' LIMIT 5;
```

### Comprehensive SQL Init File

See [`sql/init_pgvector_with_indexes.sql`](sql/init_pgvector_with_indexes.sql) for:
- All three strategies with proper indexes
- Multiple distance metric examples
- Best practices and comments
- Performance tuning settings

---

## Running the Examples

### Prerequisites

1. **Start PostgreSQL with pgVector**:
   ```bash
   docker-compose up -d postgres
   ```

2. **Start ChromaDB**:
   ```bash
   docker-compose up -d chromadb
   ```

3. **Install dependencies**:
   ```bash
   uv sync
   ```

### pgVector Examples

```bash
# Strategy 1: Metadata Filtering
python src/pgvector_examples/metadata_filtering_example.py

# Strategy 2: Multiple Tables
python src/pgvector_examples/multi_table_example.py

# Strategy 3: Multi-tenant
python src/pgvector_examples/multi_tenant_example.py
```

### ChromaDB Examples

```bash
# Strategy 1: Metadata Filtering
python src/chroma_db_examples/metadata_filtering_example.py

# Strategy 2: Multiple Collections
python src/chroma_db_examples/multi_collection_example.py

# Strategy 3: Multi-tenant
python src/chroma_db_examples/multi_tenant_example.py
```

---

## Best Practices

### 1. Consistent Dimensionality
✅ All vectors in a collection/table must have the same dimensions  
✅ Match embedding model output to vector column size

```python
# Check model dimensions before creating table
model = SentenceTransformer("all-MiniLM-L6-v2")
dims = model.get_sentence_embedding_dimension()  # 384
print(f"Create table with vector({dims})")
```

### 2. Rich Metadata for Filtering
✅ Include relevant metadata for filtering  
✅ Use appropriate data types (JSONB in pgVector)  
✅ Create indexes on frequently filtered fields

```python
metadata = {
    "category": "Technology",
    "tags": ["AI", "databases"],
    "year": 2024,
    "confidential": False
}
```

### 3. Proper Chunking
✅ Chunk documents to 256-512 tokens  
✅ Use overlap (50-100 tokens) between chunks  
✅ Preserve context in metadata

### 4. Index Creation Timing
✅ Insert data FIRST, then create indexes  
✅ Building index on populated table is much faster  
❌ Don't create index on empty table then insert

```sql
-- Good ✅
INSERT INTO items (content, embedding) VALUES (...);
CREATE INDEX items_embedding_idx ON items USING hnsw (embedding vector_cosine_ops);

-- Bad ❌
CREATE INDEX items_embedding_idx ON items USING hnsw (embedding vector_cosine_ops);
INSERT INTO items (content, embedding) VALUES (...);
```

### 5. Choose the Right Strategy

| Scenario | Recommended Strategy |
|----------|---------------------|
| Single app, one embedding model | Strategy 1: Single Collection |
| Multiple data types (text, images) | Strategy 2: Multiple Collections |
| SaaS with multiple customers | Strategy 3: Multi-tenant |
| RAG application | Strategy 1 with metadata filtering |
| E-commerce with products + users | Strategy 2: Separate collections |

### 6. Monitor Performance

```sql
-- Check index usage
EXPLAIN (ANALYZE, BUFFERS) 
SELECT * FROM items 
ORDER BY embedding <=> '[...]' 
LIMIT 5;

-- Check index size
SELECT pg_size_pretty(pg_relation_size('items_embedding_idx'));
```

---

## Embedding Model Dimensions

Common models and their dimensions:

| Model | Dimensions | Use Case |
|-------|-----------|----------|
| `all-MiniLM-L6-v2` | 384 | Fast, lightweight text |
| `all-mpnet-base-v2` | 768 | Higher quality text |
| `text-embedding-3-small` | 1536 | OpenAI, high quality |
| `CLIP-ViT-B-32` | 512 | Images and text |

---

## Troubleshooting

### Error: "can't adapt type 'numpy.ndarray'"
**Solution**: Install and register pgvector adapter
```python
from pgvector.psycopg2 import register_vector
register_vector(conn)
```

### Error: "expected 768 dimensions, not 384"
**Solution**: Match vector column size to model output
```sql
-- Check model dimensions first
-- Then create table with matching size
CREATE TABLE items (embedding vector(384));  -- Match model dims
```

### Index not being used
**Solution**: Ensure operator matches index class
```sql
-- Index with cosine ops
CREATE INDEX idx USING hnsw (embedding vector_cosine_ops);

-- Query must use <=> operator
SELECT * FROM items ORDER BY embedding <=> '[...]';
```

---

## References

- Presentation: [`presentation/Vector-Databases-Presentation-Part-A.md`](presentation/Vector-Databases-Presentation-Part-A.md)
- pgVector Documentation: https://github.com/pgvector/pgvector
- ChromaDB Documentation: https://docs.trychroma.com/
- SentenceTransformers: https://www.sbert.net/

---

## Summary

This guide demonstrates three fundamental data modeling strategies for vector databases:

1. **Single Collection**: Simple, metadata-rich pattern for homogeneous data
2. **Multiple Collections**: Organized approach for different data types and models
3. **Multi-tenant**: Isolated namespaces for SaaS applications

Each strategy is implemented for both pgVector and ChromaDB with comprehensive examples, proper indexing, and best practices.

Choose the strategy that best fits your use case, and refer to the example files for implementation details.
