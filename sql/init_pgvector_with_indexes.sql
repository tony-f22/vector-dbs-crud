-- ============================================================================
-- pgVector Initialization with Comprehensive Indexing Examples
-- ============================================================================
-- This file demonstrates various indexing strategies for pgVector
-- including HNSW and IVFFlat indexes with different distance operators

-- Install the pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- ============================================================================
-- Strategy 1: Single Table with Metadata (Enhanced)
-- ============================================================================
-- Demonstrates: Vector + Metadata pattern with GIN and HNSW indexes

DROP TABLE IF EXISTS documents CASCADE;

CREATE TABLE documents (
    id bigserial PRIMARY KEY,
    content TEXT NOT NULL,
    embedding vector(384),
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create GIN index for efficient JSONB metadata filtering
CREATE INDEX documents_metadata_idx ON documents USING GIN (metadata);

-- Create HNSW index for vector similarity search (cosine distance)
-- HNSW parameters:
--   m: number of connections per layer (default: 16, range: 2-100)
--   ef_construction: size of dynamic candidate list (default: 64)
CREATE INDEX documents_embedding_idx ON documents 
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- Alternative: IVFFlat index (uncomment if preferred)
-- IVFFlat parameters:
--   lists: number of clusters (rule of thumb: rows/1000 for <1M rows)
-- CREATE INDEX documents_embedding_idx ON documents 
-- USING ivfflat (embedding vector_cosine_ops)
-- WITH (lists = 100);

-- Regular B-tree index for timestamp filtering
CREATE INDEX documents_created_at_idx ON documents(created_at);

COMMENT ON TABLE documents IS 'Strategy 1: Single table with vector embeddings and JSONB metadata';
COMMENT ON INDEX documents_embedding_idx IS 'HNSW index for fast cosine similarity search';
COMMENT ON INDEX documents_metadata_idx IS 'GIN index for efficient JSONB queries';

-- ============================================================================
-- Strategy 2: Multiple Tables (Organized)
-- ============================================================================
-- Demonstrates: Different tables for different embedding dimensions

-- Table 1: Product Descriptions (768 dimensions)
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

-- HNSW index for 768-dim embeddings
CREATE INDEX product_desc_embedding_idx ON product_descriptions 
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- Regular indexes for filtering
CREATE INDEX product_desc_category_idx ON product_descriptions(category);
CREATE INDEX product_desc_price_idx ON product_descriptions(price);

COMMENT ON TABLE product_descriptions IS 'Strategy 2: Product descriptions with 768-dim embeddings';

-- Table 2: Product Images (512 dimensions - for CLIP-like embeddings)
DROP TABLE IF EXISTS product_images CASCADE;

CREATE TABLE product_images (
    id bigserial PRIMARY KEY,
    product_id VARCHAR(50) NOT NULL,
    image_url TEXT NOT NULL,
    alt_text TEXT,
    embedding vector(512),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- HNSW index for 512-dim embeddings
CREATE INDEX product_img_embedding_idx ON product_images 
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

COMMENT ON TABLE product_images IS 'Strategy 2: Product images with 512-dim embeddings';

-- Table 3: User Profiles (384 dimensions)
DROP TABLE IF EXISTS user_profiles CASCADE;

CREATE TABLE user_profiles (
    id bigserial PRIMARY KEY,
    user_id VARCHAR(50) UNIQUE NOT NULL,
    bio TEXT,
    interests TEXT,
    embedding vector(384),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- HNSW index for 384-dim embeddings
CREATE INDEX user_profile_embedding_idx ON user_profiles 
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

COMMENT ON TABLE user_profiles IS 'Strategy 2: User profiles with 384-dim embeddings';

-- ============================================================================
-- Strategy 3: Multi-Tenant (Namespaces/Partitions)
-- ============================================================================
-- Demonstrates: Schema-based multi-tenancy with isolated data

-- Create schemas for different tenants
CREATE SCHEMA IF NOT EXISTS tenant_company_a;
CREATE SCHEMA IF NOT EXISTS tenant_company_b;
CREATE SCHEMA IF NOT EXISTS tenant_company_c;

-- Create identical table structure in each tenant schema
-- Tenant A
CREATE TABLE tenant_company_a.documents (
    id bigserial PRIMARY KEY,
    content TEXT NOT NULL,
    embedding vector(384),
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX documents_embedding_idx ON tenant_company_a.documents 
USING hnsw (embedding vector_cosine_ops);

CREATE INDEX documents_metadata_idx ON tenant_company_a.documents 
USING GIN (metadata);

-- Tenant B
CREATE TABLE tenant_company_b.documents (
    id bigserial PRIMARY KEY,
    content TEXT NOT NULL,
    embedding vector(384),
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX documents_embedding_idx ON tenant_company_b.documents 
USING hnsw (embedding vector_cosine_ops);

CREATE INDEX documents_metadata_idx ON tenant_company_b.documents 
USING GIN (metadata);

-- Tenant C
CREATE TABLE tenant_company_c.documents (
    id bigserial PRIMARY KEY,
    content TEXT NOT NULL,
    embedding vector(384),
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX documents_embedding_idx ON tenant_company_c.documents 
USING hnsw (embedding vector_cosine_ops);

CREATE INDEX documents_metadata_idx ON tenant_company_c.documents 
USING GIN (metadata);

COMMENT ON SCHEMA tenant_company_a IS 'Strategy 3: Isolated namespace for Company A';
COMMENT ON SCHEMA tenant_company_b IS 'Strategy 3: Isolated namespace for Company B';
COMMENT ON SCHEMA tenant_company_c IS 'Strategy 3: Isolated namespace for Company C';

-- ============================================================================
-- Advanced Indexing Examples
-- ============================================================================

-- Example: Multiple indexes for different distance metrics
DROP TABLE IF EXISTS multi_metric_vectors CASCADE;

CREATE TABLE multi_metric_vectors (
    id bigserial PRIMARY KEY,
    content TEXT,
    embedding vector(384)
);

-- Index for cosine similarity (most common for text embeddings)
CREATE INDEX multi_metric_cosine_idx ON multi_metric_vectors 
USING hnsw (embedding vector_cosine_ops);

-- Index for L2/Euclidean distance (common for image embeddings)
CREATE INDEX multi_metric_l2_idx ON multi_metric_vectors 
USING hnsw (embedding vector_l2_ops);

-- Index for inner product (useful for recommendations)
CREATE INDEX multi_metric_ip_idx ON multi_metric_vectors 
USING hnsw (embedding vector_ip_ops);

COMMENT ON TABLE multi_metric_vectors IS 'Example: Table with multiple distance metric indexes';

-- ============================================================================
-- Query Performance Settings
-- ============================================================================

-- For IVFFlat indexes, you can adjust the number of lists to probe at query time:
-- SET ivfflat.probes = 10;  -- Default is 1, higher = more accurate but slower

-- For HNSW indexes, you can adjust the search quality:
-- SET hnsw.ef_search = 40;  -- Default is 40, higher = more accurate but slower

-- ============================================================================
-- Useful Queries for Index Management
-- ============================================================================

-- View all indexes on vector columns
-- SELECT 
--     schemaname,
--     tablename,
--     indexname,
--     indexdef
-- FROM pg_indexes
-- WHERE indexdef LIKE '%vector%'
-- ORDER BY schemaname, tablename;

-- Check index size
-- SELECT
--     schemaname || '.' || tablename AS table_name,
--     indexname,
--     pg_size_pretty(pg_relation_size(indexrelid)) AS index_size
-- FROM pg_stat_user_indexes
-- WHERE indexname LIKE '%embedding%'
-- ORDER BY pg_relation_size(indexrelid) DESC;

-- Verify index usage with EXPLAIN
-- EXPLAIN (ANALYZE, BUFFERS) 
-- SELECT * FROM documents 
-- ORDER BY embedding <=> '[0.1, 0.2, ...]' 
-- LIMIT 5;

-- ============================================================================
-- Best Practices Summary
-- ============================================================================

/*
1. HNSW vs IVFFlat:
   - HNSW: Better recall, faster queries, more memory
   - IVFFlat: Good balance, configurable accuracy/speed trade-off
   - Flat: Only for small datasets (<10K vectors)

2. Distance Operators:
   - Cosine (<=>): Text embeddings, normalized vectors
   - L2 (<->): Image embeddings, spatial data
   - Inner Product (<#>): Recommendations, dot product similarity

3. Index Creation Timing:
   - Create indexes AFTER inserting data, not before
   - Building index on populated table is much faster

4. HNSW Parameters:
   - m=16: Good default (higher = better recall, more memory)
   - ef_construction=64: Good default (higher = better quality, slower build)

5. IVFFlat Parameters:
   - lists: rows/1000 for <1M rows, sqrt(rows) for >1M rows
   - Adjust ivfflat.probes at query time for accuracy/speed trade-off

6. Metadata Indexing:
   - Use GIN indexes for JSONB columns
   - Create B-tree indexes for frequently filtered columns

7. Multi-tenancy:
   - Use schemas for complete isolation
   - Use table partitioning for large-scale multi-tenancy
   - Consider row-level security for shared tables
*/
