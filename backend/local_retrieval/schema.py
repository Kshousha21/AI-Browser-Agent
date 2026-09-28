"""Separate disposable retrieval schema; never migrates the ingestion database."""
SCHEMA_VERSION = 1
APPLICATION_ID = 0x48594231
DDL = (
    "CREATE TABLE index_metadata (singleton INTEGER PRIMARY KEY CHECK(singleton=1), identity_json TEXT NOT NULL, pipeline TEXT NOT NULL, source_pipeline TEXT NOT NULL, generation TEXT NOT NULL)",
    "CREATE TABLE indexed_documents (document_id TEXT PRIMARY KEY, fingerprint TEXT NOT NULL)",
    """CREATE TABLE indexed_chunks (
        id INTEGER PRIMARY KEY, chunk_id TEXT NOT NULL UNIQUE,
        document_id TEXT NOT NULL REFERENCES indexed_documents(document_id) ON DELETE CASCADE,
        chunk_index INTEGER NOT NULL, page_start INTEGER NOT NULL, page_end INTEGER NOT NULL,
        char_start INTEGER NOT NULL, char_end INTEGER NOT NULL, text TEXT NOT NULL,
        trust TEXT NOT NULL, text_hash TEXT NOT NULL, fingerprint TEXT NOT NULL,
        UNIQUE(document_id,chunk_index))""",
    "CREATE TABLE chunk_embeddings (chunk_id TEXT PRIMARY KEY REFERENCES indexed_chunks(chunk_id) ON DELETE CASCADE, vector BLOB NOT NULL)",
    "CREATE VIRTUAL TABLE chunk_fts USING fts5(text, content='indexed_chunks', content_rowid='id', tokenize='unicode61 remove_diacritics 2')",
    "CREATE TRIGGER chunk_insert AFTER INSERT ON indexed_chunks BEGIN INSERT INTO chunk_fts(rowid,text) VALUES(new.id,new.text); END",
    "CREATE TRIGGER chunk_delete AFTER DELETE ON indexed_chunks BEGIN INSERT INTO chunk_fts(chunk_fts,rowid,text) VALUES('delete',old.id,old.text); END",
)
