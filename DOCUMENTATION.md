# OFB Intelligent Document Discovery System

**Version 2.1** | Production-Grade RAG + Vector Search with Dynamic Database Management

---

## Table of Contents

1.  [System Overview](#1-system-overview)
2.  [Architecture](#2-architecture)
3.  [Project Structure](#3-project-structure)
4.  [Installation & Setup](#4-installation--setup)
5.  [Running the Application](#5-running-the-application)
6.  [Environment Variables Reference](#6-environment-variables-reference)
7.  [SQLite Configuration Database](#7-sqlite-configuration-database)
8.  [API Reference](#8-api-reference)
9.  [Admin Panel Guide](#9-admin-panel-guide)
10. [Dynamic Database Management](#10-dynamic-database-management)
11. [Table Selection & Configuration](#11-table-selection--configuration)
12. [Vectorization Pipeline](#12-vectorization-pipeline)
13. [File Processing & Extraction](#13-file-processing--extraction)
14. [RAG Pipeline (LangGraph)](#14-rag-pipeline-langgraph)
15. [Background Sync & Update Detection](#15-background-sync--update-detection)
16. [Security Model](#16-security-model)
17. [Fallback Mechanisms](#17-fallback-mechanisms)
18. [End-to-End Workflow Examples](#18-end-to-end-workflow-examples)
19. [Troubleshooting](#19-troubleshooting)
20. [Design Principles](#20-design-principles)

---

## 1. System Overview

The OFB Intelligent Document Discovery System is a production-grade FastAPI application that provides:

- **Vector Search**: Semantic similarity search across multiple MySQL databases using FAISS
- **RAG (Retrieval-Augmented Generation)**: LangGraph-orchestrated question answering with reranking
- **File Vectorization**: Automatic extraction and vectorization of files (PDF, DOCX, TXT, CSV, XLSX, MD) referenced in database tables
- **Dynamic Database Management**: Register, discover, and manage any number of MySQL databases at runtime
- **Admin Panel**: Web-based UI for database connections, table selection, and admin user management
- **Background Sync**: APScheduler-based automatic incremental and full rescan synchronization
- **LLM Schema Discovery**: Automatic table configuration via LLM analysis of database schemas

### Technology Stack

| Layer | Technology |
|---|---|
| Web Framework | FastAPI + Uvicorn |
| Vector Store | FAISS (faiss-cpu) |
| LLM / Embeddings | Ollama (llama3.1:8b + nomic-embed-text) |
| RAG Orchestration | LangGraph + LangChain |
| Reranking | FlashRank (ms-marco-MiniLM-L-12-v2) |
| Data Sources | MySQL (any number of databases) |
| Config Storage | SQLite (WAL mode, Fernet encryption) |
| Auth | bcrypt (passwords) + PyJWT (sessions) |
| Background Jobs | APScheduler |
| File Extraction | PyPDF2, python-docx, openpyxl, csv |
| Settings | Pydantic Settings + .env |

---

## 2. Architecture

### High-Level System Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                     Frontend (Mytest :8888)                          │
│  ┌──────────┐  ┌─────────────────────┐  ┌───────────────────────┐  │
│  │ Chat UI  │  │ Admin Panel         │  │ Toggle: Local / OFB   │  │
│  │          │  │  - Login Overlay     │  │                       │  │
│  │          │  │  - DB Management     │  │                       │  │
│  │          │  │  - Table Selection   │  │                       │  │
│  └──────────┘  └─────────────────────┘  └───────────────────────┘  │
└──────────────────────┬──────────────────────────────────────────────┘
                       │ HTTP (API Key or JWT)
                       ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   OFB FastAPI Backend (:8000)                        │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                     Route Layer                              │   │
│  │  admin.py  databases.py  search.py  rag.py  index.py       │   │
│  │  discover.py  health.py  status.py  files.py  sync_status  │   │
│  └────────────────────────────┬────────────────────────────────┘   │
│                               │                                     │
│  ┌────────────────────────────▼────────────────────────────────┐   │
│  │                    Core Services Layer                       │   │
│  │                                                              │   │
│  │  ┌─────────────┐  ┌──────────────┐  ┌────────────────────┐ │   │
│  │  │ config_db   │  │ admin_auth   │  │ connection_store   │ │   │
│  │  │ (SQLite)    │  │ (JWT/bcrypt) │  │ (SQLite-backed)    │ │   │
│  │  └──────┬──────┘  └──────────────┘  └────────────────────┘ │   │
│  │         │                                                    │   │
│  │  ┌──────▼──────┐  ┌──────────────┐  ┌────────────────────┐ │   │
│  │  │ schema      │  │ vectorizer   │  │ search_engine      │ │   │
│  │  │ _config     │  │ (embed +     │  │ (FAISS + metadata) │ │   │
│  │  │ + intel.    │  │  index)      │  │                    │ │   │
│  │  └─────────────┘  └──────────────┘  └────────────────────┘ │   │
│  │                                                              │   │
│  │  ┌─────────────┐  ┌──────────────┐  ┌────────────────────┐ │   │
│  │  │ rag_graph   │  │ sync_service │  │ file_processor     │ │   │
│  │  │ (LangGraph) │  │ (APScheduler)│  │ (ThreadPool)       │ │   │
│  │  └─────────────┘  └──────────────┘  └────────────────────┘ │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                               │                                     │
│  ┌────────────────────────────▼────────────────────────────────┐   │
│  │                    External Dependencies                     │   │
│  │                                                              │   │
│  │  ┌──────────┐  ┌──────────────┐  ┌───────────────────────┐ │   │
│  │  │ SQLite   │  │ MySQL        │  │ Ollama                │ │   │
│  │  │ Config   │  │ ofbdb        │  │ llama3.1:8b (RAG)     │ │   │
│  │  │ Database │  │ misofb       │  │ nomic-embed-text      │ │   │
│  │  │          │  │ vendor_onb.  │  │ (embeddings)          │ │   │
│  │  │          │  │ [any DB]     │  │                       │ │   │
│  │  └──────────┘  └──────────────┘  └───────────────────────┘ │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

### Data Flow: Query to Answer

```
User Question
      │
      ▼
[Query Rewrite] ──LLM──> Optimized query
      │
      ▼
[Vector Search] ──FAISS──> Top-K similar documents (DB records + file chunks)
      │
      ▼
[Reranking] ──FlashRank──> Re-ordered by cross-encoder relevance
      │
      ▼
[Document Grading] ──LLM──> Filter: keep only relevant docs
      │
      ▼
[Answer Generation] ──LLM──> Final answer with source citations
      │
      ▼
Response (answer + sources + metadata)
```

---

## 3. Project Structure

```
files (1)/
├── app/
│   ├── __init__.py
│   ├── main.py                    # FastAPI app factory, lifespan, middleware
│   ├── config.py                  # Pydantic Settings (reads .env)
│   ├── dependencies.py            # FastAPI DI: auth, singletons
│   ├── exceptions.py              # Centralised exception hierarchy
│   │
│   ├── core/                      # Core business logic
│   │   ├── config_db.py           # SQLite manager (CRUD, Fernet encryption)
│   │   ├── admin_auth.py          # bcrypt hashing, JWT, dual-source auth
│   │   ├── connection_store.py    # Thread-safe DB credential store (SQLite-backed)
│   │   ├── schema_config.py       # Table config registry (handcrafted + SQLite + JSON)
│   │   ├── schema_intelligence.py # LLM-based schema discovery
│   │   ├── document_builder.py    # Row → text conversion (per-table templates)
│   │   ├── embedder.py            # Ollama embedding client
│   │   ├── vectorizer.py          # Batch embed + index orchestrator
│   │   ├── vector_store.py        # FAISS wrapper (upsert, delete, search)
│   │   ├── search_engine.py       # High-level search API
│   │   ├── database.py            # MySQL connection helper
│   │   ├── lc_embeddings.py       # LangChain Ollama embeddings adapter
│   │   ├── lc_llm.py              # LangChain Ollama LLM adapter
│   │   ├── lc_loaders.py          # LangChain document loaders (PDF, DOCX, etc.)
│   │   ├── lc_reranker.py         # FlashRank reranker wrapper
│   │   ├── lc_vector_store.py     # LangChain FAISS vector store adapter
│   │   └── interfaces.py          # Abstract base classes
│   │
│   ├── models/                    # Pydantic request/response schemas
│   │   ├── requests.py            # SearchRequest, RagAskRequest, AdminLoginRequest, etc.
│   │   └── responses.py           # SearchResponse, RagResponse, AdminUserResponse, etc.
│   │
│   ├── routes/                    # FastAPI route handlers
│   │   ├── admin.py               # Login, user CRUD, table selection
│   │   ├── databases.py           # Database connection CRUD
│   │   ├── search.py              # POST /search
│   │   ├── rag.py                 # POST /rag/ask
│   │   ├── index.py               # POST /index (vectorization trigger)
│   │   ├── discover.py            # POST /discover (LLM schema discovery)
│   │   ├── health.py              # GET /health
│   │   ├── status.py              # GET /status
│   │   ├── files.py               # POST /files/process
│   │   └── sync_status.py         # GET /sync/status, POST /sync/trigger
│   │
│   └── services/                  # Background services
│       ├── sync_service.py        # Incremental + full rescan orchestrator
│       ├── sync_state.py          # Sync state persistence
│       ├── scheduler.py           # APScheduler setup
│       ├── file_processor.py      # Multi-threaded file text extraction
│       ├── file_extractor.py      # Per-format text extractors
│       └── rag_graph.py           # LangGraph RAG state machine
│
├── data/
│   └── ofb_config.db              # SQLite config database (auto-created)
│
├── configs/                       # Legacy JSON config directory
│   └── connections.json           # (auto-migrated to SQLite on startup)
│
├── faiss_index/                   # FAISS index persistence
│   ├── ofb_documents.faiss        # FAISS binary index
│   └── ofb_documents.meta         # Document metadata (JSON)
│
├── uploads/                       # Uploaded/referenced files
│   └── vendor_onboarding/         # Per-database file directories
│
├── flashrank_cache/               # FlashRank model cache
│
├── .env                           # Environment configuration
├── .env.example                   # Template for .env
├── requirements.txt               # Python dependencies
├── cli.py                         # CLI tool (legacy)
└── DOCUMENTATION.md               # This file
```

---

## 4. Installation & Setup

### Prerequisites

| Requirement | Version | Purpose |
|---|---|---|
| Python | 3.10+ | Runtime |
| Ollama | Latest | LLM + embedding server |
| MySQL | 5.7+ / 8.0+ | Source databases |
| pip | Latest | Package management |

### Step-by-Step

```bash
# 1. Navigate to the project
cd "/home/wadmin/Pictures/files (1)"

# 2. Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env with your MySQL credentials, Ollama URL, etc.

# 5. Pull Ollama models (one-time, requires Ollama running)
ollama pull nomic-embed-text       # ~274 MB — embeddings
ollama pull llama3.1:8b            # ~4.7 GB — RAG + schema intelligence

# 6. Start the OFB backend
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### First Startup Sequence

On the very first run, the system automatically:

1. Creates `data/ofb_config.db` (SQLite WAL mode)
2. Generates a Fernet encryption key (logged to console — save it to `.env`)
3. Seeds the super admin user from `.env` (`SUPER_ADMIN_USERNAME` / `SUPER_ADMIN_PASSWORD`)
4. Seeds default MySQL connections (`ofbdb`, `misofb`) from `.env`
5. Seeds handcrafted table configurations (11 ofbdb tables, 7 misofb tables)
6. Migrates any existing `configs/connections.json` to SQLite
7. Loads the FAISS index (if one exists from a prior run)
8. Starts the background sync scheduler

---

## 5. Running the Application

### OFB Backend Only

```bash
cd "/home/wadmin/Pictures/files (1)"
source venv/bin/activate
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Access points:
- API Docs: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
- OpenAPI JSON: http://localhost:8000/openapi.json

### Both Backends (OFB + Mytest)

The Mytest frontend provides a Chat UI and Admin Panel that can switch between the local Mytest backend and the OFB backend via a toggle.

**Terminal 1 — OFB Backend (port 8000):**
```bash
cd "/home/wadmin/Pictures/files (1)"
source venv/bin/activate
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

**Terminal 2 — Mytest Backend (port 8888):**
```bash
cd /home/wadmin/Pictures/Mytest
# Create venv if first time:
# python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt
source venv/bin/activate   # or use global Python if no venv
uvicorn app.main:app --host 0.0.0.0 --port 8888 --reload
```

Access points:
- Admin Panel: http://localhost:8888/admin
- Chat UI: http://localhost:8888/chat
- OFB API Docs: http://localhost:8000/docs

### Quick Start (Both in One Terminal)

```bash
# Start OFB in background
cd "/home/wadmin/Pictures/files (1)" && ./venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000 &

# Start Mytest in foreground
cd /home/wadmin/Pictures/Mytest && uvicorn app.main:app --host 0.0.0.0 --port 8888
```

### Frontend Toggle Behaviour

The admin panel at `http://localhost:8888/admin` has a **Local / OFB** toggle:

| Position | Behaviour |
|---|---|
| **Local** (left) | All API calls go to `localhost:8888` (Mytest backend) |
| **OFB** (right) | API calls go to `localhost:8000` (OFB backend), using API key + JWT |

When switching to OFB mode:
1. If no JWT is cached, a login overlay appears
2. Enter admin credentials (default: `admin` / `OfbAdmin@2026`)
3. JWT is stored in `localStorage` and used for subsequent requests
4. Database management and table selection sections become visible

---

## 6. Environment Variables Reference

### API & Server

| Variable | Default | Description |
|---|---|---|
| `API_KEYS` | `ofb-dev-key-2026` | Comma-separated API keys |
| `APP_HOST` | `0.0.0.0` | Server bind host |
| `APP_PORT` | `8000` | Server bind port |
| `LOG_LEVEL` | `INFO` | Logging level (DEBUG, INFO, WARNING, ERROR) |
| `CORS_ORIGINS` | `*` | Allowed CORS origins |
| `DEBUG` | `false` | Debug mode |

### LLM & Embeddings

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `ollama` | Provider: `ollama`, `openai`, `anthropic` |
| `OLLAMA_BASE_URL` | `http://192.168.0.207:8080` | Ollama server URL |
| `OLLAMA_LLM_MODEL` | `llama3.1:8b` | LLM model name |
| `OLLAMA_EMBED_MODEL` | `nomic-embed-text:latest` | Embedding model name |
| `OLLAMA_TIMEOUT` | `120` | Request timeout (seconds) |
| `EMBEDDING_DIMS` | `768` | Embedding vector dimensions |

### MySQL Defaults

| Variable | Default | Description |
|---|---|---|
| `OFBDB_HOST` | `127.0.0.1` | ofbdb MySQL host |
| `OFBDB_PORT` | `3307` | ofbdb MySQL port |
| `OFBDB_USER` | `root` | ofbdb MySQL user |
| `OFBDB_PASSWORD` | `rootpassword` | ofbdb MySQL password |
| `OFBDB_DATABASE` | `mydb` | ofbdb database name |
| `MISOFB_HOST` | `127.0.0.1` | misofb MySQL host |
| `MISOFB_PORT` | `3307` | misofb MySQL port |
| `MISOFB_USER` | `root` | misofb MySQL user |
| `MISOFB_PASSWORD` | `rootpassword` | misofb MySQL password |
| `MISOFB_DATABASE` | `misofb` | misofb database name |

### Admin & Security

| Variable | Default | Description |
|---|---|---|
| `SUPER_ADMIN_USERNAME` | `admin` | Super admin login name |
| `SUPER_ADMIN_PASSWORD` | `OfbAdmin@2026` | Super admin password |
| `JWT_SECRET` | `ofb-jwt-secret-...` | JWT signing secret (**change in production**) |
| `JWT_EXPIRY_HOURS` | `8` | JWT token lifetime |
| `ENCRYPTION_KEY` | _(auto-generated)_ | Fernet key for encrypting DB passwords |
| `SQLITE_DB_PATH` | `./data/ofb_config.db` | SQLite config database path |

### FAISS & Vectorization

| Variable | Default | Description |
|---|---|---|
| `FAISS_PERSIST_DIR` | `./faiss_index` | Index storage directory |
| `FAISS_COLLECTION_NAME` | `ofb_documents` | Index collection name |
| `EMBED_BATCH_SIZE` | `16` | Embedding batch size |
| `FETCH_CHUNK_SIZE` | `500` | DB rows per fetch chunk |
| `NUM_EMBED_WORKERS` | `2` | Parallel embedding workers |
| `SAVE_EVERY_N` | `2000` | Save index every N documents |

### RAG

| Variable | Default | Description |
|---|---|---|
| `RAG_TOP_K` | `5` | Source documents for RAG |
| `RAG_MAX_TOKENS` | `2048` | Max answer tokens |
| `RAG_TEMPERATURE` | `0.1` | LLM sampling temperature |
| `RAG_RERANK_ENABLED` | `true` | Enable FlashRank reranking |

### Sync

| Variable | Default | Description |
|---|---|---|
| `SYNC_ENABLED` | `true` | Enable background sync |
| `SYNC_INTERVAL_SECONDS` | `120` | Incremental sync interval |
| `SYNC_FULL_RESCAN_HOUR` | `2` | Daily full rescan hour (0-23) |
| `SYNC_STATE_FILE` | `./sync_state.json` | Sync state persistence |

---

## 7. SQLite Configuration Database

**Location**: `data/ofb_config.db` (configurable via `SQLITE_DB_PATH`)
**Mode**: WAL (Write-Ahead Logging) for concurrent read/write safety

### Tables

#### `admin_users`

Stores admin credentials. Passwords hashed with bcrypt.

| Column | Type | Description |
|---|---|---|
| `id` | INTEGER PK | Auto-increment ID |
| `username` | TEXT UNIQUE | Login username |
| `password_hash` | TEXT | bcrypt hash (one-way) |
| `role` | TEXT | `superadmin` or `admin` |
| `is_active` | INTEGER | 1 = active, 0 = deactivated |
| `created_at` | TEXT | ISO timestamp |
| `updated_at` | TEXT | ISO timestamp |

#### `database_connections`

Replaces the former `configs/connections.json`. Passwords encrypted with Fernet.

| Column | Type | Description |
|---|---|---|
| `id` | INTEGER PK | Auto-increment ID |
| `name` | TEXT UNIQUE | Logical name (e.g., `ofbdb`, `vendor_onboarding`) |
| `host` | TEXT | MySQL host |
| `port` | INTEGER | MySQL port (default 3307) |
| `db_user` | TEXT | MySQL username |
| `db_password` | TEXT | Fernet-encrypted MySQL password |
| `database_name` | TEXT | Actual MySQL database name |
| `is_active` | INTEGER | 1 = active, 0 = soft-deleted |
| `created_at` | TEXT | ISO timestamp |
| `updated_at` | TEXT | ISO timestamp |

#### `table_configurations`

Central registry of all table schemas used for vectorization and search.

| Column | Type | Description |
|---|---|---|
| `id` | INTEGER PK | Auto-increment ID |
| `db_connection_name` | TEXT | References `database_connections.name` |
| `table_name` | TEXT | MySQL table name |
| `is_selected` | INTEGER | 1 = included in vectorization/search, 0 = excluded |
| `text_columns` | TEXT (JSON) | Columns whose content is embedded |
| `metadata_columns` | TEXT (JSON) | Columns stored as metadata |
| `pk_column` | TEXT | Primary key column name |
| `label` | TEXT | Human-readable label |
| `description` | TEXT | Table description |
| `date_column` | TEXT | Column used for incremental sync |
| `file_columns` | TEXT (JSON) | `[[path_col, type_col], ...]` for file references |
| `source` | TEXT | `handcrafted`, `auto`, or `manual` |
| `created_at` | TEXT | ISO timestamp |
| `updated_at` | TEXT | ISO timestamp |
| | | UNIQUE constraint on `(db_connection_name, table_name)` |

#### `system_settings`

Key-value store for runtime settings.

| Column | Type | Description |
|---|---|---|
| `key` | TEXT PK | Setting key |
| `value` | TEXT | Setting value |
| `updated_at` | TEXT | ISO timestamp |

---

## 8. API Reference

### Authentication

| Method | Header | Used By |
|---|---|---|
| API Key | `X-API-Key: ofb-dev-key-2026` | All data endpoints |
| JWT Token | `Authorization: Bearer <jwt>` | Admin endpoints |
| Either | Both accepted | Database management endpoints |

### Admin Endpoints

| Method | Endpoint | Auth | Description |
|---|---|---|---|
| POST | `/api/v1/admin/login` | None | Authenticate, receive JWT |
| GET | `/api/v1/admin/me` | JWT | Current admin info |
| GET | `/api/v1/admin/users` | JWT + superadmin | List admin users |
| POST | `/api/v1/admin/users` | JWT + superadmin | Create admin user |
| DELETE | `/api/v1/admin/users/{id}` | JWT + superadmin | Deactivate admin user |

#### Login Example

```bash
curl -X POST http://localhost:8000/api/v1/admin/login \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "OfbAdmin@2026"}'
```

Response:
```json
{"token": "eyJ...", "username": "admin", "role": "superadmin", "expires_in_hours": 8}
```

### Database Management Endpoints

| Method | Endpoint | Auth | Description |
|---|---|---|---|
| GET | `/api/v1/databases` | API Key or JWT | List all registered databases |
| POST | `/api/v1/databases` | API Key or JWT | Register new database |
| PUT | `/api/v1/databases/{name}` | API Key or JWT | Update connection details |
| DELETE | `/api/v1/databases/{name}` | API Key or JWT | Remove database + configs |

#### Register Database Example

```bash
curl -X POST http://localhost:8000/api/v1/databases \
  -H "X-API-Key: ofb-dev-key-2026" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "vendor_onboarding",
    "host": "127.0.0.1", "port": 3307,
    "user": "root", "password": "rootpassword",
    "database": "vendor_onboarding",
    "auto_discover": true
  }'
```

### Table Configuration Endpoints

| Method | Endpoint | Auth | Description |
|---|---|---|---|
| GET | `/api/v1/admin/databases/{name}/tables` | JWT | List tables with selection state |
| PUT | `/api/v1/admin/databases/{name}/tables` | JWT | Toggle individual table selection |
| POST | `/api/v1/admin/databases/{name}/tables/select-all` | JWT | Select all tables |
| POST | `/api/v1/admin/databases/{name}/tables/deselect-all` | JWT | Deselect all tables |

#### Toggle Tables Example

```bash
curl -X PUT http://localhost:8000/api/v1/admin/databases/ofbdb/tables \
  -H "Authorization: Bearer <jwt>" \
  -H "Content-Type: application/json" \
  -d '{"selections": {"vendor_debar": false, "certificate_report": true}}'
```

### Search Endpoints

| Method | Endpoint | Auth | Description |
|---|---|---|---|
| POST | `/api/v1/search` | API Key | Vector similarity search |
| POST | `/api/v1/rag/ask` | API Key | RAG question answering |

#### Search Example

```bash
curl -X POST http://localhost:8000/api/v1/search \
  -H "X-API-Key: ofb-dev-key-2026" \
  -H "Content-Type: application/json" \
  -d '{"query": "titanium alloy supplier", "top_k": 5, "db_filter": "vendor_onboarding"}'
```

#### RAG Example

```bash
curl -X POST http://localhost:8000/api/v1/rag/ask \
  -H "X-API-Key: ofb-dev-key-2026" \
  -H "Content-Type: application/json" \
  -d '{"query": "Which vendors have ISO 9001 certification?", "top_k": 5}'
```

### Indexing & Sync Endpoints

| Method | Endpoint | Auth | Description |
|---|---|---|---|
| POST | `/api/v1/index` | API Key | Start vectorization (background) |
| POST | `/api/v1/discover` | API Key | LLM schema discovery |
| GET | `/api/v1/sync/status` | API Key | Background sync status |
| POST | `/api/v1/sync/trigger` | API Key | Trigger manual sync |
| POST | `/api/v1/files/process` | API Key | Process files from a table |

### Other Endpoints

| Method | Endpoint | Auth | Description |
|---|---|---|---|
| GET | `/api/v1/status` | API Key | System status (databases, index size) |
| GET | `/docs` | None | Swagger UI (OpenAPI) |
| GET | `/redoc` | None | ReDoc documentation |

---

## 9. Admin Panel Guide

### Accessing the Admin Panel

1. Start both backends (OFB on :8000, Mytest on :8888)
2. Open http://localhost:8888/admin
3. Toggle the switch from **Local** to **OFB**
4. Enter admin credentials in the login overlay (default: `admin` / `OfbAdmin@2026`)
5. The Database Connections and Table Configuration sections appear

### Database Connections Section

- **View**: Table showing all registered databases with name, host, connection status (green/red dot), and table count
- **Add**: Click "+ Add Database" button to show the inline form. Fields: Name, Database, Host, Port, User, Password, Auto-discover toggle
- **Test**: Click "Test" on any row to verify connectivity
- **Delete**: Click "Delete" to remove (built-in databases `ofbdb` and `misofb` are protected)

### Table Configuration Section

1. Select a database from the dropdown
2. A checkbox grid shows all configured tables
3. Each table card shows: name, label, source badge (`handcrafted`/`auto`/`manual`), FILE badge (if file columns exist)
4. Toggle individual tables on/off — changes save immediately
5. Use "Select All" / "Deselect All" for bulk operations
6. Only **selected** tables are included in vectorization and search

---

## 10. Dynamic Database Management

### How to Add a New Database

**Method 1: Admin Panel UI**
1. Open Admin Panel → Database Connections → "+ Add Database"
2. Fill in connection details
3. Toggle "Auto-discover schema" to use LLM for automatic table configuration
4. Click "Save"

**Method 2: API**
```bash
curl -X POST http://localhost:8000/api/v1/databases \
  -H "X-API-Key: ofb-dev-key-2026" \
  -H "Content-Type: application/json" \
  -d '{"name": "student_db", "host": "127.0.0.1", "port": 3307,
       "user": "root", "password": "rootpassword",
       "database": "student_onboarding", "auto_discover": true}'
```

**Method 3: LLM Discovery**
```bash
curl -X POST http://localhost:8000/api/v1/discover \
  -H "X-API-Key: ofb-dev-key-2026" \
  -H "Content-Type: application/json" \
  -d '{"host": "127.0.0.1", "port": 3307, "user": "root",
       "password": "rootpassword", "database": "student_onboarding"}'
```

### What Happens During Registration

1. Connection credentials are saved to SQLite `database_connections` (password Fernet-encrypted)
2. Connection is tested (reports success/failure but registers either way)
3. If `auto_discover: true`, the LLM analyzes the database schema to identify:
   - Which columns contain searchable text
   - Which columns are metadata
   - Which columns hold file paths (PDF, DOCX, etc.)
   - Primary key, date column, label, and description
4. Table configurations are saved to SQLite `table_configurations`
5. The database appears in the admin panel and is ready for vectorization

### Database Lifecycle

```
Register  →  Discover Schema  →  Select Tables  →  Vectorize  →  Search/RAG
    ↕              ↕                   ↕                ↕
  Update       Re-discover          Toggle          Re-index
  Delete       (force=true)        on/off
```

---

## 11. Table Selection & Configuration

### Selection States

| State | `is_selected` | Behaviour |
|---|---|---|
| Selected | 1 | Included in vectorization, sync, and search |
| Deselected | 0 | Excluded from all operations |
| Not configured | N/A | Not yet discovered or manually added |

### Configuration Sources (Priority Order)

1. **Handcrafted** (`source: handcrafted`): Python code in `schema_config.py` for ofbdb/misofb tables. Always present. Can be deselected via SQLite.
2. **SQLite** (`source: auto` or `manual`): Stored in `table_configurations` table. Created by LLM discovery or manual API calls.
3. **Legacy JSON**: Configs in `configs/<db_name>.json`. Used as fallback for databases not yet in SQLite.

### `file_columns` Format

Tables with file references use the `file_columns` field to identify which columns hold file paths:

```json
[["file_path", "file_type"], ["cert_file", ""]]
```

Each pair is `[path_column, type_column]`. If there's no type column, use an empty string. The system uses this to:
- Locate files on disk
- Determine file format (from extension or type column)
- Extract text and vectorize it alongside the DB record

---

## 12. Vectorization Pipeline

### How Vectorization Works

```
POST /api/v1/index?db_filter=vendor_onboarding
        │
        ▼
┌─ For each selected table in the database: ─────────────────┐
│                                                             │
│  1. Connect to MySQL, SELECT * FROM table                   │
│  2. For each row:                                           │
│     a. Build document text (template or generic)            │
│     b. Generate document ID: "{db}.{table}.{pk}"            │
│     c. Extract metadata from metadata_columns               │
│  3. Batch embed texts via Ollama (nomic-embed-text)         │
│  4. Upsert vectors + metadata into FAISS index              │
│                                                             │
│  5. If table has file_columns:                              │
│     a. For each row with a non-empty file path:             │
│        - Resolve file path (plain, base64, URL-encoded)     │
│        - Load file via LangChain loaders                    │
│        - Chunk into ~1000-char segments                     │
│        - Embed each chunk separately                        │
│        - Upsert chunks: "{db}.{table}.{pk}.chunk_{i}"       │
│                                                             │
│  6. Save FAISS index to disk                                │
│  7. Reload search engine with new index                     │
└─────────────────────────────────────────────────────────────┘
```

### Document ID Format

| Type | Format | Example |
|---|---|---|
| DB record | `{db}.{table}.{pk_value}` | `vendor_onboarding.vendors.1` |
| File chunk | `{db}.{table}.{pk_value}.chunk_{i}` | `vendor_onboarding.file_master.3.chunk_0` |

Upsert semantics: if a document with the same ID already exists, it is replaced. This handles updates.

---

## 13. File Processing & Extraction

### Supported File Formats

| Extension | Extractor | Library |
|---|---|---|
| `.pdf` | PyPDF2 + LangChain PyPDFLoader | PyPDF2 |
| `.docx` | python-docx + LangChain Docx2txtLoader | python-docx, docx2txt |
| `.txt`, `.log`, `.rst` | Plain text reader (multi-encoding) | Built-in |
| `.md` | Text reader + LangChain TextLoader | Built-in |
| `.csv`, `.tsv` | CSV reader + LangChain CSVLoader | Built-in |
| `.xlsx`, `.xls` | openpyxl sheet-by-sheet reader | openpyxl |
| `.html`, `.htm` | HTML tag stripper | Built-in |

### Path Resolution

File paths stored in the database may be encoded. The system tries these in order:

1. **Plain path**: Use as-is if file exists
2. **Base64 decode**: Try base64 decoding the string
3. **URL decode**: Try URL-decoding (`%20` → space, etc.)
4. **Prefix scan**: Try common prefixes (`/var/www/`, `/opt/ofb/uploads/`, `./uploads/`)

### Chunking Strategy

Files are split into chunks using `RecursiveCharacterTextSplitter`:
- **Chunk size**: 1000 characters
- **Chunk overlap**: 200 characters
- **Separators**: `\n\n`, `\n`, `. `, ` `, `` (in order of preference)

Each chunk is embedded separately and stored with metadata linking it back to the source file and DB record.

---

## 14. RAG Pipeline (LangGraph)

The RAG pipeline is a multi-step LangGraph state machine:

### Steps

1. **Query Rewrite**: LLM rewrites the user's question for better retrieval
2. **Vector Retrieval**: FAISS top-K similarity search
3. **Reranking** (optional): FlashRank cross-encoder re-scores and re-orders results
4. **Document Grading**: LLM evaluates each document for relevance (yes/no)
5. **Answer Generation**: LLM generates the final answer using only relevant documents
6. **Source Citation**: Sources are attached to the response with scores and metadata

### Self-Corrective Behaviour

If no documents pass the grading step, the pipeline:
1. Rewrites the query differently
2. Retrieves new documents
3. Re-grades and re-generates

This self-corrective loop runs up to 2 times before returning a "no relevant information found" response.

---

## 15. Background Sync & Update Detection

### Two Sync Modes

| Mode | Trigger | What It Does |
|---|---|---|
| **Incremental** | Every 120 seconds (configurable) | For tables with a `date_column`, queries `WHERE date_column >= last_sync_at`. Only changed/new rows are re-embedded. |
| **Full Rescan** | Daily at 2 AM (configurable) | Re-reads ALL rows from ALL selected tables. Safety net for anything incremental sync missed. |

### What Happens When Data Changes

| Change Type | Detection Method | Result |
|---|---|---|
| New row inserted | Incremental sync (date_column check) | New vector added to FAISS |
| Existing row updated | Incremental sync (updated_at >= last sync) | Existing vector replaced (upsert by doc_id) |
| Row deleted | Full rescan (row no longer present) | Stale vector remains until full rescan rebuilds index |
| New file uploaded | When row with file_path is synced | File text extracted, chunked, and embedded |
| File content changed | Full rescan re-reads the file | File chunks replaced with new content |
| New database registered | `_reload_connections()` at start of each cycle | New DB added to sync scope |
| Table deselected | `get_all_configs()` filters by `is_selected` | Deselected tables skipped |

### Sync Error Isolation

Each table is synced independently. If one table fails (e.g., MySQL connection error), the others continue. Errors are logged and recorded in the sync state.

---

## 16. Security Model

### Password Security

| Data | Method | Details |
|---|---|---|
| Admin passwords | **bcrypt** | One-way hash with built-in salt. Cannot be reversed. |
| MySQL passwords | **Fernet** (AES-128-CBC) | Symmetric encryption. Key in `ENCRYPTION_KEY` env var. Auto-generated on first run. |

### Authentication Flow

```
Login Request → Check SQLite admin_users → Found? → Verify bcrypt hash
                                             │
                                             No
                                             │
                                             ▼
                               Check .env SUPER_ADMIN_* → Match? → Issue JWT
                                                           │
                                                           No → 401 Unauthorized
```

### JWT Structure

```json
{
  "sub": "admin",        // username
  "role": "superadmin",  // role
  "iat": 1773927814,     // issued at
  "exp": 1773956614      // expires (iat + JWT_EXPIRY_HOURS)
}
```

Signed with HS256 using `JWT_SECRET`.

### Role-Based Access Control

| Role | User CRUD | DB Management | Table Selection | Search/RAG |
|---|---|---|---|---|
| `superadmin` | Full | Full | Full | Full |
| `admin` | Read only | Full | Full | Full |
| API Key only | None | Full | None | Full |

---

## 17. Fallback Mechanisms

### SQLite Unavailable

| Component | Fallback |
|---|---|
| Admin login | `.env` `SUPER_ADMIN_*` credentials |
| DB connections | Empty store (system degrades gracefully) |
| Table configs | Handcrafted Python configs still work |
| System settings | Defaults from `config.py` |

### Encryption Key Lost

- Stored MySQL passwords become unreadable
- On next restart, default connections (`ofbdb`, `misofb`) are re-seeded from `.env`
- Dynamically registered databases must be re-registered

### Ollama Unavailable

- Search still works (uses pre-built FAISS index)
- RAG returns "LLM offline" message
- Sync skips cycles with a warning, retries next tick
- Schema discovery fails gracefully

### JSON → SQLite Migration

On first startup after the SQLite upgrade:
- `configs/connections.json` is automatically migrated
- Renamed to `connections.json.migrated`
- All connections preserved in SQLite

---

## 18. End-to-End Workflow Examples

### Example 1: Adding a New Vendor Onboarding Database

```bash
# Step 1: Register the database
curl -X POST http://localhost:8000/api/v1/databases \
  -H "X-API-Key: ofb-dev-key-2026" \
  -H "Content-Type: application/json" \
  -d '{"name": "vendor_onboarding", "host": "127.0.0.1", "port": 3307,
       "user": "root", "password": "rootpassword",
       "database": "vendor_onboarding", "auto_discover": true}'

# Step 2: (Optional) Verify and select tables
JWT=$(curl -s -X POST http://localhost:8000/api/v1/admin/login \
  -H "Content-Type: application/json" \
  -d '{"username":"admin","password":"OfbAdmin@2026"}' | python3 -c "import sys,json;print(json.load(sys.stdin)['token'])")

curl http://localhost:8000/api/v1/admin/databases/vendor_onboarding/tables \
  -H "Authorization: Bearer $JWT"

# Step 3: Vectorize
curl -X POST http://localhost:8000/api/v1/index \
  -H "X-API-Key: ofb-dev-key-2026" \
  -H "Content-Type: application/json" \
  -d '{"db_filter": "vendor_onboarding"}'

# Step 4: Search
curl -X POST http://localhost:8000/api/v1/search \
  -H "X-API-Key: ofb-dev-key-2026" \
  -H "Content-Type: application/json" \
  -d '{"query": "titanium alloy supplier", "top_k": 5, "db_filter": "vendor_onboarding"}'

# Step 5: Ask a question
curl -X POST http://localhost:8000/api/v1/rag/ask \
  -H "X-API-Key: ofb-dev-key-2026" \
  -H "Content-Type: application/json" \
  -d '{"query": "Which vendors have ISO 9001 certification?", "top_k": 5}'
```

### Example 2: Handling File-Bearing Tables

If a table like `file_master` contains file paths:

1. The `file_columns` field in `table_configurations` tells the system which columns hold file paths
2. During vectorization, files are automatically extracted and chunked
3. File chunks appear in search results alongside DB records
4. RAG can answer questions from both file content and related DB entries

Example: Searching for "BHEL quality management" returns:
- The `vendor_certifications` DB record (ISO 9001 cert details)
- The `file_master` entry pointing to the actual certificate file
- Chunks from the certificate file itself (surveillance audit results, manufacturing capabilities)

---

## 19. Troubleshooting

### Startup Issues

| Error | Cause | Solution |
|---|---|---|
| `Generated new Fernet encryption key` | No `ENCRYPTION_KEY` in `.env` | Copy the logged key to `.env` to persist across restarts |
| `Access denied for user 'root'` | Wrong MySQL credentials | Check `OFBDB_*` / `MISOFB_*` vars in `.env` |
| `Ollama unreachable` | Ollama server not running | Start Ollama: `ollama serve` |
| `Index may not exist yet` | First run, no FAISS index | Run `POST /api/v1/index` to create the index |

### Runtime Issues

| Error | Cause | Solution |
|---|---|---|
| `Invalid or expired JWT token` | JWT expired | Re-login via `POST /admin/login` |
| `Cannot delete built-in database` | Attempting to delete ofbdb/misofb | These are protected by design |
| `Discovery timed out` | Ollama LLM slow on schema analysis | Increase `OLLAMA_TIMEOUT` or run discovery again |
| `No connection found for 'X'` | Database not registered | Register via `POST /databases` |
| Search returns no results | Index empty or wrong `db_filter` | Check index count via status endpoint |

### Log Interpretation

| Log Message | Meaning |
|---|---|
| `Indexing complete. N docs in Xs. Index size: M live docs` | Vectorization finished successfully |
| `Synced N records from X.Y` | Incremental sync found and processed changes |
| `Sync failed for X.Y: DB connect failed` | MySQL connection issue for that database |
| `Ollama unreachable — skipping sync` | Embedding server down, will retry next cycle |
| `Vectorized N file chunks for X.Y` | Files from that table were successfully extracted and embedded |
| `File chunk vectorization failed for /path` | File not found or unreadable (non-fatal) |

---

## 20. Design Principles

### SOLID Principles

| Principle | Application |
|---|---|
| **Single Responsibility** | Each module owns one concern: `config_db.py` = SQLite CRUD, `admin_auth.py` = authentication, `file_extractor.py` = text extraction |
| **Open/Closed** | New file formats added by registering an extractor function. New LLM providers added via `LLMProvider` enum. |
| **Liskov Substitution** | `ConnectionStore` interface unchanged after SQLite migration — all consumers unaffected |
| **Interface Segregation** | Routes depend only on the functions they need (e.g., `require_api_key`, `get_search_engine`) |
| **Dependency Inversion** | FastAPI `Depends()` for all injections. Routes never directly import singletons. |

### Error Handling Strategy

- **Exception hierarchy**: All domain errors extend `AppError` with machine-readable `error_code` and HTTP status
- **Per-table isolation**: One table failing during sync does not block others
- **Graceful degradation**: Ollama down → search works, RAG returns clear error message
- **Fail-safe registration**: Connection test failure does not block database registration

### Thread Safety

- SQLite WAL mode allows concurrent readers with one writer
- Connection-per-call pattern prevents connection sharing across threads
- `ConnectionStore` uses `threading.Lock` for critical sections
- FAISS vector store has internal locking for upsert operations
- `ThreadPoolExecutor` used for parallel table processing and file extraction

### Security Practices

- Passwords never stored in plaintext (bcrypt for admin, Fernet for MySQL)
- JWT tokens have configurable short expiry (default 8 hours)
- API keys support rotation via comma-separated list in `.env`
- Super admin credentials maintained in both SQLite and `.env` for resilience
- Role-based access control for admin operations
