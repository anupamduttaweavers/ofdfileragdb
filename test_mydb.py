#!/usr/bin/env python3
"""
test_mydb.py — End-to-end verification for mydb
─────────────────────────────────────────────────
Tests:
  1. Direct MySQL connection and row counts
  2. FAISS index presence and document counts
  3. Semantic search (vector) queries against mydb data
  4. RAG ask queries to verify the full pipeline

Usage:
  python test_mydb.py                  # run all tests
  python test_mydb.py --db-only        # only test DB connection
  python test_mydb.py --faiss-only     # only test FAISS index
  python test_mydb.py --search-only    # only test search queries
  python test_mydb.py --rag-only       # only test RAG pipeline
  python test_mydb.py --api            # test via HTTP API (server must be running)
"""

import argparse
import json
import os
import sys
import time

from dotenv import load_dotenv
load_dotenv()

MYDB_HOST     = os.getenv("OFBDB_HOST", "127.0.0.1")
MYDB_PORT     = int(os.getenv("OFBDB_PORT", "3307"))
MYDB_USER     = os.getenv("OFBDB_USER", "root")
MYDB_PASSWORD = os.getenv("OFBDB_PASSWORD", "rootpassword")
MYDB_DATABASE = os.getenv("OFBDB_DATABASE", "mydb")
API_BASE      = os.getenv("API_BASE", "http://localhost:8000")
API_KEY       = os.getenv("API_KEYS", "ofb-dev-key-2026").split(",")[0]

PASS = "\033[92m PASS \033[0m"
FAIL = "\033[91m FAIL \033[0m"
WARN = "\033[93m WARN \033[0m"
INFO = "\033[96m INFO \033[0m"

KEY_TABLES = [
    "vendor_master",
    "vendor_user",
    "vendor_financial",
    "vendor_technical",
    "certificate_report",
    "certificate_report_product",
    "factory_master",
    "factory_user",
    "item_master",
    "advertisement",
    "advertisement_items",
    "file_master",
    "vendor_registration_certificate",
    "vendor_registration_code",
    "vendor_debar",
    "Rejven",
    "vendorcom_filtered",
    "vendor_renewal_request",
    "vendor_assessment",
    "vendor_deemed_registration",
]


def section(title: str):
    print(f"\n{'═' * 64}")
    print(f"  {title}")
    print(f"{'═' * 64}")


# ─── TEST 1: Direct MySQL Connection ─────────────────────────────

def test_db_connection():
    section("TEST 1: MySQL Connection & Data Access")
    try:
        import mysql.connector
    except ImportError:
        print(f"[{FAIL}] mysql-connector-python not installed")
        return False

    try:
        conn = mysql.connector.connect(
            host=MYDB_HOST, port=MYDB_PORT,
            user=MYDB_USER, password=MYDB_PASSWORD,
            database=MYDB_DATABASE,
            charset="utf8mb4", use_unicode=True,
            connect_timeout=10,
        )
        print(f"[{PASS}] Connected to {MYDB_HOST}:{MYDB_PORT}/{MYDB_DATABASE}")
    except Exception as exc:
        print(f"[{FAIL}] Connection failed: {exc}")
        return False

    cursor = conn.cursor()

    cursor.execute("SHOW TABLES")
    all_tables = [row[0] for row in cursor.fetchall()]
    print(f"[{INFO}] Total tables in mydb: {len(all_tables)}")

    total_rows = 0
    empty_tables = []
    table_counts = {}

    for table in KEY_TABLES:
        if table not in all_tables:
            print(f"[{WARN}] Table '{table}' not found in database")
            continue
        try:
            cursor.execute(f"SELECT COUNT(*) FROM `{table}`")
            count = cursor.fetchone()[0]
            table_counts[table] = count
            total_rows += count
            status = PASS if count > 0 else WARN
            if count == 0:
                empty_tables.append(table)
            print(f"[{status}] {table:45s} → {count:>8,} rows")
        except Exception as exc:
            print(f"[{FAIL}] {table:45s} → ERROR: {exc}")

    print(f"\n[{INFO}] Total rows across key tables: {total_rows:,}")
    if empty_tables:
        print(f"[{WARN}] Empty tables: {', '.join(empty_tables)}")

    print(f"\n── Sample Data Spot-Check ──")

    spot_checks = [
        ("vendor_master", "vm_id, name, regd_office_address, application_type", 3),
        ("certificate_report", "report_id, vendor_name, factory_name, grade, items", 3),
        ("factory_master", "id, name, factory_code, city, state", 3),
        ("item_master", "id, name, item_code, item_description", 3),
        ("vendor_debar", "id, vm_id, reason, debar_date", 3),
    ]

    for table, cols, limit in spot_checks:
        if table not in all_tables:
            continue
        try:
            cursor.execute(f"SELECT {cols} FROM `{table}` LIMIT {limit}")
            rows = cursor.fetchall()
            col_names = [d[0] for d in cursor.description]
            if rows:
                print(f"\n  ▶ {table} (sample {len(rows)} rows):")
                for row in rows:
                    preview = {col_names[i]: (str(v)[:60] if v else "NULL") for i, v in enumerate(row)}
                    print(f"    {preview}")
            else:
                print(f"\n  ▶ {table}: (empty)")
        except Exception as exc:
            print(f"\n  ▶ {table}: ERROR — {exc}")

    cursor.close()
    conn.close()
    print(f"\n[{PASS}] MySQL connection test complete")
    return True


# ─── TEST 2: FAISS Index Verification ────────────────────────────

def test_faiss_index():
    section("TEST 2: FAISS Index Verification")

    try:
        from app.core.vector_store import FaissStore
    except ImportError as exc:
        print(f"[{FAIL}] Cannot import FaissStore: {exc}")
        return False

    faiss_dir = os.getenv("FAISS_PERSIST_DIR", "./faiss_index")
    collection = os.getenv("FAISS_COLLECTION_NAME", "ofb_documents")

    index_file = os.path.join(faiss_dir, f"{collection}.faiss")
    meta_file = os.path.join(faiss_dir, f"{collection}.meta")

    if not os.path.exists(index_file):
        print(f"[{FAIL}] FAISS index file not found: {index_file}")
        print(f"        → You need to run Force Sync first!")
        return False
    print(f"[{PASS}] FAISS index file exists: {index_file}")

    if not os.path.exists(meta_file):
        print(f"[{FAIL}] FAISS metadata file not found: {meta_file}")
        return False
    print(f"[{PASS}] FAISS metadata file exists: {meta_file}")

    try:
        store = FaissStore(faiss_dir, collection)
        total_docs = store.count()
        print(f"[{PASS}] FAISS store loaded: {total_docs:,} live documents")
    except Exception as exc:
        print(f"[{FAIL}] Failed to load FAISS store: {exc}")
        return False

    if total_docs == 0:
        print(f"[{FAIL}] Index is empty — sync has not run or failed")
        return False

    mydb_docs = 0
    mydb_tables = set()
    chunk_docs = 0
    sample_ids = []

    for i, doc_id in enumerate(store._doc_ids):
        if doc_id and doc_id.startswith("mydb."):
            mydb_docs += 1
            parts = doc_id.split(".")
            if len(parts) >= 2:
                mydb_tables.add(parts[1])
            if ".chunk_" in doc_id:
                chunk_docs += 1
            if len(sample_ids) < 10:
                sample_ids.append(doc_id)

    print(f"\n[{INFO}] mydb documents in FAISS: {mydb_docs:,}")
    print(f"[{INFO}] mydb file chunks in FAISS: {chunk_docs:,}")
    print(f"[{INFO}] mydb tables represented: {len(mydb_tables)}")

    if mydb_docs == 0:
        print(f"[{FAIL}] No mydb documents found in FAISS — sync may not have included mydb")
        return False

    if mydb_tables:
        print(f"\n  Tables in index:")
        for t in sorted(mydb_tables):
            print(f"    • {t}")

    if sample_ids:
        print(f"\n  Sample doc IDs:")
        for sid in sample_ids:
            print(f"    {sid}")

    print(f"\n[{PASS}] FAISS index verification complete")
    return True


# ─── TEST 3: Semantic Search Queries ─────────────────────────────

SEARCH_QUERIES = [
    {
        "query": "Show me all vendors registered with their office address",
        "expected_tables": ["vendor_master", "vendorcom_filtered"],
        "note": "Tests vendor data — vendor_master has 2 rows",
    },
    {
        "query": "Which vendors have been debarred and what was the reason?",
        "expected_tables": ["vendor_debar"],
        "note": "Tests debarment data — vendor_debar has 1 row",
    },
    {
        "query": "Find ammunition items and their specifications",
        "expected_tables": ["item_master"],
        "note": "Tests item search — item_master has 5 rows with ammo data",
    },
    {
        "query": "Ordnance factory Kanpur Medak Khamaria address and code",
        "expected_tables": ["factory_master"],
        "note": "Tests factory data — factory_master has 3 rows",
    },
    {
        "query": "Defence equipment vendor certificate report with grade",
        "expected_tables": ["certificate_report", "vendor_master"],
        "note": "Tests certificate data — certificate_report has 2 rows",
    },
    {
        "query": "Defence Systems private limited vendor application and industry",
        "expected_tables": ["vendor_master"],
        "note": "Tests specific vendor lookup by name",
    },
    {
        "query": "Hand grenade and detonator product specifications",
        "expected_tables": ["item_master"],
        "note": "Tests specific item search — matches HE-36 grenade",
    },
    {
        "query": "XYZ Ammunition Corp vendor information and registration",
        "expected_tables": ["vendor_master", "vendorcom_filtered"],
        "note": "Tests second vendor lookup by name",
    },
    {
        "query": "OFK factory code Kanpur Uttar Pradesh GT Road",
        "expected_tables": ["factory_master"],
        "note": "Tests factory location search — OFK in Kanpur",
    },
    {
        "query": "Uploaded files and documents in the system",
        "expected_tables": ["file_master"],
        "note": "Tests file_master vectorization — has 3 rows + 2 chunks",
    },
]


def test_search_queries():
    section("TEST 3: Semantic Search Queries (Direct)")

    try:
        from app.core.search_engine import DocumentSearchEngine
    except ImportError as exc:
        print(f"[{FAIL}] Cannot import DocumentSearchEngine: {exc}")
        return False

    faiss_dir = os.getenv("FAISS_PERSIST_DIR", "./faiss_index")
    collection = os.getenv("FAISS_COLLECTION_NAME", "ofb_documents")

    try:
        engine = DocumentSearchEngine(faiss_dir, collection)
    except Exception as exc:
        print(f"[{FAIL}] Failed to init search engine: {exc}")
        return False

    passed_queries = 0
    total_queries = len(SEARCH_QUERIES)

    for i, q in enumerate(SEARCH_QUERIES, 1):
        print(f"\n  ▶ Query {i}/{total_queries}: \"{q['query']}\"")
        print(f"    ({q['note']})")

        start = time.perf_counter()
        try:
            results = engine.search(q["query"], top_k=5, db_filter="mydb")
            elapsed = (time.perf_counter() - start) * 1000
        except Exception as exc:
            print(f"    [{FAIL}] Search failed: {exc}")
            continue

        if not results:
            print(f"    [{WARN}] No results (expected data from {q['expected_tables']})")
            continue

        found_tables = {r.source_table for r in results}
        expected_hit = any(t in found_tables for t in q["expected_tables"])
        status = PASS if expected_hit else WARN

        if expected_hit:
            passed_queries += 1

        print(f"    [{status}] {len(results)} results in {elapsed:.0f}ms | tables: {', '.join(sorted(found_tables))}")

        for r in results[:3]:
            score_bar = "█" * int(r.score * 20)
            snippet = r.snippet[:120].replace("\n", " | ")
            print(f"      #{r.rank} sim={r.score:.4f} {score_bar}")
            print(f"         [{r.source_table}] {snippet}")

        if not expected_hit:
            print(f"    [{WARN}] Expected one of {q['expected_tables']} but got {sorted(found_tables)}")

    pass_rate = passed_queries / total_queries if total_queries > 0 else 0
    overall = pass_rate >= 0.7
    print(f"\n  Search accuracy: {passed_queries}/{total_queries} ({pass_rate:.0%})")
    print(f"[{PASS if overall else FAIL}] Search query tests {'passed' if overall else 'failed'} (threshold: 70%)")
    return overall


# ─── TEST 4: RAG Pipeline Queries ────────────────────────────────

RAG_QUERIES = [
    "List all ordnance factories in the system with their codes and locations.",
    "What are the reasons vendors get debarred? Give specific examples from the data.",
    "What ammunition items exist in the system? List them with their codes and descriptions.",
    "Tell me about ABC Defence Systems — their address, business type, and registration details.",
    "What is the certificate report for XYZ Ammunition Corp? What grade do they have?",
]


def test_rag_queries():
    section("TEST 4: RAG Pipeline (LLM + Retrieval)")

    try:
        from app.core.search_engine import DocumentSearchEngine
        from app.core.lc_vector_store import FaissRetrieverAdapter
        from app.services.rag_graph import RAGPipeline
    except ImportError as exc:
        print(f"[{FAIL}] Cannot import RAG components: {exc}")
        return False

    faiss_dir = os.getenv("FAISS_PERSIST_DIR", "./faiss_index")
    collection = os.getenv("FAISS_COLLECTION_NAME", "ofb_documents")

    try:
        engine = DocumentSearchEngine(faiss_dir, collection)
        retriever = FaissRetrieverAdapter(engine.store, top_k=5)
        pipeline = RAGPipeline(retriever, default_top_k=5, rerank_enabled=True)
    except Exception as exc:
        print(f"[{FAIL}] Failed to init RAG pipeline: {exc}")
        return False

    all_pass = True
    for i, query in enumerate(RAG_QUERIES, 1):
        print(f"\n  ▶ RAG Query {i}/{len(RAG_QUERIES)}:")
        print(f"    \"{query}\"")

        try:
            result = pipeline.ask(query=query, top_k=5, db_filter="mydb", temperature=0.1)
        except Exception as exc:
            print(f"    [{FAIL}] RAG failed: {exc}")
            all_pass = False
            continue

        if not result.answer or result.answer.strip() == "":
            print(f"    [{FAIL}] Empty answer from LLM")
            all_pass = False
            continue

        answer_preview = result.answer[:300].replace("\n", " | ")
        source_tables = {s.source_table for s in result.sources} if result.sources else set()

        print(f"    [{PASS}] Answer ({len(result.answer)} chars, {result.elapsed_ms:.0f}ms, model: {result.model_used})")
        print(f"    Sources: {', '.join(sorted(source_tables)) if source_tables else 'none'}")
        print(f"    Answer: {answer_preview}...")

    print(f"\n[{PASS if all_pass else WARN}] RAG pipeline tests complete")
    return all_pass


# ─── TEST 5: API-based tests (server must be running) ────────────

def test_api():
    section("TEST 5: HTTP API Tests (server must be running)")

    try:
        import requests
    except ImportError:
        print(f"[{FAIL}] 'requests' package not installed")
        return False

    headers = {"X-API-Key": API_KEY, "Content-Type": "application/json"}

    print(f"  API: {API_BASE}")
    print(f"  Key: {API_KEY[:10]}...\n")

    # Health check
    try:
        r = requests.get(f"{API_BASE}/api/v1/health", timeout=5)
        print(f"[{PASS if r.status_code == 200 else FAIL}] GET /health → {r.status_code}")
    except Exception as exc:
        print(f"[{FAIL}] Server unreachable: {exc}")
        return False

    # Sync status
    try:
        r = requests.get(f"{API_BASE}/api/v1/sync/status", headers=headers, timeout=10)
        if r.status_code == 200:
            data = r.json()
            print(f"[{PASS}] GET /sync/status → scheduler_running={data.get('scheduler_running')}")
            tables = data.get("tables", [])
            mydb_tables = [t for t in tables if t.get("table_key", "").startswith("mydb.")]
            print(f"[{INFO}] mydb tables in sync state: {len(mydb_tables)}")
            for t in mydb_tables[:5]:
                print(f"       {t['table_key']}: {t['status']} ({t['records_synced']} records)")
        else:
            print(f"[{FAIL}] GET /sync/status → {r.status_code}")
    except Exception as exc:
        print(f"[{FAIL}] /sync/status failed: {exc}")

    # Index status
    try:
        r = requests.get(f"{API_BASE}/api/v1/status", headers=headers, timeout=10)
        if r.status_code == 200:
            data = r.json()
            print(f"[{PASS}] GET /status → index_count={data.get('index_count', '?')}")
        else:
            print(f"[{WARN}] GET /status → {r.status_code}")
    except Exception as exc:
        print(f"[{WARN}] /status failed: {exc}")

    # Search via API
    search_tests = [
        {"query": "vendor registration certificates", "top_k": 5, "db_filter": "mydb"},
        {"query": "debarred vendors and rejection reasons", "top_k": 5, "db_filter": "mydb"},
        {"query": "factory master list with locations", "top_k": 5, "db_filter": "mydb"},
    ]

    print(f"\n── API Search Tests ──")
    for body in search_tests:
        try:
            r = requests.post(f"{API_BASE}/api/v1/search", headers=headers, json=body, timeout=30)
            if r.status_code == 200:
                data = r.json()
                results = data.get("results", [])
                tables = {r_["source_table"] for r_ in results}
                print(f"[{PASS}] \"{body['query']}\" → {len(results)} results from {tables}")
            else:
                print(f"[{FAIL}] \"{body['query']}\" → HTTP {r.status_code}: {r.text[:200]}")
        except Exception as exc:
            print(f"[{FAIL}] \"{body['query']}\" → {exc}")

    # RAG via API
    rag_tests = [
        {"query": "What types of vendor registrations exist in the system?", "top_k": 5, "db_filter": "mydb"},
        {"query": "List factories and their locations.", "top_k": 5, "db_filter": "mydb"},
    ]

    print(f"\n── API RAG Tests ──")
    for body in rag_tests:
        try:
            r = requests.post(f"{API_BASE}/api/v1/rag/ask", headers=headers, json=body, timeout=120)
            if r.status_code == 200:
                data = r.json()
                answer = data.get("answer", "")
                sources = data.get("sources", [])
                print(f"[{PASS}] \"{body['query']}\"")
                print(f"       Answer ({len(answer)} chars), {len(sources)} sources, {data.get('elapsed_ms', '?')}ms")
                print(f"       Preview: {answer[:200].replace(chr(10), ' | ')}...")
            else:
                print(f"[{FAIL}] \"{body['query']}\" → HTTP {r.status_code}: {r.text[:200]}")
        except Exception as exc:
            print(f"[{FAIL}] \"{body['query']}\" → {exc}")

    print(f"\n[{PASS}] API tests complete")
    return True


# ─── Main ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="End-to-end test for mydb vectorization pipeline")
    parser.add_argument("--db-only", action="store_true", help="Only test DB connection")
    parser.add_argument("--faiss-only", action="store_true", help="Only test FAISS index")
    parser.add_argument("--search-only", action="store_true", help="Only test search queries")
    parser.add_argument("--rag-only", action="store_true", help="Only test RAG pipeline")
    parser.add_argument("--api", action="store_true", help="Test via HTTP API (server must be running)")
    args = parser.parse_args()

    print("╔════════════════════════════════════════════════════════════╗")
    print("║     mydb End-to-End Verification Test Suite               ║")
    print("╚════════════════════════════════════════════════════════════╝")
    print(f"  DB: {MYDB_HOST}:{MYDB_PORT}/{MYDB_DATABASE}")
    print(f"  FAISS: {os.getenv('FAISS_PERSIST_DIR', './faiss_index')}")

    run_all = not any([args.db_only, args.faiss_only, args.search_only, args.rag_only, args.api])

    results = {}
    start = time.perf_counter()

    if run_all or args.db_only:
        results["db_connection"] = test_db_connection()

    if run_all or args.faiss_only:
        results["faiss_index"] = test_faiss_index()

    if run_all or args.search_only:
        results["search"] = test_search_queries()

    if run_all or args.rag_only:
        results["rag"] = test_rag_queries()

    if args.api:
        results["api"] = test_api()

    elapsed = time.perf_counter() - start

    section("SUMMARY")
    for test_name, passed in results.items():
        status = PASS if passed else FAIL
        print(f"  [{status}] {test_name}")

    total = len(results)
    passed = sum(1 for v in results.values() if v)
    print(f"\n  {passed}/{total} tests passed in {elapsed:.1f}s")

    if passed < total:
        print(f"\n  Troubleshooting:")
        if not results.get("db_connection", True):
            print(f"    • Check MySQL is running on {MYDB_HOST}:{MYDB_PORT}")
            print(f"    • Verify credentials in .env (OFBDB_USER, OFBDB_PASSWORD)")
        if not results.get("faiss_index", True):
            print(f"    • Run Force Sync from the UI to build the FAISS index")
            print(f"    • Or: python cli.py index --db mydb")
        if not results.get("search", True):
            print(f"    • Ensure Ollama is running with nomic-embed-text model")
            print(f"    • Check OLLAMA_BASE_URL in .env")
        if not results.get("rag", True):
            print(f"    • Ensure Ollama is running with the LLM model (llama3.1:8b)")
            print(f"    • Check OLLAMA_LLM_MODEL in .env")

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
