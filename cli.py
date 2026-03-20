"""
main.py — OFB Intelligent Document Discovery System
──────────────────────────────────────────────────────

Commands:
  index                              Full re-index all known DBs
  index --db ofbdb                   Re-index one DB only
  search "<query>"                   Semantic search
  search "<query>" --db misofb       Search within one DB
  demo                               Run example queries
  sync ofbdb certificate_report      Incremental re-index one table
  discover --host H --user U --db D  Analyse a NEW DB with Llama 3 8B
  status                             Show index stats

Setup (all offline after initial pull):
  ollama pull nomic-embed-text
  ollama pull llama3:8b
  ollama serve                       # keep running in background

  pip install -r requirements.txt

Environment variables (or edit CONFIG below):
  OFBDB_HOST / OFBDB_USER / OFBDB_PASSWORD
  MISOFB_HOST / MISOFB_USER / MISOFB_PASSWORD
"""

import argparse
import logging
import os
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    datefmt="%H:%M:%S",
)

from app.core.schema_config import get_all_configs, OFBDB_CONFIGS, MISOFB_CONFIGS
from app.core.schema_intelligence import discover_and_configure, print_config_summary
from app.core.search_engine import DocumentSearchEngine, EXAMPLE_QUERIES
from app.core.vectorizer import VectorizationEngine, VectorizerConfig, DBConnectionConfig

# ─────────────────────────────────────────────────────────────────────
# ⚠️  DB credentials — edit here or set environment variables
# ─────────────────────────────────────────────────────────────────────

def _make_config() -> VectorizerConfig:
    return VectorizerConfig(
        db_connections={
            "ofbdb": DBConnectionConfig(
                host     = os.getenv("OFBDB_HOST",     "localhost"),
                port     = int(os.getenv("OFBDB_PORT", "3306")),
                user     = os.getenv("OFBDB_USER",     "root"),
                password = os.getenv("OFBDB_PASSWORD", ""),
                database = "ofbdb",
            ),
            "misofb": DBConnectionConfig(
                host     = os.getenv("MISOFB_HOST",     "localhost"),
                port     = int(os.getenv("MISOFB_PORT", "3306")),
                user     = os.getenv("MISOFB_USER",     "root"),
                password = os.getenv("MISOFB_PASSWORD", ""),
                database = "misofb",
            ),
        },
        faiss_persist_dir = "./faiss_index",
        collection_name   = "ofb_documents",
        embed_batch_size  = 16,
        fetch_chunk_size  = 500,
        num_embed_workers = 2,
        save_every_n      = 2000,
    )


# ─────────────────────────────────────────────────────────────────────
# Commands
# ─────────────────────────────────────────────────────────────────────

def cmd_index(db_filter: str = None):
    config = _make_config()
    all_cfgs = get_all_configs()

    if db_filter:
        all_cfgs = [c for c in all_cfgs if c.db == db_filter]
        print(f"Indexing {len(all_cfgs)} tables from [{db_filter}]…")
    else:
        print(f"Indexing {len(all_cfgs)} tables across all DBs…")

    # Add DB connection for any auto-discovered DBs that need it
    # (user must pass credentials via env vars  NEW_DB_HOST / NEW_DB_USER / etc.)
    for cfg in all_cfgs:
        if cfg.db not in config.db_connections:
            config.db_connections[cfg.db] = DBConnectionConfig(
                host     = os.getenv(f"{cfg.db.upper()}_HOST", "localhost"),
                port     = int(os.getenv(f"{cfg.db.upper()}_PORT", "3306")),
                user     = os.getenv(f"{cfg.db.upper()}_USER", "root"),
                password = os.getenv(f"{cfg.db.upper()}_PASSWORD", ""),
                database = cfg.db,
            )

    engine = VectorizationEngine(config)
    engine.start_background_indexing(all_cfgs)
    engine.wait_until_complete()
    print(f"\n✓ Done.  Live docs in index: {engine.collection_count()}")


def cmd_search(query: str, db_filter: str = None, top_k: int = 5):
    engine = DocumentSearchEngine(
        faiss_persist_dir = "./faiss_index",
        collection_name   = "ofb_documents",
    )
    print(engine.search_pretty(query, top_k=top_k, db_filter=db_filter))


def cmd_demo():
    engine = DocumentSearchEngine(
        faiss_persist_dir = "./faiss_index",
        collection_name   = "ofb_documents",
    )
    print("\n" + "═"*64)
    print("  OFB Intelligent Document Discovery — Demo")
    print("═"*64)
    for ex in EXAMPLE_QUERIES:
        results = engine.search(ex["query"], top_k=3)
        print(f"\n▶ {ex['query']}")
        print(f"  ({ex['note']})")
        for r in results:
            print(f"  #{r.rank} [{r.label}]  sim={r.score:.4f}  {r.source_db}.{r.source_table}")
            print(f"       {r.snippet[:100].replace(chr(10),' | ')}")


def cmd_sync(db_name: str, table_name: str):
    config = _make_config()
    engine = VectorizationEngine(config)
    count  = engine.sync_table(db_name, table_name)
    print(f"✓ Incremental sync: {count} docs in {db_name}.{table_name}")


def cmd_discover(
    host: str, port: int, user: str, password: str, database: str,
    force: bool = False,
):
    """
    Analyse a brand-new database with Llama 3 8B and save its config.
    After discovery, run  python main.py index  to vectorise it.
    """
    print(f"\nDiscovering schema for [{database}] at {host}:{port} …")
    print("(Calling Llama 3 8B via Ollama — may take 30–90 seconds)\n")

    configs = discover_and_configure(
        host=host, port=port, user=user, password=password,
        database=database, force_rediscover=force,
    )

    if not configs:
        print("✗ Discovery failed. Check Ollama logs (ollama serve).")
        sys.exit(1)

    print(f"✓ Discovered {len(configs)} tables for [{database}]:\n")
    print_config_summary(configs)
    print(f"Config saved to  configs/{database}.json")
    print(f"\nNext step:  python main.py index --db {database}")


def cmd_status():
    from app.core.schema_intelligence import list_known_databases
    all_cfgs = get_all_configs()
    dbs = {}
    for c in all_cfgs:
        dbs.setdefault(c.db, []).append(c.table)

    print("\n── Configured databases ─────────────────────────────────")
    for db, tables in dbs.items():
        source = "handcrafted" if db in ("ofbdb", "misofb") else "auto-discovered"
        print(f"  [{db}]  ({source})  {len(tables)} tables")
        for t in tables:
            print(f"    • {t}")

    print("\n── FAISS index ──────────────────────────────────────────")
    try:
        engine = DocumentSearchEngine("./faiss_index", "ofb_documents")
        print(f"  Live docs  : {engine.index_count()}")
    except Exception as e:
        print(f"  (index not found: {e})")

    print("\n── Auto-discovered DBs (configs/) ───────────────────────")
    extras = [d for d in list_known_databases() if d not in ("ofbdb", "misofb")]
    if extras:
        for d in extras:
            print(f"  {d}")
    else:
        print("  (none yet — use  python main.py discover  to add one)")


# ─────────────────────────────────────────────────────────────────────
# CLI parser
# ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="OFB Intelligent Document Discovery System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command")

    # index
    p_idx = sub.add_parser("index", help="Vectorise all (or one) DB")
    p_idx.add_argument("--db", default=None, help="Restrict to one DB name")

    # search
    p_srch = sub.add_parser("search", help="NL search the index")
    p_srch.add_argument("query",          help="Natural language query")
    p_srch.add_argument("--db",    default=None, help="Restrict to DB")
    p_srch.add_argument("--top-k", type=int, default=5)

    # demo
    sub.add_parser("demo", help="Run example queries")

    # sync
    p_sync = sub.add_parser("sync", help="Incremental re-index one table")
    p_sync.add_argument("db_name")
    p_sync.add_argument("table_name")

    # discover
    p_disc = sub.add_parser("discover", help="Auto-configure a new DB with Llama 3 8B")
    p_disc.add_argument("--host",     default="localhost")
    p_disc.add_argument("--port",     type=int, default=3306)
    p_disc.add_argument("--user",     required=True)
    p_disc.add_argument("--password", default="")
    p_disc.add_argument("--db",       dest="database", required=True)
    p_disc.add_argument("--force",    action="store_true",
                        help="Re-discover even if config already exists")

    # status
    sub.add_parser("status", help="Show config and index stats")

    args = parser.parse_args()

    if args.command == "index":
        cmd_index(db_filter=args.db)
    elif args.command == "search":
        cmd_search(args.query, db_filter=args.db, top_k=args.top_k)
    elif args.command == "demo":
        cmd_demo()
    elif args.command == "sync":
        cmd_sync(args.db_name, args.table_name)
    elif args.command == "discover":
        cmd_discover(
            host=args.host, port=args.port,
            user=args.user, password=args.password,
            database=args.database, force=args.force,
        )
    elif args.command == "status":
        cmd_status()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
