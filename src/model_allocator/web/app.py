"""FastAPI web server for Model Allocator frontend.

Serves static UI + JSON API for allocation model management.
Follows DPMtF-WebUI frontend governance: dark theme, panel groups,
expand/collapse, i18n, vanilla JS, no innerHTML.
"""
from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from ..config_loader import load_config
from ..config_writer import (
    ConfigWriteError,
    delete_alias,
    load_raw,
    set_alias,
)
from ..resolver import Resolver
from ..validator import Validator

# ── Paths ──────────────────────────────────────────────────
WEB_DIR = Path(__file__).resolve().parent
STATIC_DIR = WEB_DIR / "static"
TEMPLATES_DIR = WEB_DIR / "templates"
DB_PATH = Path(os.environ.get(
    "ALLOCATOR_DB_PATH",
    str(Path(__file__).resolve().parent.parent.parent.parent / "allocator.db"),
))
CONFIG_DIR = Path(os.environ.get(
    "ALLOCATOR_CONFIG_DIR",
    str(Path(__file__).resolve().parent.parent.parent.parent),
))

# ── Supported locales ──────────────────────────────────────
SUPPORTED_LOCALES = [
    {"locale": "en-US", "display_name": "English"},
    {"locale": "da-DK", "display_name": "Dansk"},
    {"locale": "de-DE", "display_name": "Deutsch"},
    {"locale": "el-GR", "display_name": "Ελληνικά"},
    {"locale": "sv-SE", "display_name": "Svenska"},
]
DEFAULT_LOCALE = "en-US"

# ── App ────────────────────────────────────────────────────
app = FastAPI(title="Model Allocator", docs_url="/api/docs")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ── DB helpers ─────────────────────────────────────────────
def _db() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def _init_db() -> None:
    """Create i18n + panel structure tables if they don't exist."""
    conn = _db()
    try:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS ui_labels (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                label_id TEXT UNIQUE NOT NULL,
                label_key TEXT UNIQUE NOT NULL,
                label_domain TEXT NOT NULL,
                default_text TEXT NOT NULL,
                description TEXT,
                is_active INTEGER DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS ui_label_translations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                label_id TEXT NOT NULL,
                locale TEXT NOT NULL,
                translated_text TEXT NOT NULL,
                is_active INTEGER DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(label_id, locale)
            );

            CREATE TABLE IF NOT EXISTS user_panel_groups (
                user_id TEXT NOT NULL,
                group_name TEXT NOT NULL,
                state TEXT NOT NULL DEFAULT 'expanded',
                is_visible INTEGER DEFAULT 1,
                updated_at TEXT DEFAULT (datetime('now')),
                PRIMARY KEY (user_id, group_name)
            );

            CREATE TABLE IF NOT EXISTS panel_subgroups (
                subgroup_key TEXT PRIMARY KEY,
                group_name TEXT NOT NULL,
                title_da TEXT NOT NULL,
                title_en TEXT NOT NULL,
                sort_order INTEGER DEFAULT 0,
                is_visible INTEGER DEFAULT 1
            );

            CREATE TABLE IF NOT EXISTS panel_subgroup_mappings (
                slot_key TEXT NOT NULL,
                subgroup_key TEXT NOT NULL,
                PRIMARY KEY (slot_key, subgroup_key)
            );

            CREATE TABLE IF NOT EXISTS user_panel_subgroup_states (
                user_id TEXT NOT NULL,
                subgroup_key TEXT NOT NULL,
                state TEXT NOT NULL DEFAULT 'expanded',
                PRIMARY KEY (user_id, subgroup_key)
            );
        """)
        _seed_labels(conn)
        _seed_panel_subgroups(conn)
        conn.commit()
    finally:
        conn.close()


def _seed_labels(conn: sqlite3.Connection) -> None:
    """Seed i18n labels (idempotent)."""
    labels = [
        ("lbl_page_title", "lbl_page_title", "main", "Model Allocator"),
        ("lbl_heading_main", "lbl_heading_main", "main", "Allocation Models"),
        ("lbl_tab_models", "lbl_tab_models", "main", "Allocation Models"),
        ("lbl_tab_profiles", "lbl_tab_profiles", "main", "Runtime Profiles"),
        ("lbl_btn_new_model", "lbl_btn_new_model", "main", "New Model"),
        ("lbl_btn_validate", "lbl_btn_validate", "main", "Validate"),
        ("lbl_btn_start", "lbl_btn_start", "main", "Start"),
        ("lbl_btn_stop", "lbl_btn_stop", "main", "Stop"),
        ("lbl_btn_delete", "lbl_btn_delete", "main", "Delete"),
        ("lbl_btn_save", "lbl_btn_save", "main", "Save"),
        ("lbl_btn_cancel", "lbl_btn_cancel", "main", "Cancel"),
        ("lbl_btn_edit", "lbl_btn_edit", "main", "Edit"),
        ("lbl_col_alias", "lbl_col_alias", "main", "Alias"),
        ("lbl_col_backend", "lbl_col_backend", "main", "Backend"),
        ("lbl_col_model", "lbl_col_model", "main", "Model"),
        ("lbl_col_context", "lbl_col_context", "main", "Context"),
        ("lbl_col_status", "lbl_col_status", "main", "Status"),
        ("lbl_col_actions", "lbl_col_actions", "main", "Actions"),
        ("lbl_col_profile", "lbl_col_profile", "main", "Runtime Profile"),
        ("lbl_col_lifecycle", "lbl_col_lifecycle", "main", "Lifecycle"),
        ("lbl_col_clients", "lbl_col_clients", "main", "Clients"),
        ("lbl_status_ok", "lbl_status_ok", "main", "OK"),
        ("lbl_status_error", "lbl_status_error", "main", "Error"),
        ("lbl_status_running", "lbl_status_running", "main", "Running"),
        ("lbl_status_stopped", "lbl_status_stopped", "main", "Stopped"),
        ("lbl_status_validating", "lbl_status_validating", "main", "Validating..."),
        ("lbl_lang_label", "lbl_lang_label", "main", "Language"),
        ("lbl_pg_setup", "lbl_pg_setup", "main", "Setup"),
        ("lbl_pg_daily", "lbl_pg_daily", "main", "Daily"),
        ("lbl_pg_journals", "lbl_pg_journals", "main", "Journals"),
        ("lbl_pg_reports", "lbl_pg_reports", "main", "Reports"),
        ("lbl_pg_periodic", "lbl_pg_periodic", "main", "Periodic"),
        ("lbl_sg_models", "lbl_sg_models", "main", "Allocation Models"),
        ("lbl_sg_profiles", "lbl_sg_profiles", "main", "Runtime Profiles"),
        ("lbl_sg_validation", "lbl_sg_validation", "main", "Validation"),
        ("lbl_sg_system", "lbl_sg_system", "main", "System"),
        ("lbl_field_alias", "lbl_field_alias", "main", "Alias name"),
        ("lbl_field_real_model", "lbl_field_real_model", "main", "Real model"),
        ("lbl_field_context", "lbl_field_context", "main", "Context window"),
        ("lbl_field_runtime_profile", "lbl_field_runtime_profile", "main", "Runtime profile"),
        ("lbl_field_lifecycle", "lbl_field_lifecycle", "main", "Lifecycle policy"),
        ("lbl_field_clients", "lbl_field_clients", "main", "Enabled clients"),
        ("lbl_lifecycle_persistent", "lbl_lifecycle_persistent", "main", "Persistent"),
        ("lbl_lifecycle_stop_after_step", "lbl_lifecycle_stop_after_step", "main", "Stop after step"),
        ("lbl_lifecycle_cloud_noop", "lbl_lifecycle_cloud_noop", "main", "Cloud noop"),
        ("lbl_empty_models", "lbl_empty_models", "main", "No allocation models configured"),
        ("lbl_empty_profiles", "lbl_empty_profiles", "main", "No runtime profiles configured"),
        ("lbl_confirm_delete", "lbl_confirm_delete", "main", "Are you sure you want to delete this model?"),
        ("lbl_error_load", "lbl_error_load", "main", "Failed to load data"),
        ("lbl_error_save", "lbl_error_save", "main", "Failed to save"),
        ("lbl_error_validate", "lbl_error_validate", "main", "Validation failed"),
        ("lbl_doctor_title", "lbl_doctor_title", "main", "Doctor Diagnostics"),
        ("lbl_doctor_run", "lbl_doctor_run", "main", "Run Doctor"),
        ("lbl_config_overview", "lbl_config_overview", "main", "Config Overview"),
    ]

    da_translations = {
        "lbl_page_title": "Model Allocator",
        "lbl_heading_main": "Allokeringsmodeller",
        "lbl_tab_models": "Allokeringsmodeller",
        "lbl_tab_profiles": "Runtime Profiler",
        "lbl_btn_new_model": "Ny Model",
        "lbl_btn_validate": "Valider",
        "lbl_btn_start": "Start",
        "lbl_btn_stop": "Stoppet",
        "lbl_btn_delete": "Slet",
        "lbl_btn_save": "Gem",
        "lbl_btn_cancel": "Annuller",
        "lbl_btn_edit": "Rediger",
        "lbl_col_alias": "Alias",
        "lbl_col_backend": "Backend",
        "lbl_col_model": "Model",
        "lbl_col_context": "Kontekst",
        "lbl_col_status": "Status",
        "lbl_col_actions": "Handlinger",
        "lbl_col_profile": "Runtime Profil",
        "lbl_col_lifecycle": "Livscyklus",
        "lbl_col_clients": "Klienter",
        "lbl_status_ok": "OK",
        "lbl_status_error": "Fejl",
        "lbl_status_running": "Kører",
        "lbl_status_stopped": "Stoppet",
        "lbl_status_validating": "Validerer...",
        "lbl_lang_label": "Sprog",
        "lbl_pg_setup": "Opsætning",
        "lbl_pg_daily": "Dagligt",
        "lbl_pg_journals": "Journaler",
        "lbl_pg_reports": "Rapporter",
        "lbl_pg_periodic": "Periodisk",
        "lbl_sg_models": "Allokeringsmodeller",
        "lbl_sg_profiles": "Runtime Profiler",
        "lbl_sg_validation": "Validering",
        "lbl_sg_system": "System",
        "lbl_field_alias": "Alias navn",
        "lbl_field_real_model": "Reel model",
        "lbl_field_context": "Kontekst vindue",
        "lbl_field_runtime_profile": "Runtime profil",
        "lbl_field_lifecycle": "Livscyklus politik",
        "lbl_field_clients": "Aktiverede klienter",
        "lbl_lifecycle_persistent": "Persistent",
        "lbl_lifecycle_stop_after_step": "Stop efter step",
        "lbl_lifecycle_cloud_noop": "Cloud noop",
        "lbl_empty_models": "Ingen allokeringsmodeller konfigureret",
        "lbl_empty_profiles": "Ingen runtime profiler konfigureret",
        "lbl_confirm_delete": "Er du sikker på at du vil slette denne model?",
        "lbl_error_load": "Kunne ikke hente data",
        "lbl_error_save": "Kunne ikke gemme",
        "lbl_error_validate": "Validering fejlede",
        "lbl_doctor_title": "Doctor Diagnostik",
        "lbl_doctor_run": "Kør Doctor",
        "lbl_config_overview": "Konfiguration Oversigt",
    }

    for label_id, label_key, domain, default_text in labels:
        existing = conn.execute(
            "SELECT id FROM ui_labels WHERE label_key = ?", (label_key,)
        ).fetchone()
        if not existing:
            conn.execute(
                "INSERT INTO ui_labels (label_id, label_key, label_domain, default_text) VALUES (?, ?, ?, ?)",
                (label_id, label_key, domain, default_text),
            )
            # Seed en-US translation
            conn.execute(
                "INSERT INTO ui_label_translations (label_id, locale, translated_text) VALUES (?, ?, ?)",
                (label_id, "en-US", default_text),
            )
            # Seed da-DK translation if available
            if label_key in da_translations:
                conn.execute(
                    "INSERT OR IGNORE INTO ui_label_translations (label_id, locale, translated_text) VALUES (?, ?, ?)",
                    (label_id, "da-DK", da_translations[label_key]),
                )
                # Seed other locales with en-US as fallback
                for locale in ("de-DE", "el-GR", "sv-SE"):
                    conn.execute(
                        "INSERT OR IGNORE INTO ui_label_translations (label_id, locale, translated_text) VALUES (?, ?, ?)",
                        (label_id, locale, default_text),
                    )


def _seed_panel_subgroups(conn: sqlite3.Connection) -> None:
    """Seed panel subgroups for allocator UI (idempotent)."""
    subgroups = [
        ("sg_setup_models", "setup", "Allokeringsmodeller", "Allocation Models", 1, 1),
        ("sg_setup_profiles", "setup", "Runtime Profiler", "Runtime Profiles", 2, 1),
        ("sg_setup_validation", "setup", "Validering", "Validation", 3, 1),
        ("sg_setup_system", "setup", "System", "System", 4, 1),
    ]
    for sg_key, group, title_da, title_en, sort_order, is_visible in subgroups:
        existing = conn.execute(
            "SELECT subgroup_key FROM panel_subgroups WHERE subgroup_key = ?", (sg_key,)
        ).fetchone()
        if not existing:
            conn.execute(
                "INSERT INTO panel_subgroups (subgroup_key, group_name, title_da, title_en, sort_order, is_visible) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (sg_key, group, title_da, title_en, sort_order, is_visible),
            )

    mappings = [
        ("lbl_sg_models", "sg_setup_models"),
        ("lbl_sg_profiles", "sg_setup_profiles"),
        ("lbl_sg_validation", "sg_setup_validation"),
        ("lbl_sg_system", "sg_setup_system"),
    ]
    for slot_key, sg_key in mappings:
        existing = conn.execute(
            "SELECT slot_key FROM panel_subgroup_mappings WHERE slot_key = ? AND subgroup_key = ?",
            (slot_key, sg_key),
        ).fetchone()
        if not existing:
            conn.execute(
                "INSERT INTO panel_subgroup_mappings (slot_key, subgroup_key) VALUES (?, ?)",
                (slot_key, sg_key),
            )


# ── API: i18n ──────────────────────────────────────────────
@app.get("/api/available-languages")
async def available_languages():
    return {"languages": SUPPORTED_LOCALES}


@app.get("/api/user-language")
async def get_user_language():
    conn = _db()
    try:
        row = conn.execute(
            "SELECT locale FROM user_language WHERE user_id = 'default'"
        ).fetchone()
        locale = row["locale"] if row else DEFAULT_LOCALE
        return {"locale": locale}
    except sqlite3.OperationalError:
        # Table doesn't exist yet — return default
        conn.execute(
            "CREATE TABLE IF NOT EXISTS user_language (user_id TEXT PRIMARY KEY, locale TEXT, updated_at TEXT)"
        )
        conn.commit()
        return {"locale": DEFAULT_LOCALE}
    finally:
        conn.close()


@app.post("/api/user-language")
async def set_user_language(request: Request):
    body = await request.json()
    locale = body.get("locale", DEFAULT_LOCALE)
    conn = _db()
    try:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS user_language (user_id TEXT PRIMARY KEY, locale TEXT, updated_at TEXT)"
        )
        conn.execute(
            "INSERT OR REPLACE INTO user_language (user_id, locale, updated_at) VALUES (?, ?, datetime('now'))",
            ("default", locale),
        )
        conn.commit()
        return {"status": "ok", "locale": locale}
    finally:
        conn.close()


@app.get("/api/ui-labels/{domain}")
async def get_labels(domain: str, locale: str = "en-US"):
    conn = _db()
    try:
        rows = conn.execute(
            """SELECT l.label_key, COALESCE(t.translated_text, l.default_text) as text
               FROM ui_labels l
               LEFT JOIN ui_label_translations t
                 ON l.label_id = t.label_id AND t.locale = ?
               WHERE l.label_domain = ? AND l.is_active = 1
               ORDER BY l.label_key""",
            (locale, domain),
        ).fetchall()
        labels = {row["label_key"]: row["text"] for row in rows}
        return {"labels": labels, "locale": locale}
    finally:
        conn.close()


# ── API: Panel structure ───────────────────────────────────
@app.get("/api/panel-structure")
async def get_panel_structure():
    conn = _db()
    try:
        groups = ["daily", "journals", "reports", "periodic", "setup"]
        result = {}
        for g in groups:
            subs = conn.execute(
                """SELECT s.subgroup_key, s.title_en, s.title_da, s.sort_order, s.is_visible,
                          COALESCE(u.state, 'expanded') as state
                   FROM panel_subgroups s
                   LEFT JOIN user_panel_subgroup_states u
                     ON s.subgroup_key = u.subgroup_key AND u.user_id = 'default'
                   WHERE s.group_name = ? AND s.is_visible = 1
                   ORDER BY s.sort_order""",
                (g,),
            ).fetchall()
            mappings = {}
            for sub in subs:
                slots = conn.execute(
                    "SELECT slot_key FROM panel_subgroup_mappings WHERE subgroup_key = ?",
                    (sub["subgroup_key"],),
                ).fetchall()
                mappings[sub["subgroup_key"]] = [s["slot_key"] for s in slots]

            grp_state = conn.execute(
                "SELECT state FROM user_panel_groups WHERE user_id = 'default' AND group_name = ?",
                (g,),
            ).fetchone()
            result[g] = {
                "is_visible": True,
                "state": grp_state["state"] if grp_state else "expanded",
                "subgroups": [
                    {
                        "key": sub["subgroup_key"],
                        "title": sub["title_en"],
                        "title_da": sub["title_da"],
                        "sort_order": sub["sort_order"],
                        "state": sub["state"],
                        "slot_keys": mappings.get(sub["subgroup_key"], []),
                    }
                    for sub in subs
                ],
            }
        return result
    finally:
        conn.close()


@app.get("/api/user-panel-groups")
async def get_user_panel_groups():
    conn = _db()
    try:
        rows = conn.execute(
            "SELECT group_name, state, is_visible FROM user_panel_groups WHERE user_id = 'default'"
        ).fetchall()
        groups = {}
        for r in rows:
            groups[r["group_name"]] = {
                "state": r["state"],
                "is_visible": r["is_visible"],
            }
        return groups
    finally:
        conn.close()


@app.post("/api/user-panel-groups")
async def save_panel_group_state(request: Request):
    body = await request.json()
    group_name = body.get("group_name", "")
    state = body.get("state", "expanded")
    if not group_name:
        raise HTTPException(status_code=400, detail="group_name required")
    conn = _db()
    try:
        conn.execute(
            "INSERT OR REPLACE INTO user_panel_groups (user_id, group_name, state, updated_at) "
            "VALUES (?, ?, ?, datetime('now'))",
            ("default", group_name, state),
        )
        conn.commit()
        return {"status": "ok"}
    finally:
        conn.close()


@app.post("/api/panel-structure/subgroup-state")
async def save_subgroup_state(request: Request):
    body = await request.json()
    subgroup_key = body.get("subgroup_key", "")
    state = body.get("state", "expanded")
    if not subgroup_key:
        raise HTTPException(status_code=400, detail="subgroup_key required")
    conn = _db()
    try:
        conn.execute(
            "INSERT OR REPLACE INTO user_panel_subgroup_states (user_id, subgroup_key, state) "
            "VALUES (?, ?, ?)",
            ("default", subgroup_key, state),
        )
        conn.commit()
        return {"status": "ok"}
    finally:
        conn.close()


# ── API: Allocation models ─────────────────────────────────
@app.get("/api/models")
async def list_models():
    """List all allocation models with validation status."""
    raw = load_raw(CONFIG_DIR)
    aliases = raw["aliases"]
    resolver = Resolver(config_dir=CONFIG_DIR)
    validator = Validator(resolver=resolver)

    models = []
    for name, definition in sorted(aliases.items()):
        clients = definition.get("clients", {})
        enabled_clients = [c for c, enabled in clients.items() if enabled]
        # Quick validate against first enabled client
        status = "unknown"
        if enabled_clients:
            try:
                result = validator.validate(name, enabled_clients[0])
                status = result.get("validation_status", "unknown").lower()
            except Exception:
                status = "error"

        models.append({
            "alias": name,
            "runtime_profile": definition.get("runtime_profile", ""),
            "real_model": definition.get("real_model", ""),
            "context": definition.get("context", 0),
            "lifecycle_policy": definition.get("lifecycle_policy", ""),
            "clients": enabled_clients,
            "validation_status": status,
        })

    return {"models": models}


@app.post("/api/models")
async def create_or_update_model(request: Request):
    """Create or update an allocation model."""
    body = await request.json()
    name = body.get("alias", "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="alias is required")

    definition = {
        "runtime_profile": body.get("runtime_profile", ""),
        "real_model": body.get("real_model", ""),
        "context": int(body.get("context", 131072)),
        "lifecycle_policy": body.get("lifecycle_policy", "stop_after_step"),
        "clients": {
            "opencode": bool(body.get("client_opencode", False)),
            "claude-code": bool(body.get("client_claude_code", False)),
        },
    }

    try:
        set_alias(CONFIG_DIR, name, definition)
        return {"status": "ok", "alias": name}
    except ConfigWriteError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.delete("/api/models/{alias}")
async def delete_model(alias: str):
    """Delete an allocation model."""
    try:
        delete_alias(CONFIG_DIR, alias)
        return {"status": "ok", "alias": alias}
    except ConfigWriteError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/models/{alias}/validate")
async def validate_model(alias: str, request: Request):
    """Validate an allocation model for a specific client."""
    body = await request.json() if request.headers.get("content-type") == "application/json" else {}
    client = body.get("client", "opencode") if isinstance(body, dict) else "opencode"

    resolver = Resolver(config_dir=CONFIG_DIR)
    validator = Validator(resolver=resolver)
    try:
        result = validator.validate(alias, client)
        return result
    except Exception as e:
        return {"validation_status": "ERROR", "errors": [str(e)]}


@app.post("/api/models/{alias}/start")
async def start_model(alias: str):
    """Start the model runtime."""
    from ..cli import _run_allocator_start
    try:
        result = _run_allocator_start(alias, config_dir=CONFIG_DIR, timeout=180)
        if result.returncode == 0:
            return {"status": "ok", "alias": alias}
        else:
            return {"status": "error", "stderr": result.stderr[:500] if result.stderr else ""}
    except Exception as e:
        return {"status": "error", "error": str(e)}


@app.post("/api/models/{alias}/stop")
async def stop_model(alias: str):
    """Stop the model runtime."""
    from ..cli import _run_allocator_stop
    try:
        result = _run_allocator_stop(alias, config_dir=CONFIG_DIR, timeout=45)
        if result.returncode == 0:
            return {"status": "ok", "alias": alias}
        else:
            return {"status": "error", "stderr": result.stderr[:500] if result.stderr else ""}
    except Exception as e:
        return {"status": "error", "error": str(e)}


@app.get("/api/models/{alias}/status")
async def model_status(alias: str):
    """Get runtime status for a model."""
    resolver = Resolver(config_dir=CONFIG_DIR)
    try:
        info = resolver.resolve_alias(alias)
        return {
            "alias": alias,
            "backend": info.get("backend", "unknown"),
            "model": info.get("real_model", "unknown"),
            "runtime_profile": info.get("runtime_profile", "unknown"),
        }
    except Exception as e:
        return {"alias": alias, "error": str(e)}


# ── API: Runtime profiles ──────────────────────────────────
@app.get("/api/profiles")
async def list_profiles():
    """List all runtime profiles."""
    raw = load_raw(CONFIG_DIR)
    profiles = raw["profiles"]
    result = []
    for name, definition in sorted(profiles.items()):
        result.append({
            "name": name,
            "backend": definition.get("backend", ""),
            "gpu": definition.get("gpu", ""),
            "api_base_env": definition.get("api_base_env", ""),
            "default_api_base": definition.get("default_api_base", ""),
            "provider": definition.get("provider", ""),
        })
    return {"profiles": result}


# ── API: Config + Doctor ───────────────────────────────────
@app.get("/api/config")
async def get_config():
    """Full config dump."""
    raw = load_raw(CONFIG_DIR)
    return {
        "aliases": raw["aliases"],
        "profiles": raw["profiles"],
    }


@app.post("/api/doctor")
async def run_doctor():
    """Run doctor diagnostics."""
    from ..config_writer import load_raw
    from ..schema import lint_config
    try:
        raw = load_raw(CONFIG_DIR)
        report = lint_config(raw)
        return {"status": "ok", "result": report}
    except Exception as e:
        return {"status": "error", "error": str(e)}


# ── Health ─────────────────────────────────────────────────
@app.get("/api/health")
async def health():
    return {"status": "healthy", "app": "Model Allocator"}


# ── HTML ───────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = TEMPLATES_DIR / "index.html"
    if html_path.exists():
        return HTMLResponse(html_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>Model Allocator</h1><p>Template not found</p>")


# ── Startup ────────────────────────────────────────────────
@app.on_event("startup")
async def startup():
    _init_db()


def run():
    """Entry point for model-allocator-web command."""
    import uvicorn
    port = int(os.environ.get("ALLOCATOR_WEB_PORT", "9140"))
    host = os.environ.get("ALLOCATOR_WEB_HOST", "0.0.0.0")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run()
