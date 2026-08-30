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
    {"locale": "es-ES", "display_name": "Español"},
    {"locale": "sv-SE", "display_name": "Svenska"},
]
DEFAULT_LOCALE = "en-US"

# ── App ────────────────────────────────────────────────────
from contextlib import asynccontextmanager


@asynccontextmanager
async def _lifespan(app: FastAPI):
    _init_db()
    yield


app = FastAPI(title="Model Allocator", docs_url="/api/docs", lifespan=_lifespan)
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


ES_TRANSLATIONS = {
    "lbl_page_title": "Model Allocator",
    "lbl_heading_main": "Modelos de asignación",
    "lbl_tab_models": "Modelos de asignación",
    "lbl_tab_profiles": "Backends",
    "lbl_btn_new_model": "Nuevo modelo",
    "lbl_btn_validate": "Validar",
    "lbl_btn_start": "Iniciar",
    "lbl_btn_stop": "Detener",
    "lbl_btn_delete": "Eliminar",
    "lbl_btn_save": "Guardar",
    "lbl_btn_cancel": "Cancelar",
    "lbl_btn_edit": "Editar",
    "lbl_col_alias": "Alias",
    "lbl_col_backend": "Backend",
    "lbl_col_model": "Modelo",
    "lbl_col_context": "Contexto",
    "lbl_col_status": "Estado",
    "lbl_col_actions": "Acciones",
    "lbl_col_profile": "Perfil de runtime",
    "lbl_col_lifecycle": "Ciclo de vida",
    "lbl_col_clients": "Harness",
    "lbl_col_api_base": "Base de API",
    "lbl_col_gpu": "GPU",
    "lbl_col_provider": "Proveedor",
    "lbl_status_ok": "OK",
    "lbl_status_error": "Error",
    "lbl_status_warning": "Advertencia",
    "lbl_status_unknown": "Desconocido",
    "lbl_status_running": "En ejecución",
    "lbl_status_stopped": "Detenido",
    "lbl_status_validating": "Validando...",
    "lbl_status_hint": "El estado refleja el primer harness permitido; Validar los comprueba todos.",
    "lbl_validated_clients": "Harnesses validados",
    "lbl_validation_result": "Resultado de validación",
    "lbl_validation_ok_hint": "La configuración es válida.",
    "lbl_validation_warn_hint": "Utilizable pero incompleta o sin verificar — corrija las advertencias para llegar a OK.",
    "lbl_validation_err_hint": "Inutilizable hasta corregir los errores.",
    "lbl_lang_label": "Idioma",
    "lbl_pg_setup": "Configuración",
    "lbl_pg_daily": "Diario",
    "lbl_pg_journals": "Diarios de trabajo",
    "lbl_pg_reports": "Informes",
    "lbl_pg_periodic": "Periódico",
    "lbl_sg_models": "Modelos de asignación",
    "lbl_sg_profiles": "Backends",
    "lbl_sg_validation": "Validación",
    "lbl_sg_system": "Sistema",
    "lbl_field_alias": "Nombre del alias",
    "lbl_field_real_model": "Modelo real",
    "lbl_field_context": "Ventana de contexto",
    "lbl_field_runtime_profile": "Perfil de runtime",
    "lbl_field_lifecycle": "Política de ciclo de vida",
    "lbl_field_clients": "Harnesses permitidos",
    "lbl_lifecycle_persistent": "Persistente",
    "lbl_lifecycle_stop_after_step": "Detener tras el paso",
    "lbl_lifecycle_cloud_noop": "Cloud noop",
    "lbl_empty_models": "No hay modelos de asignación configurados",
    "lbl_empty_profiles": "No hay backends configurados",
    "lbl_confirm_delete": "¿Seguro que desea eliminar este modelo?",
    "lbl_error_load": "No se pudieron cargar los datos",
    "lbl_error_save": "No se pudo guardar",
    "lbl_error_delete": "No se pudo eliminar",
    "lbl_error_validate": "Falló la validación",
    "lbl_doctor_title": "Diagnóstico Doctor",
    "lbl_doctor_run": "Ejecutar Doctor",
    "lbl_config_overview": "Resumen de configuración",
    "lbl_running_progress": "Ejecutando…",
    "lbl_status_loading": "Cargando…",
    "lbl_error_prefix": "Error",
    "lbl_show_config": "Mostrar configuración",
}

DE_TRANSLATIONS = {
    "lbl_page_title": "Model Allocator",
    "lbl_heading_main": "Allokationsmodelle",
    "lbl_tab_models": "Allokationsmodelle",
    "lbl_tab_profiles": "Backends",
    "lbl_btn_new_model": "Neues Modell",
    "lbl_btn_validate": "Validieren",
    "lbl_btn_start": "Starten",
    "lbl_btn_stop": "Stoppen",
    "lbl_btn_delete": "Löschen",
    "lbl_btn_save": "Speichern",
    "lbl_btn_cancel": "Abbrechen",
    "lbl_btn_edit": "Bearbeiten",
    "lbl_col_alias": "Alias",
    "lbl_col_backend": "Backend",
    "lbl_col_model": "Modell",
    "lbl_col_context": "Kontext",
    "lbl_col_status": "Status",
    "lbl_col_actions": "Aktionen",
    "lbl_col_profile": "Laufzeitprofil",
    "lbl_col_lifecycle": "Lebenszyklus",
    "lbl_col_clients": "Harness",
    "lbl_col_api_base": "API-Basis",
    "lbl_col_gpu": "GPU",
    "lbl_col_provider": "Anbieter",
    "lbl_status_ok": "OK",
    "lbl_status_error": "Fehler",
    "lbl_status_warning": "Warnung",
    "lbl_status_unknown": "Unbekannt",
    "lbl_status_running": "Läuft",
    "lbl_status_stopped": "Gestoppt",
    "lbl_status_validating": "Validierung läuft...",
    "lbl_status_hint": "Der Status spiegelt den ersten erlaubten Harness; Validieren prüft alle.",
    "lbl_validated_clients": "Validierte Harnesses",
    "lbl_validation_result": "Validierungsergebnis",
    "lbl_validation_ok_hint": "Die Konfiguration ist gültig.",
    "lbl_validation_warn_hint": "Nutzbar, aber unvollständig oder ungeprüft — beheben Sie die Warnungen, um OK zu erreichen.",
    "lbl_validation_err_hint": "Unbrauchbar, bis die Fehler behoben sind.",
    "lbl_lang_label": "Sprache",
    "lbl_pg_setup": "Einrichtung",
    "lbl_pg_daily": "Täglich",
    "lbl_pg_journals": "Journale",
    "lbl_pg_reports": "Berichte",
    "lbl_pg_periodic": "Periodisch",
    "lbl_sg_models": "Allokationsmodelle",
    "lbl_sg_profiles": "Backends",
    "lbl_sg_validation": "Validierung",
    "lbl_sg_system": "System",
    "lbl_field_alias": "Aliasname",
    "lbl_field_real_model": "Reales Modell",
    "lbl_field_context": "Kontextfenster",
    "lbl_field_runtime_profile": "Laufzeitprofil",
    "lbl_field_lifecycle": "Lebenszyklus-Richtlinie",
    "lbl_field_clients": "Erlaubte Harnesses",
    "lbl_lifecycle_persistent": "Persistent",
    "lbl_lifecycle_stop_after_step": "Nach dem Schritt stoppen",
    "lbl_lifecycle_cloud_noop": "Cloud noop",
    "lbl_empty_models": "Keine Allokationsmodelle konfiguriert",
    "lbl_empty_profiles": "Keine Backends konfiguriert",
    "lbl_confirm_delete": "Möchten Sie dieses Modell wirklich löschen?",
    "lbl_error_load": "Daten konnten nicht geladen werden",
    "lbl_error_save": "Speichern fehlgeschlagen",
    "lbl_error_delete": "Löschen fehlgeschlagen",
    "lbl_error_validate": "Validierung fehlgeschlagen",
    "lbl_doctor_title": "Doctor-Diagnose",
    "lbl_doctor_run": "Doctor ausführen",
    "lbl_config_overview": "Konfigurationsübersicht",
    "lbl_running_progress": "Wird ausgeführt…",
    "lbl_status_loading": "Wird geladen…",
    "lbl_error_prefix": "Fehler",
    "lbl_show_config": "Konfiguration anzeigen",
}


def _seed_labels(conn: sqlite3.Connection) -> None:
    """Seed i18n labels (idempotent)."""
    labels = [
        ("lbl_page_title", "lbl_page_title", "main", "Model Allocator"),
        ("lbl_heading_main", "lbl_heading_main", "main", "Allocation Models"),
        ("lbl_tab_models", "lbl_tab_models", "main", "Allocation Models"),
        ("lbl_tab_profiles", "lbl_tab_profiles", "main", "Backends"),
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
        ("lbl_col_clients", "lbl_col_clients", "main", "Harness"),
        ("lbl_col_api_base", "lbl_col_api_base", "main", "API base"),
        ("lbl_col_gpu", "lbl_col_gpu", "main", "GPU"),
        ("lbl_col_provider", "lbl_col_provider", "main", "Provider"),
        ("lbl_status_ok", "lbl_status_ok", "main", "OK"),
        ("lbl_status_error", "lbl_status_error", "main", "Error"),
        ("lbl_status_warning", "lbl_status_warning", "main", "Warning"),
        ("lbl_status_unknown", "lbl_status_unknown", "main", "Unknown"),
        ("lbl_status_running", "lbl_status_running", "main", "Running"),
        ("lbl_status_stopped", "lbl_status_stopped", "main", "Stopped"),
        ("lbl_status_validating", "lbl_status_validating", "main", "Validating..."),
        ("lbl_status_hint", "lbl_status_hint", "main", "Status reflects the first allowed harness; Validate checks them all."),
        ("lbl_validated_clients", "lbl_validated_clients", "main", "Validated harnesses"),
        ("lbl_validation_result", "lbl_validation_result", "main", "Validation result"),
        ("lbl_validation_ok_hint", "lbl_validation_ok_hint", "main", "The configuration is valid."),
        ("lbl_validation_warn_hint", "lbl_validation_warn_hint", "main", "Usable but incomplete or unverified — fix the warnings below to reach OK."),
        ("lbl_validation_err_hint", "lbl_validation_err_hint", "main", "Unusable until the errors below are fixed."),
        ("lbl_lang_label", "lbl_lang_label", "main", "Language"),
        ("lbl_pg_setup", "lbl_pg_setup", "main", "Setup"),
        ("lbl_pg_daily", "lbl_pg_daily", "main", "Daily"),
        ("lbl_pg_journals", "lbl_pg_journals", "main", "Journals"),
        ("lbl_pg_reports", "lbl_pg_reports", "main", "Reports"),
        ("lbl_pg_periodic", "lbl_pg_periodic", "main", "Periodic"),
        ("lbl_sg_models", "lbl_sg_models", "main", "Allocation Models"),
        ("lbl_sg_profiles", "lbl_sg_profiles", "main", "Backends"),
        ("lbl_sg_validation", "lbl_sg_validation", "main", "Validation"),
        ("lbl_sg_system", "lbl_sg_system", "main", "System"),
        ("lbl_field_alias", "lbl_field_alias", "main", "Alias name"),
        ("lbl_field_real_model", "lbl_field_real_model", "main", "Real model"),
        ("lbl_field_context", "lbl_field_context", "main", "Context window"),
        ("lbl_field_runtime_profile", "lbl_field_runtime_profile", "main", "Runtime profile"),
        ("lbl_field_lifecycle", "lbl_field_lifecycle", "main", "Lifecycle policy"),
        ("lbl_field_clients", "lbl_field_clients", "main", "Allowed harnesses"),
        ("lbl_lifecycle_persistent", "lbl_lifecycle_persistent", "main", "Persistent"),
        ("lbl_lifecycle_stop_after_step", "lbl_lifecycle_stop_after_step", "main", "Stop after step"),
        ("lbl_lifecycle_cloud_noop", "lbl_lifecycle_cloud_noop", "main", "Cloud noop"),
        ("lbl_empty_models", "lbl_empty_models", "main", "No allocation models configured"),
        ("lbl_empty_profiles", "lbl_empty_profiles", "main", "No backends configured"),
        ("lbl_confirm_delete", "lbl_confirm_delete", "main", "Are you sure you want to delete this model?"),
        ("lbl_error_load", "lbl_error_load", "main", "Failed to load data"),
        ("lbl_error_save", "lbl_error_save", "main", "Failed to save"),
        ("lbl_error_delete", "lbl_error_delete", "main", "Failed to delete"),
        ("lbl_error_validate", "lbl_error_validate", "main", "Validation failed"),
        ("lbl_doctor_title", "lbl_doctor_title", "main", "Doctor Diagnostics"),
        ("lbl_doctor_run", "lbl_doctor_run", "main", "Run Doctor"),
        ("lbl_config_overview", "lbl_config_overview", "main", "Config Overview"),
        ("lbl_running_progress", "lbl_running_progress", "main", "Running…"),
        ("lbl_status_loading", "lbl_status_loading", "main", "Loading…"),
        ("lbl_error_prefix", "lbl_error_prefix", "main", "Error"),
        ("lbl_show_config", "lbl_show_config", "main", "Show Config"),
    ]

    da_translations = {
        "lbl_page_title": "Model Allocator",
        "lbl_heading_main": "Allokeringsmodeller",
        "lbl_tab_models": "Allokeringsmodeller",
        "lbl_tab_profiles": "Backends",
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
        "lbl_col_clients": "Harness",
        "lbl_col_api_base": "API-base",
        "lbl_col_gpu": "GPU",
        "lbl_col_provider": "Udbyder",
        "lbl_status_ok": "OK",
        "lbl_status_error": "Fejl",
        "lbl_status_warning": "Advarsel",
        "lbl_status_unknown": "Ukendt",
        "lbl_status_running": "Kører",
        "lbl_status_stopped": "Stoppet",
        "lbl_status_validating": "Validerer...",
        "lbl_status_hint": "Status afspejler den første tilladte harness; Valider tjekker dem alle.",
        "lbl_validated_clients": "Validerede harnesses",
        "lbl_validation_result": "Valideringsresultat",
        "lbl_validation_ok_hint": "Konfigurationen er gyldig.",
        "lbl_validation_warn_hint": "Brugbar men ufuldstændig eller uverificeret — ret advarslerne herunder for at nå OK.",
        "lbl_validation_err_hint": "Ubrugelig indtil fejlene herunder er rettet.",
        "lbl_lang_label": "Sprog",
        "lbl_pg_setup": "Opsætning",
        "lbl_pg_daily": "Dagligt",
        "lbl_pg_journals": "Journaler",
        "lbl_pg_reports": "Rapporter",
        "lbl_pg_periodic": "Periodisk",
        "lbl_sg_models": "Allokeringsmodeller",
        "lbl_sg_profiles": "Backends",
        "lbl_sg_validation": "Validering",
        "lbl_sg_system": "System",
        "lbl_field_alias": "Alias navn",
        "lbl_field_real_model": "Reel model",
        "lbl_field_context": "Kontekst vindue",
        "lbl_field_runtime_profile": "Runtime profil",
        "lbl_field_lifecycle": "Livscyklus politik",
        "lbl_field_clients": "Tilladte harnesses",
        "lbl_lifecycle_persistent": "Persistent",
        "lbl_lifecycle_stop_after_step": "Stop efter step",
        "lbl_lifecycle_cloud_noop": "Cloud noop",
        "lbl_empty_models": "Ingen allokeringsmodeller konfigureret",
        "lbl_empty_profiles": "Ingen backends konfigureret",
        "lbl_confirm_delete": "Er du sikker på at du vil slette denne model?",
        "lbl_error_load": "Kunne ikke hente data",
        "lbl_error_save": "Kunne ikke gemme",
        "lbl_error_delete": "Kunne ikke slette",
        "lbl_error_validate": "Validering fejlede",
        "lbl_doctor_title": "Doctor Diagnostik",
        "lbl_doctor_run": "Kør Doctor",
        "lbl_config_overview": "Konfiguration Oversigt",
        "lbl_running_progress": "Kører…",
        "lbl_status_loading": "Indlæser…",
        "lbl_error_prefix": "Fejl",
        "lbl_show_config": "Vis konfiguration",
    }

    # Upsert, not insert-if-absent: the labels are seed-owned (there is no
    # editing UI), so the DB must follow the code. The old insert-only seeding
    # meant a renamed label text never reached an existing allocator.db, and
    # non-DA locales were only seeded when the key happened to exist in the
    # DA dict.
    for label_id, label_key, domain, default_text in labels:
        conn.execute(
            "INSERT INTO ui_labels (label_id, label_key, label_domain, default_text) "
            "VALUES (?, ?, ?, ?) "
            "ON CONFLICT(label_key) DO UPDATE SET "
            "default_text = excluded.default_text, updated_at = CURRENT_TIMESTAMP",
            (label_id, label_key, domain, default_text),
        )
        locale_texts = {
            "en-US": default_text,
            "da-DK": da_translations.get(label_key, default_text),
            # Mandatory locales get REAL translations (12_CODING_STANDARD:
            # en-US, da-DK, de-DE, es-ES); optional extras fall back to en-US.
            "de-DE": DE_TRANSLATIONS.get(label_key, default_text),
            "es-ES": ES_TRANSLATIONS.get(label_key, default_text),
            "el-GR": default_text,
            "sv-SE": default_text,
        }
        for locale, text in locale_texts.items():
            conn.execute(
                "INSERT INTO ui_label_translations (label_id, locale, translated_text) "
                "VALUES (?, ?, ?) "
                "ON CONFLICT(label_id, locale) DO UPDATE SET "
                "translated_text = excluded.translated_text, updated_at = CURRENT_TIMESTAMP",
                (label_id, locale, text),
            )


def _seed_panel_subgroups(conn: sqlite3.Connection) -> None:
    """Seed panel subgroups for allocator UI (idempotent upsert).

    Allocation Models (and the Validation results panel its Validate buttons
    write into) lives under Daily — it is the everyday overview. The
    sg_setup_* keys are historic and kept, because user collapse state
    references them; the group they belong to is data.
    """
    subgroups = [
        ("sg_setup_models", "daily", "Allokeringsmodeller", "Allocation Models", 1, 1),
        ("sg_setup_validation", "daily", "Validering", "Validation", 2, 1),
        ("sg_setup_profiles", "setup", "Backends", "Backends", 1, 1),
        ("sg_setup_system", "setup", "System", "System", 2, 1),
    ]
    for sg_key, group, title_da, title_en, sort_order, is_visible in subgroups:
        conn.execute(
            "INSERT INTO panel_subgroups (subgroup_key, group_name, title_da, title_en, sort_order, is_visible) "
            "VALUES (?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(subgroup_key) DO UPDATE SET "
            "group_name = excluded.group_name, title_da = excluded.title_da, "
            "title_en = excluded.title_en, sort_order = excluded.sort_order, "
            "is_visible = excluded.is_visible",
            (sg_key, group, title_da, title_en, sort_order, is_visible),
        )

    # Empty groups stay hidden until they gain content (Human decision
    # 2026-08-30). Deterministic on every start — there is no UI to re-show
    # them, so a stale 'expanded' row must not resurrect an empty shell.
    for group in ("journals", "reports", "periodic"):
        conn.execute(
            "INSERT OR REPLACE INTO user_panel_groups (user_id, group_name, state, is_visible, updated_at) "
            "VALUES ('default', ?, 'collapsed', 0, datetime('now'))",
            (group,),
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
                "SELECT state, is_visible FROM user_panel_groups WHERE user_id = 'default' AND group_name = ?",
                (g,),
            ).fetchone()
            result[g] = {
                "is_visible": bool(grp_state["is_visible"]) if grp_state else True,
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
#: Harness clients the allocator has adapters or launch knowledge for.
#: The effective vocabulary is this set UNION whatever aliases already
#: declare, so a hand-added client key never disappears from the UI.
KNOWN_CLIENTS = ["opencode", "claude-code", "pi", "headless", "freebuff", "qwen"]


@app.get("/api/clients")
async def list_clients():
    """The harness-client vocabulary the edit form offers."""
    raw = load_raw(CONFIG_DIR)
    declared: list[str] = []
    for definition in raw["aliases"].values():
        for client in (definition.get("clients") or {}):
            if client not in declared:
                declared.append(client)
    merged = list(KNOWN_CLIENTS)
    for client in declared:
        if client not in merged:
            merged.append(client)
    return {"clients": merged}


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
            # The full declared dict, disabled keys included, so the edit
            # form can round-trip every client the YAML names.
            "clients_declared": dict(clients),
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

    # The clients dict arrives whole. The previous shape hardcoded
    # opencode/claude-code, so saving an alias that declared pi, headless,
    # freebuff or qwen silently DROPPED those clients from models.yaml.
    clients_body = body.get("clients")
    if not isinstance(clients_body, dict) or not clients_body:
        raise HTTPException(status_code=400, detail="clients dict is required")
    clients = {str(name): bool(enabled) for name, enabled in clients_body.items()}

    definition = {
        "runtime_profile": body.get("runtime_profile", ""),
        "real_model": body.get("real_model", ""),
        "context": int(body.get("context", 131072)),
        "lifecycle_policy": body.get("lifecycle_policy", "stop_after_step"),
        "clients": clients,
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


def _enabled_clients(alias: str) -> list[str]:
    """The clients an alias itself declares as enabled, in declaration order."""
    definition = load_raw(CONFIG_DIR)["aliases"].get(alias) or {}
    return [c for c, enabled in (definition.get("clients") or {}).items() if enabled]


def _aggregate_validations(per_client: dict[str, dict]) -> dict:
    """Fold per-client validation results into one response.

    Keeps the single-client response shape (``validation_status``, ``errors``,
    ``warnings``, ``resolved_*``) so existing callers keep working, and adds
    ``validated_clients`` plus ``per_client`` for callers that want the detail.
    Errors and warnings are prefixed with their client, because "incompatible
    with backend 'anthropic'" is only actionable once you know which client
    raised it.
    """
    statuses = [r.get("validation_status", "UNKNOWN") for r in per_client.values()]
    if all(s == "OK" for s in statuses):
        status = "OK"
    elif any(s == "ERROR" for s in statuses):
        status = "ERROR"
    else:
        status = next((s for s in statuses if s != "OK"), "UNKNOWN")

    base = dict(next(iter(per_client.values())))
    base["validation_status"] = status
    base["validated_clients"] = list(per_client)
    base["client_support"] = {
        client: result.get("client_support", {})
        for client, result in per_client.items()
    }
    base["errors"] = [
        f"[{client}] {message}"
        for client, result in per_client.items()
        for message in result.get("errors", [])
    ]
    base["warnings"] = [
        f"[{client}] {message}"
        for client, result in per_client.items()
        for message in result.get("warnings", [])
    ]
    base["per_client"] = per_client
    return base


@app.post("/api/models/{alias}/validate")
async def validate_model(alias: str, request: Request):
    """Validate an allocation model.

    A caller may name a client explicitly (``{"client": "opencode"}``), which
    is the CLI's behaviour. When no client is named, the alias is validated
    against the clients IT declares.

    Validating against a fixed client was wrong: a claude-code-only alias
    (fable5, opus5, sonnet5, imple01-claude, ...) is legitimately incompatible
    with opencode, so a hardcoded 'opencode' reported a correct configuration
    as ERROR — 10 of the 19 current aliases. The status column in the model
    list never showed those errors, because it validates against the alias's
    own first enabled client, so the button and the table disagreed.
    """
    body = await request.json() if request.headers.get("content-type") == "application/json" else {}
    client = (body.get("client") or "").strip() if isinstance(body, dict) else ""

    resolver = Resolver(config_dir=CONFIG_DIR)
    validator = Validator(resolver=resolver)
    try:
        if client:
            return validator.validate(alias, client)

        clients = _enabled_clients(alias)
        if not clients:
            return {
                "validation_status": "ERROR",
                "logical_model_alias": alias,
                "validated_clients": [],
                "errors": [f"alias '{alias}' has no enabled client to validate against"],
                "warnings": [],
            }
        return _aggregate_validations(
            {c: validator.validate(alias, c) for c in clients}
        )
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


def run():
    """Entry point for model-allocator-web command."""
    import uvicorn
    port = int(os.environ.get("ALLOCATOR_WEB_PORT", "9141"))
    host = os.environ.get("ALLOCATOR_WEB_HOST", "0.0.0.0")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run()
