/* ════════════════════════════════════════════════════════════════
   Model Allocator Frontend — app.js
   Vanilla JS, no innerHTML, follows DPMtF frontend governance.
   ════════════════════════════════════════════════════════════════ */

/* ── 1. i18n loader ─────────────────────────────────── */
var labelMap = {};
var currentLocale = "en-US";

function loadLabels() {
    return fetch("/api/user-language")
        .then(function (resp) { return resp.json(); })
        .then(function (data) {
            currentLocale = data.locale || "en-US";
            return fetch("/api/ui-labels/main?locale=" + encodeURIComponent(currentLocale));
        })
        .then(function (resp) { return resp.json(); })
        .then(function (data) {
            labelMap = data.labels || {};
            var slots = document.querySelectorAll("[data-slot]");
            for (var i = 0; i < slots.length; i++) {
                var key = slots[i].getAttribute("data-slot");
                if (labelMap[key]) {
                    slots[i].textContent = labelMap[key];
                }
            }
        })
        .catch(function (err) {
            console.warn("Failed to load labels:", err.message);
        });
}

function switchLanguage(newLocale) {
    fetch("/api/user-language", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ locale: newLocale }),
    }).then(function () {
        currentLocale = newLocale;
        fetch("/api/ui-labels/main?locale=" + encodeURIComponent(newLocale))
            .then(function (resp) { return resp.json(); })
            .then(function (data) {
                labelMap = data.labels || {};
                var slots = document.querySelectorAll("[data-slot]");
                for (var i = 0; i < slots.length; i++) {
                    var key = slots[i].getAttribute("data-slot");
                    if (labelMap[key]) {
                        slots[i].textContent = labelMap[key];
                    }
                }
                renderAll();
            })
            .catch(function (err) {
                console.warn("Failed to switch language:", err.message);
            });
    });
}

function lbl(key, fallback) {
    return labelMap[key] || fallback || key;
}

/* ── 2. Language dropdown ───────────────────────────── */
function initLangDropdown() {
    var dropdown = document.getElementById("lang-dropdown");
    if (!dropdown) return;

    fetch("/api/available-languages")
        .then(function (resp) { return resp.json(); })
        .then(function (data) {
            var langs = data.languages || [];
            for (var i = 0; i < langs.length; i++) {
                var opt = document.createElement("option");
                opt.value = langs[i].locale;
                opt.textContent = langs[i].display_name;
                if (langs[i].locale === currentLocale) opt.selected = true;
                dropdown.appendChild(opt);
            }
        })
        .catch(function () {});

    dropdown.addEventListener("change", function () {
        switchLanguage(dropdown.value);
    });

    var metaLocale = document.querySelector('meta[name="locale"]');
    if (metaLocale) metaLocale.setAttribute("content", currentLocale);
}

/* ── 3. Panel group collapse/expand ─────────────────── */
function initPanelGroups() {
    var headers = document.querySelectorAll(".panel-group-header");
    for (var i = 0; i < headers.length; i++) {
        headers[i].addEventListener("click", function () {
            var pg = this.closest(".panel-group");
            var groupName = this.getAttribute("data-group");
            var isCollapsed = pg.classList.contains("collapsed");
            var newState = isCollapsed ? "expanded" : "collapsed";

            if (newState === "collapsed") {
                pg.classList.add("collapsed");
                var toggle = pg.querySelector(".panel-group-toggle");
                if (toggle) toggle.textContent = "▶";
            } else {
                pg.classList.remove("collapsed");
                var toggle = pg.querySelector(".panel-group-toggle");
                if (toggle) toggle.textContent = "▼";
            }

            fetch("/api/user-panel-groups", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ group_name: groupName, state: newState }),
            }).catch(function () {});
        });
    }

    var sgHeaders = document.querySelectorAll(".panel-subgroup-header");
    for (var j = 0; j < sgHeaders.length; j++) {
        sgHeaders[j].addEventListener("click", function () {
            var sg = this.closest(".panel-subgroup");
            var sgKey = this.getAttribute("data-subgroup");
            var isCollapsed = sg.classList.contains("collapsed");
            var newState = isCollapsed ? "expanded" : "collapsed";

            if (newState === "collapsed") {
                sg.classList.add("collapsed");
                var toggle = sg.querySelector(".panel-subgroup-toggle");
                if (toggle) toggle.textContent = "▶";
            } else {
                sg.classList.remove("collapsed");
                var toggle = sg.querySelector(".panel-subgroup-toggle");
                if (toggle) toggle.textContent = "▼";
            }

            fetch("/api/panel-structure/subgroup-state", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ subgroup_key: sgKey, state: newState }),
            }).catch(function () {});
        });
    }
}

function applyPanelStructure() {
    fetch("/api/panel-structure")
        .then(function (resp) { return resp.json(); })
        .then(function (structure) {
            var groups = ["daily", "journals", "reports", "periodic", "setup"];
            for (var g = 0; g < groups.length; g++) {
                var gn = groups[g];
                var info = structure[gn];
                if (!info) continue;
                var pg = document.getElementById("pg-" + gn);
                if (!pg) continue;

                if (info.is_visible === false) {
                    pg.classList.add("dpmtf-hidden");
                    continue;
                }
                pg.classList.remove("dpmtf-hidden");

                if (info.state === "collapsed") {
                    pg.classList.add("collapsed");
                    var toggle = pg.querySelector(".panel-group-toggle");
                    if (toggle) toggle.textContent = "▶";
                }
            }
        })
        .catch(function () {});
}

/* ── 4. Allocation Models ───────────────────────────── */
var knownClients = [];

function loadKnownClients() {
    return fetch("/api/clients")
        .then(function (resp) { return resp.json(); })
        .then(function (data) { knownClients = data.clients || []; })
        .catch(function (err) {
            console.warn("Failed to load harness clients:", err.message);
        });
}

function renderModels() {
    var container = document.getElementById("models-content");
    if (!container) return;
    container.textContent = "";

    fetch("/api/models")
        .then(function (resp) { return resp.json(); })
        .then(function (data) {
            var modelsData = data.models || [];
            if (modelsData.length === 0) {
                var empty = document.createElement("p");
                empty.className = "empty-state";
                empty.textContent = lbl("lbl_empty_models", "No allocation models configured");
                container.appendChild(empty);
                return;
            }

            var wrapper = document.createElement("div");
            wrapper.className = "table-wrapper";
            var table = document.createElement("table");

            var thead = document.createElement("thead");
            var headRow = document.createElement("tr");
            var cols = ["lbl_col_alias", "lbl_col_backend", "lbl_col_model",
                        "lbl_col_context", "lbl_col_lifecycle", "lbl_col_clients",
                        "lbl_col_status", "lbl_col_actions"];
            for (var c = 0; c < cols.length; c++) {
                var th = document.createElement("th");
                th.textContent = lbl(cols[c], cols[c]);
                headRow.appendChild(th);
            }
            thead.appendChild(headRow);
            table.appendChild(thead);

            var tbody = document.createElement("tbody");
            for (var i = 0; i < modelsData.length; i++) {
                var m = modelsData[i];
                var row = document.createElement("tr");

                var tdAlias = document.createElement("td");
                tdAlias.textContent = m.alias;
                row.appendChild(tdAlias);

                var tdBackend = document.createElement("td");
                tdBackend.textContent = m.runtime_profile || "";
                row.appendChild(tdBackend);

                var tdModel = document.createElement("td");
                tdModel.textContent = m.real_model || "";
                row.appendChild(tdModel);

                var tdCtx = document.createElement("td");
                tdCtx.textContent = String(m.context || 0);
                row.appendChild(tdCtx);

                var tdLife = document.createElement("td");
                tdLife.textContent = m.lifecycle_policy || "";
                row.appendChild(tdLife);

                var tdClients = document.createElement("td");
                tdClients.textContent = (m.clients || []).join(", ");
                row.appendChild(tdClients);

                var tdStatus = document.createElement("td");
                var badge = document.createElement("span");
                var st = (m.validation_status || "unknown").toLowerCase();
                badge.className = "status-badge status-" + st;
                badge.textContent = lbl("lbl_status_" + st, st);
                badge.title = lbl("lbl_status_hint", "Status reflects the first allowed harness; Validate checks them all.");
                tdStatus.appendChild(badge);
                row.appendChild(tdStatus);

                var tdActions = document.createElement("td");
                var ag = document.createElement("div");
                ag.className = "action-group";

                var btnVal = document.createElement("button");
                btnVal.className = "btn btn-small";
                btnVal.textContent = lbl("lbl_btn_validate", "Validate");
                btnVal.addEventListener("click", createValidateHandler(m.alias));
                ag.appendChild(btnVal);

                var btnStart = document.createElement("button");
                btnStart.className = "btn btn-small";
                btnStart.textContent = lbl("lbl_btn_start", "Start");
                btnStart.addEventListener("click", createStartHandler(m.alias));
                ag.appendChild(btnStart);

                var btnStop = document.createElement("button");
                btnStop.className = "btn btn-small";
                btnStop.textContent = lbl("lbl_btn_stop", "Stop");
                btnStop.addEventListener("click", createStopHandler(m.alias));
                ag.appendChild(btnStop);

                var btnEdit = document.createElement("button");
                btnEdit.className = "btn btn-small";
                btnEdit.textContent = lbl("lbl_btn_edit", "Edit");
                btnEdit.addEventListener("click", createEditHandler(m));
                ag.appendChild(btnEdit);

                var btnDel = document.createElement("button");
                btnDel.className = "btn btn-small btn-danger";
                btnDel.textContent = lbl("lbl_btn_delete", "Delete");
                btnDel.addEventListener("click", createDeleteHandler(m.alias));
                ag.appendChild(btnDel);

                tdActions.appendChild(ag);
                row.appendChild(tdActions);

                tbody.appendChild(row);
            }
            table.appendChild(tbody);
            wrapper.appendChild(table);
            container.appendChild(wrapper);

            var hint = document.createElement("p");
            hint.className = "table-hint";
            hint.textContent = lbl("lbl_status_hint", "Status reflects the first allowed harness; Validate checks them all.");
            container.appendChild(hint);

            var btnNew = document.createElement("button");
            btnNew.className = "btn btn-primary btn-detached";
            btnNew.textContent = lbl("lbl_btn_new_model", "New Model");
            btnNew.addEventListener("click", function () { showModelModal(null); });
            container.appendChild(btnNew);
        })
        .catch(function (err) {
            container.textContent = lbl("lbl_error_load", "Failed to load data") + ": " + err.message;
        });
}

function renderValidationResult(alias, result) {
    var container = document.getElementById("validation-content");
    if (!container) return;
    container.textContent = "";

    var card = document.createElement("div");
    card.className = "dpmtf-card";

    var heading = document.createElement("h3");
    heading.textContent = lbl("lbl_validation_result", "Validation result") + ": " + alias;
    card.appendChild(heading);

    var status = (result.validation_status || "unknown").toLowerCase();
    var badge = document.createElement("span");
    badge.className = "status-badge status-" + status;
    badge.textContent = lbl("lbl_status_" + status, status);
    card.appendChild(badge);

    // What the status means, and what it takes to reach OK.
    var hintKey = status === "ok" ? "lbl_validation_ok_hint"
        : status === "warning" ? "lbl_validation_warn_hint"
        : "lbl_validation_err_hint";
    var hint = document.createElement("p");
    hint.className = "validation-hint";
    hint.textContent = lbl(hintKey, "");
    card.appendChild(hint);

    var clients = result.validated_clients || [];
    if (clients.length) {
        var cl = document.createElement("p");
        cl.className = "validation-meta";
        cl.textContent = lbl("lbl_validated_clients", "Validated harnesses") + ": " + clients.join(", ");
        card.appendChild(cl);
    }

    function appendLines(items, cssClass) {
        if (!items || !items.length) return;
        var list = document.createElement("ul");
        list.className = cssClass;
        for (var i = 0; i < items.length; i++) {
            var li = document.createElement("li");
            li.textContent = items[i];
            list.appendChild(li);
        }
        card.appendChild(list);
    }
    appendLines(result.errors, "validation-errors");
    appendLines(result.warnings, "validation-warnings");

    container.appendChild(card);

    // Make sure the results are actually visible when the button is pressed.
    var sg = container.closest(".panel-subgroup");
    if (sg) sg.classList.remove("collapsed");
}

function createValidateHandler(alias) {
    return function () {
        // Send NO client: the backend validates the alias against the clients
        // it declares. Hardcoding "opencode" here reported every
        // claude-code-only alias as broken (fable5, opus5, sonnet5, ...) while
        // the status column — which uses the alias's own clients — said OK.
        fetch("/api/models/" + encodeURIComponent(alias) + "/validate", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({}),
        })
            .then(function (resp) { return resp.json(); })
            .then(function (result) {
                renderValidationResult(alias, result);
                renderModels();
            })
            .catch(function (err) {
                renderValidationResult(alias, {
                    validation_status: "ERROR",
                    errors: [lbl("lbl_error_validate", "Validation failed") + ": " + err.message],
                });
            });
    };
}

function createStartHandler(alias) {
    return function () {
        fetch("/api/models/" + encodeURIComponent(alias) + "/start", { method: "POST" })
            .then(function (resp) { return resp.json(); })
            .then(function (result) {
                if (result.status === "error") {
                    alert(alias + ": " + (result.error || result.stderr || "error"));
                }
                renderModels();
            })
            .catch(function (err) {
                alert(alias + ": " + err.message);
            });
    };
}

function createStopHandler(alias) {
    return function () {
        fetch("/api/models/" + encodeURIComponent(alias) + "/stop", { method: "POST" })
            .then(function (resp) { return resp.json(); })
            .then(function (result) {
                if (result.status === "error") {
                    alert(alias + ": " + (result.error || result.stderr || "error"));
                }
                renderModels();
            })
            .catch(function (err) {
                alert(alias + ": " + err.message);
            });
    };
}

function createEditHandler(model) {
    return function () { showModelModal(model); };
}

function createDeleteHandler(alias) {
    return function () {
        if (!confirm(lbl("lbl_confirm_delete", "Delete this model?"))) return;
        fetch("/api/models/" + encodeURIComponent(alias), { method: "DELETE" })
            .then(function (resp) { return resp.json(); })
            .then(function () { renderModels(); })
            .catch(function (err) {
                alert(lbl("lbl_error_delete", "Failed to delete") + ": " + err.message);
            });
    };
}

/* ── 5. Model modal (create/edit) ───────────────────── */
function showModelModal(model) {
    var existing = document.querySelector(".modal-overlay");
    if (existing) existing.remove();

    var overlay = document.createElement("div");
    overlay.className = "modal-overlay";

    var content = document.createElement("div");
    content.className = "modal-content";

    var title = document.createElement("h3");
    title.textContent = model ? lbl("lbl_btn_edit", "Edit") + " " + model.alias : lbl("lbl_btn_new_model", "New Model");
    content.appendChild(title);

    var form = document.createElement("div");

    var aliasInput = createFormField(form, "lbl_field_alias", "Alias name", model ? model.alias : "");
    if (model) aliasInput.disabled = true;

    var modelInput = createFormField(form, "lbl_field_real_model", "Real model", model ? model.real_model : "");
    var ctxInput = createFormField(form, "lbl_field_context", "Context window", model ? String(model.context) : "131072");
    var profileInput = createFormField(form, "lbl_field_runtime_profile", "Runtime profile", model ? model.runtime_profile : "");

    var lifeSelect = document.createElement("select");
    lifeSelect.className = "form-select";
    var policies = [
        { value: "persistent", labelKey: "lbl_lifecycle_persistent" },
        { value: "stop_after_step", labelKey: "lbl_lifecycle_stop_after_step" },
        { value: "cloud_noop", labelKey: "lbl_lifecycle_cloud_noop" },
    ];
    for (var p = 0; p < policies.length; p++) {
        var opt = document.createElement("option");
        opt.value = policies[p].value;
        opt.textContent = lbl(policies[p].labelKey, policies[p].value);
        if (model && model.lifecycle_policy === policies[p].value) opt.selected = true;
        lifeSelect.appendChild(opt);
    }
    var lifeGroup = document.createElement("div");
    lifeGroup.className = "form-group";
    var lifeLabel = document.createElement("label");
    lifeLabel.className = "form-label";
    lifeLabel.textContent = lbl("lbl_field_lifecycle", "Lifecycle policy");
    lifeGroup.appendChild(lifeLabel);
    lifeGroup.appendChild(lifeSelect);
    form.appendChild(lifeGroup);

    var cbGroup = document.createElement("div");
    cbGroup.className = "form-group";
    var cbLabel = document.createElement("label");
    cbLabel.className = "form-label";
    cbLabel.textContent = lbl("lbl_field_clients", "Enabled clients");
    cbGroup.appendChild(cbLabel);

    var cbWrap = document.createElement("div");
    cbWrap.className = "form-checkbox-group";

    // One checkbox per known harness client, plus any client this alias
    // already declares that the vocabulary does not know — the form must be
    // able to round-trip EVERY client models.yaml names, or saving deletes
    // the ones it cannot see (measured: pi/headless/freebuff/qwen vanished).
    var declared = (model && model.clients_declared) ? model.clients_declared : {};
    var names = knownClients.slice();
    Object.keys(declared).forEach(function (name) {
        if (names.indexOf(name) < 0) names.push(name);
    });
    var clientInputs = {};
    names.forEach(function (name) {
        var cbLab = document.createElement("label");
        cbLab.className = "form-checkbox-label";
        var input = document.createElement("input");
        input.type = "checkbox";
        if (model) {
            input.checked = !!declared[name];
        } else {
            input.checked = (name === "opencode");
        }
        cbLab.appendChild(input);
        cbLab.appendChild(document.createTextNode(" " + name));
        cbWrap.appendChild(cbLab);
        clientInputs[name] = input;
    });

    cbGroup.appendChild(cbWrap);
    form.appendChild(cbGroup);

    content.appendChild(form);

    var actions = document.createElement("div");
    actions.className = "modal-actions";

    var btnSave = document.createElement("button");
    btnSave.className = "btn btn-primary";
    btnSave.textContent = lbl("lbl_btn_save", "Save");
    btnSave.addEventListener("click", function () {
        var clientsBody = {};
        Object.keys(clientInputs).forEach(function (name) {
            clientsBody[name] = clientInputs[name].checked;
        });
        var body = {
            alias: aliasInput.value.trim(),
            real_model: modelInput.value.trim(),
            context: parseInt(ctxInput.value, 10) || 131072,
            runtime_profile: profileInput.value.trim(),
            lifecycle_policy: lifeSelect.value,
            clients: clientsBody,
        };
        fetch("/api/models", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(body),
        })
            .then(function (resp) { return resp.json(); })
            .then(function () {
                overlay.remove();
                renderModels();
            })
            .catch(function (err) {
                alert(lbl("lbl_error_save", "Failed to save") + ": " + err.message);
            });
    });
    actions.appendChild(btnSave);

    var btnCancel = document.createElement("button");
    btnCancel.className = "btn";
    btnCancel.textContent = lbl("lbl_btn_cancel", "Cancel");
    btnCancel.addEventListener("click", function () { overlay.remove(); });
    actions.appendChild(btnCancel);

    content.appendChild(actions);
    overlay.appendChild(content);
    document.body.appendChild(overlay);
}

function createFormField(parent, labelKey, fallback, value) {
    var group = document.createElement("div");
    group.className = "form-group";
    var lab = document.createElement("label");
    lab.className = "form-label";
    lab.textContent = lbl(labelKey, fallback);
    group.appendChild(lab);
    var input = document.createElement("input");
    input.className = "form-input";
    input.value = value || "";
    group.appendChild(input);
    parent.appendChild(group);
    return input;
}

/* ── 6. Runtime Profiles ────────────────────────────── */
function renderProfiles() {
    var container = document.getElementById("profiles-content");
    if (!container) return;
    container.textContent = "";

    fetch("/api/profiles")
        .then(function (resp) { return resp.json(); })
        .then(function (data) {
            var profiles = data.profiles || [];
            if (profiles.length === 0) {
                var empty = document.createElement("p");
                empty.className = "empty-state";
                empty.textContent = lbl("lbl_empty_profiles", "No runtime profiles configured");
                container.appendChild(empty);
                return;
            }

            var wrapper = document.createElement("div");
            wrapper.className = "table-wrapper";
            var table = document.createElement("table");

            var thead = document.createElement("thead");
            var headRow = document.createElement("tr");
            // Backend first (Human decision 2026-08-30). The third column
            // used to be headed "Context" while showing the API base, and
            // "Actions" was a permanently empty cell.
            var cols = ["lbl_col_backend", "lbl_col_profile", "lbl_col_api_base",
                        "lbl_col_gpu", "lbl_col_provider"];
            for (var c = 0; c < cols.length; c++) {
                var th = document.createElement("th");
                th.textContent = lbl(cols[c], cols[c]);
                headRow.appendChild(th);
            }
            thead.appendChild(headRow);
            table.appendChild(thead);

            var tbody = document.createElement("tbody");
            for (var i = 0; i < profiles.length; i++) {
                var p = profiles[i];
                var row = document.createElement("tr");

                var tdBackend = document.createElement("td");
                tdBackend.textContent = p.backend;
                row.appendChild(tdBackend);

                var tdName = document.createElement("td");
                tdName.textContent = p.name;
                row.appendChild(tdName);

                var tdApi = document.createElement("td");
                tdApi.textContent = p.default_api_base || p.api_base_env || "";
                row.appendChild(tdApi);

                var tdGpu = document.createElement("td");
                tdGpu.textContent = p.gpu || "";
                row.appendChild(tdGpu);

                var tdProvider = document.createElement("td");
                tdProvider.textContent = p.provider || "";
                row.appendChild(tdProvider);

                tbody.appendChild(row);
            }
            table.appendChild(tbody);
            wrapper.appendChild(table);
            container.appendChild(wrapper);
        })
        .catch(function (err) {
            container.textContent = lbl("lbl_error_load", "Failed to load data") + ": " + err.message;
        });
}

/* ── 7. System / Doctor ─────────────────────────────── */
function renderSystem() {
    var container = document.getElementById("system-content");
    if (!container) return;
    container.textContent = "";

    var card = document.createElement("div");
    card.className = "dpmtf-card";

    var title = document.createElement("h3");
    title.textContent = lbl("lbl_doctor_title", "Doctor Diagnostics");
    card.appendChild(title);

    var btnRun = document.createElement("button");
    btnRun.className = "btn btn-primary";
    btnRun.textContent = lbl("lbl_doctor_run", "Run Doctor");
    btnRun.addEventListener("click", function () {
        var out = document.getElementById("doctor-output");
        if (out) out.textContent = lbl("lbl_running_progress", "Running…");
        fetch("/api/doctor", { method: "POST" })
            .then(function (resp) { return resp.json(); })
            .then(function (data) {
                if (out) out.textContent = JSON.stringify(data, null, 2);
            })
            .catch(function (err) {
                if (out) out.textContent = lbl("lbl_error_prefix", "Error") + ": " + err.message;
            });
    });
    card.appendChild(btnRun);

    var output = document.createElement("pre");
    output.id = "doctor-output";
    output.className = "output-pre";
    output.textContent = "";
    card.appendChild(output);

    container.appendChild(card);

    var configCard = document.createElement("div");
    configCard.className = "dpmtf-card";
    var configTitle = document.createElement("h3");
    configTitle.textContent = lbl("lbl_config_overview", "Config Overview");
    configCard.appendChild(configTitle);

    var configBtn = document.createElement("button");
    configBtn.className = "btn";
    configBtn.textContent = lbl("lbl_show_config", "Show Config");
    configBtn.addEventListener("click", function () {
        var pre = document.getElementById("config-output");
        if (pre) pre.textContent = lbl("lbl_status_loading", "Loading…");
        fetch("/api/config")
            .then(function (resp) { return resp.json(); })
            .then(function (data) {
                if (pre) pre.textContent = JSON.stringify(data, null, 2);
            })
            .catch(function (err) {
                if (pre) pre.textContent = lbl("lbl_error_prefix", "Error") + ": " + err.message;
            });
    });
    configCard.appendChild(configBtn);

    var configPre = document.createElement("pre");
    configPre.id = "config-output";
    configPre.className = "output-pre";
    configPre.textContent = "";
    configCard.appendChild(configPre);

    container.appendChild(configCard);
}

/* ── 8. Render all ──────────────────────────────────── */
function renderAll() {
    renderModels();
    renderProfiles();
    renderSystem();
}

/* ── 9. Init ────────────────────────────────────────── */
document.addEventListener("DOMContentLoaded", function () {
    Promise.all([loadLabels(), loadKnownClients()]).then(function () {
        initLangDropdown();
        initPanelGroups();
        applyPanelStructure();
        renderAll();
    });
});
