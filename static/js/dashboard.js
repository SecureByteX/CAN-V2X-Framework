const REFRESH_INTERVAL_MS = 30_000;
const REQUEST_TIMEOUT_MS = 8_000;
const NORMAL_LABELS = new Set(['NORMAL', 'BENIGN']);
let searchDebounce = null;
let latestRefreshId = 0;

document.addEventListener('DOMContentLoaded', () => {
    document.getElementById('refreshButton').addEventListener('click', refreshDashboard);
    document.getElementById('alertFilters').addEventListener('submit', event => {
        event.preventDefault();
        refreshDashboard();
    });
    document.getElementById('alertFilters').addEventListener('change', refreshDashboard);
    const searchInput = document.querySelector('#alertFilters input[name="q"]');
    searchInput.addEventListener('input', () => {
        window.clearTimeout(searchDebounce);
        searchDebounce = window.setTimeout(refreshDashboard, 250);
    });
    document.getElementById('closeDetails').addEventListener('click', () => {
        document.getElementById('alertDetails').close();
    });
    document.getElementById('alertDetails').addEventListener('click', event => {
        if (event.target === event.currentTarget) event.currentTarget.close();
    });

    refreshDashboard();
    window.setInterval(refreshDashboard, REFRESH_INTERVAL_MS);
});

async function requestJson(url) {
    const controller = new AbortController();
    const timeout = window.setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
    try {
        const response = await fetch(url, {
            headers: { Accept: 'application/json' },
            signal: controller.signal,
            cache: 'no-store'
        });
        const body = await response.json().catch(() => ({}));
        if (!response.ok) {
            throw new Error(body.error || `Request failed with status ${response.status}.`);
        }
        return body;
    } finally {
        window.clearTimeout(timeout);
    }
}

async function refreshDashboard() {
    const refreshId = ++latestRefreshId;
    const refreshButton = document.getElementById('refreshButton');
    refreshButton.disabled = true;
    try {
        const [dashboardResult, performanceResult] = await Promise.allSettled([
            requestJson(buildDashboardUrl()),
            requestJson('/api/model-performance')
        ]);

        if (refreshId !== latestRefreshId) return;
        if (dashboardResult.status === 'fulfilled') {
            renderDashboard(dashboardResult.value);
            setBackendNotice('');
        } else {
            renderUnavailableDashboard();
            setBackendNotice(errorMessage(dashboardResult.reason));
        }

        if (performanceResult.status === 'fulfilled') {
            renderPerformance(performanceResult.value);
        } else {
            renderPerformanceUnavailable(errorMessage(performanceResult.reason));
        }
    } finally {
        if (refreshId === latestRefreshId) refreshButton.disabled = false;
    }
}

function buildDashboardUrl() {
    const form = document.getElementById('alertFilters');
    const query = new URLSearchParams(new FormData(form));
    return `/api/soc?${query.toString()}`;
}

function errorMessage(error) {
    if (error && error.name === 'AbortError') return 'Dashboard request timed out.';
    return error?.message || 'Could not connect to the dashboard service.';
}

function setBackendNotice(message) {
    const notice = document.getElementById('backendNotice');
    const text = document.getElementById('backendNoticeText');
    text.textContent = message;
    notice.hidden = !message;
}

function renderUnavailableDashboard() {
    ['canThreatCount', 'v2xThreatCount', 'criticalCount', 'alertCount'].forEach(id => {
        document.getElementById(id).textContent = '—';
    });
    document.getElementById('lastUpdated').textContent = 'Updated: unavailable';
    renderAlerts([]);
    renderDistribution('canDistribution', {});
    renderDistribution('v2xDistribution', {});
    renderHealth(null);
}

function renderDashboard(data) {
    const { summary, health } = data;
    document.getElementById('canThreatCount').textContent = formatCount(summary.can_threats);
    document.getElementById('v2xThreatCount').textContent = formatCount(summary.v2x_threats);
    document.getElementById('criticalCount').textContent = formatCount(summary.critical_alerts);
    document.getElementById('alertCount').textContent = data.has_more
        ? `${formatCount(data.alert_count)} records · latest 100 shown`
        : `${formatCount(data.alert_count)} records`;
    document.getElementById('lastUpdated').textContent = data.last_event_at
        ? `Last data event: ${formatDateTime(data.last_event_at)}`
        : `No event records · checked ${formatDateTime(data.server_time)}`;

    renderHealth(health);
    renderDistribution('canDistribution', summary.distribution.CAN);
    renderDistribution('v2xDistribution', summary.distribution.V2X);
    renderAlerts(data.alerts);
    updateAttackTypeFilter(data.available_attack_types || []);
}

function renderHealth(health) {
    const values = health || {};
    const can = values.can || { ready: false };
    const v2x = values.legacy_synthetic_v2x || { ready: false };
    setStatus('systemStatus', health ? values.status : 'offline');
    setStatus('apiHealth', health?.api?.ready ? 'online' : 'offline');
    setStatus('databaseHealth', health?.database?.ready ? 'online' : 'offline');
    setStatus('canHealth', can.ready ? 'ready' : 'unavailable');
    setStatus('v2xHealth', v2x.ready ? 'ready' : 'unavailable');
    setStatus('feedHealth', values.live_feed?.configured ? 'connected' : 'not configured');
    setStatus('canModelHealth', can.ready ? 'ready' : 'unavailable');
    setStatus('v2xModelHealth', v2x.ready ? 'ready' : 'unavailable');
    setStatus('canReadyCard', can.ready ? 'Ready' : 'Unavailable');
    setStatus('v2xReadyCard', v2x.ready ? 'Ready' : 'Unavailable');
    document.getElementById('canModelName').textContent = can.model || 'Not available';
    document.getElementById('v2xModelName').textContent = v2x.model || 'Not available';

    const evaluationStatus = document.getElementById('v2xEvaluationStatus');
    const veremi = values.veremi || { ready: false };
    setStatus('v2xEvaluationStatus', veremi.ready ? 'VeReMi model loaded' : 'VeReMi model unavailable');
}

function setStatus(elementId, status) {
    const element = document.getElementById(elementId);
    if (!element) return;
    const normalized = String(status || 'unavailable').toLowerCase();
    element.textContent = String(status || 'Unavailable').toUpperCase();
    element.classList.remove('status-online', 'status-ready', 'status-degraded', 'status-offline', 'status-pending');
    if (normalized.includes('unavailable') || normalized === 'offline') {
        element.classList.add('status-offline');
    } else if (
        ['online', 'ready', 'connected'].includes(normalized)
        || normalized.includes('available')
        || normalized.includes('loaded')
    ) {
        element.classList.add('status-ready');
    } else if (
        normalized === 'degraded'
        || normalized === 'not configured'
        || normalized.includes('under evaluation')
        || normalized.includes('below')
        || normalized.includes('checking')
    ) {
        element.classList.add('status-degraded');
    } else if (normalized === 'checking') {
        element.classList.add('status-pending');
    } else {
        element.classList.add('status-offline');
    }
}

function renderDistribution(containerId, distribution) {
    const container = document.getElementById(containerId);
    container.replaceChildren();
    const entries = Object.entries(distribution || {}).sort((left, right) => right[1] - left[1]);
    if (!entries.length) {
        container.append(createEmpty('No verified attack records in this period.'));
        return;
    }

    const maxValue = Math.max(...entries.map(([, count]) => count), 1);
    entries.forEach(([label, count]) => {
        const row = document.createElement('div');
        row.className = 'soc-distribution-row';
        const name = document.createElement('span');
        name.className = 'soc-distribution-label';
        name.textContent = label;
        const track = document.createElement('span');
        track.className = 'soc-bar-track';
        const bar = document.createElement('span');
        bar.className = `soc-bar ${NORMAL_LABELS.has(label.toUpperCase()) ? 'soc-bar-normal' : ''}`;
        bar.style.width = `${Math.max((count / maxValue) * 100, 2)}%`;
        track.append(bar);
        const value = document.createElement('strong');
        value.textContent = formatCount(count);
        row.append(name, track, value);
        container.append(row);
    });
}

function renderAlerts(alerts) {
    const body = document.getElementById('alertsTable');
    body.replaceChildren();
    if (!alerts.length) {
        const row = document.createElement('tr');
        const cell = document.createElement('td');
        cell.colSpan = 6;
        cell.className = 'soc-empty';
        cell.textContent = 'No recorded verified alerts match these filters.';
        row.append(cell);
        body.append(row);
        return;
    }

    alerts.forEach(alert => {
        const row = document.createElement('tr');
        row.className = 'soc-alert-row';
        row.tabIndex = 0;
        row.setAttribute('role', 'button');
        row.setAttribute('aria-label', `${alert.source} ${alert.prediction} alert details`);
        row.addEventListener('click', () => showAlertDetails(alert));
        row.addEventListener('keydown', event => {
            if (event.key === 'Enter' || event.key === ' ') {
                event.preventDefault();
                showAlertDetails(alert);
            }
        });
        appendCell(row, formatDateTime(alert.recorded_at), 'Recorded');
        appendCell(row, alert.source, 'Source');
        const attackCell = document.createElement('td');
        const attack = document.createElement('span');
        attack.className = `soc-severity severity-${safeClass(alert.severity)}`;
        attack.textContent = alert.prediction || 'Not available';
        attackCell.dataset.label = 'Attack';
        attackCell.append(attack);
        row.append(attackCell);
        appendCell(row, formatPercent(alert.confidence), 'Confidence');
        const severityCell = document.createElement('td');
        const severity = document.createElement('span');
        severity.className = `soc-severity severity-${safeClass(alert.severity)}`;
        severity.textContent = alert.severity || 'Not available';
        severityCell.dataset.label = 'Severity';
        severityCell.append(severity);
        row.append(severityCell);
        appendCell(row, 'Confirmed TP · labeled test', 'Outcome');
        body.append(row);
    });
}

function appendCell(row, value, label) {
    const cell = document.createElement('td');
    if (label) cell.dataset.label = label;
    cell.textContent = value == null || value === '' ? 'Not available' : String(value);
    row.append(cell);
}

function showAlertDetails(alert) {
    const details = [
        ['Alert ID', alert.id],
        ['Recorded at', alert.recorded_at],
        ['Source', alert.source],
        ['Origin', alert.origin],
        ['Event timestamp', alert.event_timestamp],
        ['CAN ID / V2X sender', alert.identifier],
        ['Attack type', alert.actual_label],
        ['Predicted class', alert.prediction],
        ['Confidence', formatPercent(alert.confidence)],
        ['Severity', alert.severity],
        ['Model used', alert.model],
        ['Prediction outcome', alert.outcome],
        ['Status', alert.status],
        ['Risk score', null],
        ['Explainability', null],
        ['Threat correlation', null],
        ['Recommended response', null],
    ];
    const list = document.getElementById('detailsList');
    list.replaceChildren();
    details.forEach(([label, value]) => {
        const term = document.createElement('dt');
        term.textContent = label;
        const description = document.createElement('dd');
        description.textContent = displayValue(value);
        list.append(term, description);
    });

    const featureTitle = document.createElement('dt');
    featureTitle.textContent = 'Recorded model features';
    const featureValues = document.createElement('dd');
    featureValues.textContent = alert.features
        ? JSON.stringify(alert.features, null, 2)
        : 'Not available';
    featureValues.className = 'soc-feature-values';
    list.append(featureTitle, featureValues);
    document.getElementById('alertDetails').showModal();
}

function updateAttackTypeFilter(labels) {
    const select = document.getElementById('attackTypeFilter');
    const selected = select.value;
    const current = [...select.options].slice(1).map(option => option.value);
    if (current.length === labels.length && current.every((label, index) => label === labels[index])) return;
    select.replaceChildren(new Option('All attack types', ''));
    labels.forEach(label => select.add(new Option(label, label)));
    select.value = labels.includes(selected) ? selected : '';
}

async function renderPerformance(data) {
    const body = document.getElementById('performanceTable');
    body.replaceChildren();
    const sets = [
        { name: data.can_dataset?.name || 'HCRL CAN', metadata: data.can_dataset, best: data.can_best, results: data.can_models || [] },
        { name: data.veremi_dataset?.name || 'VeReMi', metadata: data.veremi_dataset, best: data.veremi_best, results: data.veremi_models || [] },
    ];
    let rowCount = 0;
    sets.forEach(dataset => {
        dataset.results.forEach(model => {
            const row = document.createElement('tr');
            appendCell(row, `${dataset.name} · ${dataSplit(model, dataset)}`, 'Dataset / split');
            appendCell(row, `${model.model_name || 'Not available'}${model.model_name === dataset.best ? ' · selected (macro F1)' : ''}`, 'Algorithm');
            appendCell(row, formatPercent(model.accuracy), 'Accuracy');
            appendCell(row, formatPercent(model.precision_macro), 'Macro precision');
            appendCell(row, formatPercent(model.recall_macro), 'Macro recall');
            appendCell(row, formatPercent(model.f1_macro), 'Macro F1');
            appendCell(row, formatPercent(model.false_positive_rate), 'False-positive rate');
            appendCell(row, model.inference_time_ms != null && Number.isFinite(Number(model.inference_time_ms))
                ? `${Number(model.inference_time_ms).toFixed(6)}`
                : 'Not available', 'Inference ms/sample');
            body.append(row);
            rowCount += 1;
        });
    });

    if (!rowCount) {
        body.append(createTableMessage(8, 'No stored offline evaluation results are available.'));
        document.getElementById('performanceNote').textContent =
            'Evaluation artifacts are missing or contain no results. No metrics are being substituted.';
        return;
    }
    const veremi = sets[1].results.find(model => model.model_name === sets[1].best);
    const evaluationStatus = document.getElementById('v2xEvaluationStatus');
    const reviewThreshold = Number(data.v2x_macro_f1_review_threshold);
    if (veremi && Number.isFinite(reviewThreshold) && Number(veremi.f1_macro) < reviewThreshold) {
        setStatus('v2xEvaluationStatus', `Under evaluation · below ${(reviewThreshold * 100).toFixed(0)}% macro F1`);
    } else if (veremi) {
        setStatus('v2xEvaluationStatus', 'Evaluation results available');
    } else {
        setStatus('v2xEvaluationStatus', 'VeReMi evaluation unavailable');
    }
    document.getElementById('performanceNote').textContent =
        Number.isFinite(reviewThreshold)
            ? `Offline held-out results only. Best model is selected by macro F1. V2X review threshold: ${(reviewThreshold * 100).toFixed(0)}% macro F1; this status does not change measured results.`
            : 'Offline held-out results only. Best model is selected by macro F1. No V2X review threshold is configured.';
}

function dataSplit(model, dataset) {
    return model.split_strategy || dataset.metadata?.split_strategy || 'stored holdout';
}

function renderPerformanceUnavailable(message) {
    const body = document.getElementById('performanceTable');
    body.replaceChildren(createTableMessage(8, `Offline evaluation unavailable: ${message}`));
    document.getElementById('performanceNote').textContent =
        'Evaluation metrics are not available; no placeholder values are shown.';
    setStatus('v2xEvaluationStatus', 'Evaluation unavailable');
}

function createTableMessage(colSpan, message) {
    const row = document.createElement('tr');
    const cell = document.createElement('td');
    cell.colSpan = colSpan;
    cell.className = 'soc-empty';
    cell.textContent = message;
    row.append(cell);
    return row;
}

function createEmpty(message) {
    const empty = document.createElement('p');
    empty.className = 'soc-empty-text';
    empty.textContent = message;
    return empty;
}

function displayValue(value) {
    return value == null || value === '' ? 'Not available' : String(value);
}

function formatCount(value) {
    return value != null && value !== '' && Number.isFinite(Number(value))
        ? Number(value).toLocaleString()
        : 'Not available';
}

function formatPercent(value) {
    return value != null && value !== '' && Number.isFinite(Number(value))
        ? `${(Number(value) * 100).toFixed(2)}%`
        : 'Not available';
}

function formatDateTime(value) {
    if (!value) return 'Not available';
    const date = new Date(value.includes('T') ? value : `${value.replace(' ', 'T')}Z`);
    return Number.isNaN(date.getTime()) ? String(value) : date.toLocaleString();
}

function safeClass(value) {
    return String(value || 'info').toLowerCase().replace(/[^a-z0-9_-]/g, '');
}
