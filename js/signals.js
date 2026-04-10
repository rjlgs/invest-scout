/**
 * Signal display and table rendering
 */

// Global state
let allSignals = [];
let availableTickers = [];
let showAllTickers = false;
let sortColumn = 0;
let sortDirection = 'asc';

/**
 * Load signals data from JSON file
 */
async function loadSignals() {
    try {
        const response = await fetch('data/signals.json');
        if (!response.ok) throw new Error('Failed to load signals');
        allSignals = await response.json();
        return allSignals;
    } catch (error) {
        console.error('Error loading signals:', error);
        return [];
    }
}

/**
 * Load metadata (available tickers list)
 */
async function loadMeta() {
    try {
        const response = await fetch('data/meta.json');
        if (!response.ok) throw new Error('Failed to load meta');
        const meta = await response.json();
        // Available tickers come from signals.json (allSignals), not meta.json
        // This ensures users can add any ticker that was scanned
        availableTickers = allSignals.map(s => s.ticker).sort();
        return meta;
    } catch (error) {
        console.error('Error loading meta:', error);
        return { tickers: [], last_updated: null };
    }
}

/**
 * Filter signals based on watchlist or show all
 */
function getFilteredSignals() {
    if (showAllTickers) {
        return allSignals;
    }
    const watchlist = Watchlist.get();
    return allSignals.filter(s => watchlist.includes(s.ticker));
}

/**
 * Sort signals by column
 */
function sortSignals(signals, column, direction) {
    const sorted = [...signals];
    const dir = direction === 'asc' ? 1 : -1;

    sorted.sort((a, b) => {
        let aVal, bVal;

        switch (column) {
            case 0: // Ticker
                aVal = a.ticker;
                bVal = b.ticker;
                break;
            case 1: // Price
                aVal = a.price || 0;
                bVal = b.price || 0;
                return (aVal - bVal) * dir;
            case 2: // Signal
                aVal = a.signal_type;
                bVal = b.signal_type;
                break;
            case 3: // On-Ramp
                aVal = a.on_ramp_score || 0;
                bVal = b.on_ramp_score || 0;
                return (aVal - bVal) * dir;
            case 4: // Off-Ramp
                aVal = a.off_ramp_score || 0;
                bVal = b.off_ramp_score || 0;
                return (aVal - bVal) * dir;
            case 5: // Updated
                aVal = a.timestamp || '';
                bVal = b.timestamp || '';
                break;
            default:
                return 0;
        }

        return aVal.localeCompare(bVal) * dir;
    });

    return sorted;
}

/**
 * Render a single table row
 */
function renderRow(signal, inWatchlist) {
    const rowClass = signal.signal_type === 'on-ramp' ? 'row-on-ramp' :
                     signal.signal_type === 'off-ramp' ? 'row-off-ramp' : '';

    const timestamp = signal.timestamp ?
        formatTimestamp(signal.timestamp) : '—';

    const removeBtn = inWatchlist && !showAllTickers ?
        `<button class="btn-remove" onclick="removeTicker('${signal.ticker}')" title="Remove from watchlist">&times;</button>` :
        `<span class="btn-remove-spacer"></span>`;

    return `
        <tr class="${rowClass}">
            <td>
                <span class="ticker-cell">${removeBtn}<a href="ticker.html?symbol=${signal.ticker}" class="ticker-link">${signal.ticker}</a></span>
            </td>
            <td data-sort="${signal.price || 0}">
                ${signal.price ? '$' + signal.price.toFixed(2) : 'N/A'}
            </td>
            <td>
                <span class="badge badge-${signal.signal_type}">${signal.signal_type}</span>
            </td>
            <td data-sort="${signal.on_ramp_score}">
                <div class="score-bar">
                    <span class="score score-green">${signal.on_ramp_score}</span>
                    <div class="score-bar-track">
                        <div class="score-bar-fill green" style="width:${signal.on_ramp_score * 20}%"></div>
                    </div>
                </div>
            </td>
            <td data-sort="${signal.off_ramp_score}">
                <div class="score-bar">
                    <span class="score score-red">${signal.off_ramp_score}</span>
                    <div class="score-bar-track">
                        <div class="score-bar-fill red" style="width:${signal.off_ramp_score * 20}%"></div>
                    </div>
                </div>
            </td>
            <td class="timestamp-cell">${timestamp}</td>
        </tr>
    `;
}

/**
 * Render the signals table
 */
function renderTable() {
    const filtered = getFilteredSignals();
    const sorted = sortSignals(filtered, sortColumn, sortDirection);
    const watchlist = Watchlist.get();

    const tbody = document.getElementById('signal-tbody');
    if (!tbody) return;

    if (sorted.length === 0) {
        tbody.innerHTML = `
            <tr>
                <td colspan="6" style="text-align:center;color:#666;padding:2rem">
                    ${showAllTickers ? 'No signal data available.' : 'No tickers in your watchlist. Add some above!'}
                </td>
            </tr>
        `;
        return;
    }

    tbody.innerHTML = sorted.map(s => renderRow(s, watchlist.includes(s.ticker))).join('');
}

/**
 * Render summary cards
 */
function renderSummaryCards() {
    const filtered = getFilteredSignals();

    const bullish = filtered.filter(s => s.signal_type === 'on-ramp').length;
    const bearish = filtered.filter(s => s.signal_type === 'off-ramp').length;
    const neutral = filtered.filter(s => s.signal_type === 'neutral').length;

    document.getElementById('bullish-count').textContent = bullish;
    document.getElementById('bearish-count').textContent = bearish;
    document.getElementById('neutral-count').textContent = neutral;
}

/**
 * Sort table by column index
 */
function sortTable(colIdx) {
    if (sortColumn === colIdx) {
        sortDirection = sortDirection === 'asc' ? 'desc' : 'asc';
    } else {
        sortColumn = colIdx;
        sortDirection = 'asc';
    }
    renderTable();
}

/**
 * Toggle between watchlist and all tickers view
 */
function setFilter(showAll) {
    showAllTickers = showAll;

    // Update button states
    document.querySelectorAll('.filter-toggle button').forEach(btn => {
        btn.classList.remove('active');
    });
    document.getElementById(showAll ? 'btn-all' : 'btn-watchlist').classList.add('active');

    renderTable();
    renderSummaryCards();
}

/**
 * Add ticker from input
 */
function addTicker() {
    const input = document.getElementById('ticker-input');
    const symbol = input.value.toUpperCase().trim();

    if (!symbol) return;

    if (!availableTickers.includes(symbol)) {
        alert(`"${symbol}" is not in the available ticker list.`);
        return;
    }

    if (Watchlist.add(symbol)) {
        input.value = '';
        renderTable();
        renderSummaryCards();
        hideAutocomplete();
    } else {
        alert(`"${symbol}" is already in your watchlist.`);
    }
}

/**
 * Remove ticker from watchlist
 */
function removeTicker(symbol) {
    if (Watchlist.remove(symbol)) {
        renderTable();
        renderSummaryCards();
    }
}

/**
 * Autocomplete functionality
 */
let autocompleteIndex = -1;

function showAutocomplete(query) {
    const list = document.getElementById('autocomplete-list');
    if (!list || !query) {
        hideAutocomplete();
        return;
    }

    const q = query.toUpperCase();
    const watchlist = Watchlist.get();
    const matches = availableTickers
        .filter(t => t.startsWith(q) && !watchlist.includes(t))
        .slice(0, 8);

    if (matches.length === 0) {
        hideAutocomplete();
        return;
    }

    list.innerHTML = matches.map((t, i) =>
        `<div class="autocomplete-item ${i === autocompleteIndex ? 'selected' : ''}"
              onclick="selectAutocomplete('${t}')">${t}</div>`
    ).join('');
    list.style.display = 'block';
}

function hideAutocomplete() {
    const list = document.getElementById('autocomplete-list');
    if (list) {
        list.style.display = 'none';
        autocompleteIndex = -1;
    }
}

function selectAutocomplete(symbol) {
    const input = document.getElementById('ticker-input');
    input.value = symbol;
    hideAutocomplete();
    addTicker();
}

function handleAutocompleteKeydown(e) {
    const list = document.getElementById('autocomplete-list');
    const items = list ? list.querySelectorAll('.autocomplete-item') : [];

    if (e.key === 'ArrowDown') {
        e.preventDefault();
        autocompleteIndex = Math.min(autocompleteIndex + 1, items.length - 1);
        updateAutocompleteSelection(items);
    } else if (e.key === 'ArrowUp') {
        e.preventDefault();
        autocompleteIndex = Math.max(autocompleteIndex - 1, 0);
        updateAutocompleteSelection(items);
    } else if (e.key === 'Enter') {
        e.preventDefault();
        if (autocompleteIndex >= 0 && items[autocompleteIndex]) {
            selectAutocomplete(items[autocompleteIndex].textContent);
        } else {
            addTicker();
        }
    } else if (e.key === 'Escape') {
        hideAutocomplete();
    }
}

function updateAutocompleteSelection(items) {
    items.forEach((item, i) => {
        item.classList.toggle('selected', i === autocompleteIndex);
    });
}

/**
 * Format timestamp for table display (human readable, localized)
 */
function formatTimestamp(timestamp) {
    if (!timestamp) return '—';
    const date = new Date(timestamp);
    const now = new Date();
    const diffMs = now - date;
    const diffMins = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMs / 3600000);
    const diffDays = Math.floor(diffMs / 86400000);

    // Relative time for recent updates
    if (diffMins < 1) return 'Just now';
    if (diffMins < 60) return `${diffMins}m ago`;
    if (diffHours < 24) return `${diffHours}h ago`;
    if (diffDays < 7) return `${diffDays}d ago`;

    // Absolute date for older updates
    return date.toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
}

/**
 * Format last updated time for header
 */
function formatLastUpdated(timestamp) {
    if (!timestamp) return 'Unknown';
    const date = new Date(timestamp);
    return date.toLocaleString(undefined, {
        month: 'short',
        day: 'numeric',
        hour: 'numeric',
        minute: '2-digit'
    });
}

/**
 * Initialize the signals page
 */
async function initSignals() {
    // Show loading state
    document.body.classList.add('is-loading');

    // Load data - signals first (meta depends on allSignals being populated)
    await loadSignals();
    const meta = await loadMeta();

    // Update last updated time
    const lastUpdated = document.getElementById('last-updated');
    if (lastUpdated && meta.last_updated) {
        lastUpdated.textContent = formatLastUpdated(meta.last_updated);
    }

    // Setup input handlers
    const input = document.getElementById('ticker-input');
    if (input) {
        input.addEventListener('input', (e) => showAutocomplete(e.target.value));
        input.addEventListener('keydown', handleAutocompleteKeydown);
        input.addEventListener('blur', () => setTimeout(hideAutocomplete, 200));
    }

    // Initial render
    renderTable();
    renderSummaryCards();

    // Hide loading state
    document.body.classList.remove('is-loading');
}

// Export for global access
window.sortTable = sortTable;
window.setFilter = setFilter;
window.addTicker = addTicker;
window.removeTicker = removeTicker;
window.initSignals = initSignals;
