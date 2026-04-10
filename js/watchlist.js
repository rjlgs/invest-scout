/**
 * Watchlist management using localStorage
 */

const WATCHLIST_KEY = 'invest-scout-watchlist';
const DEFAULT_WATCHLIST = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'SPY', 'QQQ', 'TSLA', 'NVDA'];

/**
 * Get the user's watchlist from localStorage
 * @returns {string[]} Array of ticker symbols
 */
function getWatchlist() {
    const stored = localStorage.getItem(WATCHLIST_KEY);
    if (stored) {
        try {
            return JSON.parse(stored);
        } catch (e) {
            console.error('Error parsing watchlist:', e);
        }
    }
    // Return default watchlist for new users
    setWatchlist(DEFAULT_WATCHLIST);
    return DEFAULT_WATCHLIST;
}

/**
 * Save the watchlist to localStorage
 * @param {string[]} watchlist - Array of ticker symbols
 */
function setWatchlist(watchlist) {
    localStorage.setItem(WATCHLIST_KEY, JSON.stringify(watchlist));
}

/**
 * Add a ticker to the watchlist
 * @param {string} symbol - Ticker symbol to add
 * @returns {boolean} True if added, false if already exists
 */
function addToWatchlist(symbol) {
    const watchlist = getWatchlist();
    const upperSymbol = symbol.toUpperCase().trim();

    if (!upperSymbol) return false;
    if (watchlist.includes(upperSymbol)) return false;

    watchlist.push(upperSymbol);
    watchlist.sort();
    setWatchlist(watchlist);
    return true;
}

/**
 * Remove a ticker from the watchlist
 * @param {string} symbol - Ticker symbol to remove
 * @returns {boolean} True if removed, false if not found
 */
function removeFromWatchlist(symbol) {
    const watchlist = getWatchlist();
    const upperSymbol = symbol.toUpperCase().trim();
    const index = watchlist.indexOf(upperSymbol);

    if (index === -1) return false;

    watchlist.splice(index, 1);
    setWatchlist(watchlist);
    return true;
}

/**
 * Check if a ticker is in the watchlist
 * @param {string} symbol - Ticker symbol to check
 * @returns {boolean} True if in watchlist
 */
function isInWatchlist(symbol) {
    return getWatchlist().includes(symbol.toUpperCase().trim());
}

/**
 * Clear the entire watchlist
 */
function clearWatchlist() {
    setWatchlist([]);
}

/**
 * Export watchlist as a comma-separated string
 * @returns {string} Comma-separated ticker symbols
 */
function exportWatchlist() {
    return getWatchlist().join(', ');
}

/**
 * Import watchlist from a comma-separated string
 * @param {string} str - Comma-separated ticker symbols
 * @param {string[]} availableTickers - List of valid tickers to filter against
 * @returns {number} Number of tickers imported
 */
function importWatchlist(str, availableTickers) {
    const symbols = str.split(/[,\s]+/)
        .map(s => s.toUpperCase().trim())
        .filter(s => s && availableTickers.includes(s));

    const unique = [...new Set(symbols)].sort();
    setWatchlist(unique);
    return unique.length;
}

// Export for use in other modules
window.Watchlist = {
    get: getWatchlist,
    set: setWatchlist,
    add: addToWatchlist,
    remove: removeFromWatchlist,
    has: isInWatchlist,
    clear: clearWatchlist,
    export: exportWatchlist,
    import: importWatchlist,
};
