"""MT5 Client Exceptions."""


class MT5ConnectionError(Exception):
    """Custom exception for MT5 connection errors."""
    pass


class MT5SymbolError(Exception):
    """Custom exception for symbol-related errors."""
    pass


class MT5CacheError(Exception):
    """Custom exception for cache-related errors."""
    pass
