# =============================================================================
# FILE: lib/utils/logging.py
# =============================================================================
"""Thread-safe logging utilities."""

import threading

# Global lock for thread-safe printing
print_lock = threading.Lock()


def safe_print(*args, **kwargs):
    """Thread-safe print function."""
    with print_lock:
        print(*args, **kwargs)


def print_progress(current: int, total: int, message: str, extra: str = ""):
    """Print progress indicator."""
    with print_lock:
        print(f"  [{current:3d}/{total}] {message}{extra}")


def print_header(title: str, char: str = "=", width: int = 120):
    """Print a formatted header."""
    with print_lock:
        print(f"\n{char * width}")
        print(title)
        print(f"{char * width}")


def print_section(title: str, char: str = "-", width: int = 110):
    """Print a formatted section header."""
    with print_lock:
        print(f"\n{char * width}")
        print(f"  {title}")
        print(f"{char * width}\n")
