"""Beautiful terminal logging with ANSI colors and emojis for the pipeline."""

import sys
from datetime import datetime

# ANSI escape codes for colors
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"

COLORS = {
    "info": "\033[34m",     # Blue
    "success": "\033[32m",  # Green
    "warning": "\033[33m",  # Yellow
    "error": "\033[31m",    # Red
    "step": "\033[36m",     # Cyan
    "retry": "\033[35m",    # Magenta
}

EMOJIS = {
    "info": "ℹ️ ",
    "success": "✅ ",
    "warning": "⚠️ ",
    "error": "❌ ",
    "step": "➡️ ",
    "retry": "🔁 ",
}

class Logger:
    @staticmethod
    def _log(level: str, message: str, bold: bool = False) -> None:
        color = COLORS.get(level, "")
        emoji = EMOJIS.get(level, "")
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        style = f"{BOLD}" if bold else ""
        
        print(f"{DIM}[{timestamp}]{RESET} {color}{style}{emoji}{message}{RESET}")

    @classmethod
    def info(cls, message: str) -> None:
        """Standard informational messages."""
        cls._log("info", message)
        
    @classmethod
    def success(cls, message: str, bold: bool = False) -> None:
        """Success messages for passing validation."""
        cls._log("success", message, bold)
        
    @classmethod
    def warning(cls, message: str) -> None:
        """Warning messages."""
        cls._log("warning", message)
        
    @classmethod
    def error(cls, message: str, bold: bool = True) -> None:
        """Error messages for failures."""
        cls._log("error", message, bold)
        
    @classmethod
    def step(cls, message: str, bold: bool = True) -> None:
        """Major step announcements."""
        cls._log("step", message, bold)
        
    @classmethod
    def retry(cls, message: str) -> None:
        """Retry loops and fallback messages."""
        cls._log("retry", message)
        
    @classmethod
    def debug(cls, message: str) -> None:
        """Debug messages."""
        cls._log("info", f"[DEBUG] {message}")
        
    @classmethod
    def divider(cls) -> None:
        """Visual divider."""
        print(f"{DIM}{'━' * 70}{RESET}")
