"""Rich console utilities for beautiful CLI output."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Generator

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.theme import Theme
from rich.markdown import Markdown

# Custom theme for Active RAG
THEME = Theme({
    "info": "cyan",
    "success": "bold green",
    "warning": "bold yellow",
    "error": "bold red",
    "path.direct": "bold green",
    "path.rag_memory": "bold blue",
    "path.rag_web": "bold magenta",
    "confidence.high": "bold green",
    "confidence.low": "bold yellow",
    "citation": "dim cyan",
})

console = Console(theme=THEME)


def print_banner() -> None:
    """Print the Active RAG banner."""
    banner = """
[bold cyan]╔═══════════════════════════════════════════════════════════════╗
║     _        _   _          ____      _    ____               ║
║    / \\   ___| |_(_)_   ____| __ )    / \\  / ___|              ║
║   / _ \\ / __| __| \\ \\ / / _ \\  _ \\  / _ \\| |  _               ║
║  / ___ \\ (__| |_| |\\ V /  __/ |_) |/ ___ \\ |_| |              ║
║ /_/   \\_\\___|\\__|_| \\_/ \\___|____/_/   \\_\\____|              ║
║                                                               ║
║  [bold white]Refined Retrieval-Augmented Generation[/bold white]                     ║
╚═══════════════════════════════════════════════════════════════╝[/bold cyan]
"""
    console.print(banner)


def print_result(
    answer_text: str,
    path: str,
    confidence: float | None = None,
    citations: list[str] | None = None,
    web_pages_indexed: int = 0,
    reasoning: str = "",
) -> None:
    """Print a beautifully formatted result."""
    # Path styling
    path_styles = {
        "direct": ("path.direct", "⚡ Direct Answer"),
        "rag_memory": ("path.rag_memory", "🧠 From Memory"),
        "rag_web": ("path.rag_web", "🌐 Web Search"),
        "error": ("error", "❌ Error"),
    }
    style, label = path_styles.get(path, ("info", path))

    # Create info table
    info_table = Table(show_header=False, box=None, padding=(0, 2))
    info_table.add_column("Key", style="dim")
    info_table.add_column("Value")

    info_table.add_row("Path", f"[{style}]{label}[/{style}]")

    if confidence is not None:
        conf_style = "confidence.high" if confidence >= 0.7 else "confidence.low"
        conf_bar = "█" * int(confidence * 10) + "░" * (10 - int(confidence * 10))
        info_table.add_row(
            "Confidence",
            f"[{conf_style}]{conf_bar} {confidence:.0%}[/{conf_style}]"
        )

    if reasoning:
        info_table.add_row("Reasoning", f"[dim]{reasoning[:80]}{'...' if len(reasoning) > 80 else ''}[/dim]")

    if web_pages_indexed > 0:
        info_table.add_row("Pages Indexed", f"[info]{web_pages_indexed}[/info]")

    console.print()
    console.print(Panel(info_table, title="[bold]Query Info[/bold]", border_style="dim"))

    # Print answer
    console.print()
    console.print(Panel(
        Markdown(answer_text),
        title="[bold green]Answer[/bold green]",
        border_style="green",
        padding=(1, 2),
    ))

    # Print citations if any
    if citations:
        console.print()
        citation_text = "\n".join(f"[citation]• {url}[/citation]" for url in citations)
        console.print(Panel(
            citation_text,
            title="[bold]Sources[/bold]",
            border_style="cyan",
        ))


@contextmanager
def status_spinner(message: str) -> Generator[None, None, None]:
    """Show a spinner while processing."""
    with console.status(f"[bold cyan]{message}[/bold cyan]", spinner="dots"):
        yield


def create_progress() -> Progress:
    """Create a progress bar for multi-step operations."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    )


def print_error(message: str) -> None:
    """Print an error message."""
    console.print(f"[error]✗ {message}[/error]")


def print_success(message: str) -> None:
    """Print a success message."""
    console.print(f"[success]✓ {message}[/success]")


def print_info(message: str) -> None:
    """Print an info message."""
    console.print(f"[info]ℹ {message}[/info]")


def print_warning(message: str) -> None:
    """Print a warning message."""
    console.print(f"[warning]⚠ {message}[/warning]")
