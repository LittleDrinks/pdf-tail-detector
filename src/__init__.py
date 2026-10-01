"""PDF-only detection and its scan_pdf compatibility interface."""
from .compat import Finding, scan_pdf
from .pipeline import analyze_pdf

__all__ = ["analyze_pdf", "scan_pdf", "Finding"]
