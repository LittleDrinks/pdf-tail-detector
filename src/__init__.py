"""PDF-only detection and the original geometry API."""
from .pipeline import analyze_pdf
from .legacy import Finding, Line, scan_pdf

__all__ = ["analyze_pdf", "scan_pdf", "Finding", "Line"]
