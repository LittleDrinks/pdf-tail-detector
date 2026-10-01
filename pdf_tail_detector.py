#!/usr/bin/env python3
"""Command-line entry point for the PDF-only detector."""
from src import analyze_pdf as analyze_pdf
from src import scan_pdf as scan_pdf
from src.pipeline import cli

if __name__ == "__main__":
    raise SystemExit(cli())
