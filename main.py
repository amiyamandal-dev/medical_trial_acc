#!/usr/bin/env python3
"""
Clinical Trial Protocol Matcher - Simple CLI

Match requirements documents to clinical trial protocols using vector similarity
or intelligent LLM-based analysis.
"""
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from src.processor import DocumentProcessor
from src.report_generator import ReportGenerator


console = Console()


def print_help():
    """Print usage help"""
    help_text = """
[bold cyan]Clinical Trial Protocol Matcher[/bold cyan]

[bold]Usage:[/bold]

  1. Match a single protocol to requirements (basic similarity):
     [green]python main.py <protocol_file> <requirements_file>[/green]

     Example:
     python main.py data/protocols/diabetes.pdf data/requirements/req.txt

  2. Find best matching protocol from a folder (basic similarity):
     [green]python main.py <protocols_folder> <requirements_file>[/green]

     Example:
     python main.py data/protocols/ data/requirements/req.txt

  3. Intelligent compliance analysis (requires API key):
     [green]python main.py <protocol_or_folder> <requirements_file> --analyze[/green]

     Example:
     python main.py data/protocols/diabetes.pdf data/requirements/req.txt --analyze
     python main.py data/protocols/ data/requirements/req.txt --analyze

[bold]Supported file types:[/bold] .pdf, .txt, .md

[bold]Options:[/bold]
  --help, -h     Show this help message
  --analyze      Use intelligent LLM agent for detailed compliance analysis
                 (requires DEEPSEEK_API_KEY in .env file)

[bold]Analysis Modes:[/bold]

  Basic Mode (default):
    - Fast vector similarity matching
    - Finds best matching protocols
    - Shows similarity scores

  Intelligent Mode (--analyze):
    - Extracts individual requirements using LLM
    - Analyzes each requirement against protocol
    - Provides compliance status (followed/partial/not_followed)
    - Generates detailed explanations
    - Calculates compliance rate
"""
    console.print(Panel(help_text, title="Help", border_style="cyan"))


def format_score(score: float) -> str:
    """Format similarity score with color"""
    percentage = score * 100
    if percentage >= 70:
        return f"[green]{percentage:.1f}%[/green]"
    elif percentage >= 40:
        return f"[yellow]{percentage:.1f}%[/yellow]"
    else:
        return f"[red]{percentage:.1f}%[/red]"


def format_status(status: str) -> str:
    """Format compliance status with color and emoji"""
    status_map = {
        "followed": "[green]✅ Followed[/green]",
        "partial": "[yellow]⚠️  Partial[/yellow]",
        "not_followed": "[red]❌ Not Followed[/red]",
        "not_applicable": "[dim]➖ N/A[/dim]",
        "error": "[red]⚠️  Error[/red]"
    }
    return status_map.get(status, status)


def display_analysis_results(result: dict):
    """Display intelligent analysis results"""
    console.print()
    console.print(Panel(
        f"[bold]Protocol:[/bold] {result['protocol_name']}\n"
        f"[bold]Requirements:[/bold] {result['requirements_name']}",
        title="📊 Compliance Analysis Report",
        border_style="cyan"
    ))

    # Summary
    summary = result['summary']
    console.print()
    console.print("[bold]Summary[/bold]")
    console.print("─" * 60)

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="white")

    table.add_row("Total Requirements", str(summary['total_requirements']))
    table.add_row("✅ Followed", f"[green]{summary['followed']}[/green]")
    table.add_row("⚠️  Partial", f"[yellow]{summary['partial']}[/yellow]")
    table.add_row("❌ Not Followed", f"[red]{summary['not_followed']}[/red]")
    table.add_row("➖ Not Applicable", f"[dim]{summary['not_applicable']}[/dim]")

    if summary['errors'] > 0:
        table.add_row("⚠️  Errors", f"[red]{summary['errors']}[/red]")

    table.add_row("", "")
    compliance_color = "green" if summary['compliance_rate'] >= 70 else "yellow" if summary['compliance_rate'] >= 40 else "red"
    table.add_row("[bold]Compliance Rate[/bold]", f"[bold {compliance_color}]{summary['compliance_rate']:.1f}%[/bold {compliance_color}]")

    console.print(table)

    # Detailed results
    console.print()
    console.print("[bold]Detailed Requirements Analysis[/bold]")
    console.print("─" * 60)

    for i, req_result in enumerate(result['detailed_results'], 1):
        status_formatted = format_status(req_result['compliance_status'])
        confidence = req_result['confidence_score'] * 100

        console.print(f"\n[cyan]{i}.[/cyan] {status_formatted}")
        console.print(f"   [bold]Requirement:[/bold] {req_result['requirement'][:150]}{'...' if len(req_result['requirement']) > 150 else ''}")
        console.print(f"   [bold]Confidence:[/bold] {confidence:.0f}%")
        console.print(f"   [bold]Explanation:[/bold] {req_result['explanation']}")


def main():
    # Parse arguments
    args = [arg for arg in sys.argv[1:] if not arg.startswith('-')]
    flags = [arg for arg in sys.argv[1:] if arg.startswith('-')]

    # Check for help flag
    if "--help" in flags or "-h" in flags or len(sys.argv) == 1:
        print_help()
        return

    # Check for analyze flag
    analyze_mode = "--analyze" in flags

    # Validate arguments
    if len(args) != 2:
        console.print("[red]Error:[/red] Expected 2 arguments (protocol/folder and requirements file)\n", style="bold")
        print_help()
        sys.exit(1)

    arg1 = Path(args[0])
    arg2 = Path(args[1])

    # Validate inputs
    if not arg1.exists():
        console.print(f"[red]Error:[/red] File or directory not found: {arg1}", style="bold")
        sys.exit(1)

    if not arg2.exists():
        console.print(f"[red]Error:[/red] File not found: {arg2}", style="bold")
        sys.exit(1)

    if not arg2.is_file():
        console.print(f"[red]Error:[/red] Requirements must be a file, not a directory: {arg2}", style="bold")
        sys.exit(1)

    # Initialize processor
    try:
        processor = DocumentProcessor(use_agent=analyze_mode)
    except ValueError as e:
        console.print(f"[red]Error:[/red] {str(e)}", style="bold")
        sys.exit(1)

    try:
        if analyze_mode:
            # Intelligent analysis mode
            console.print("\n[bold cyan]Mode:[/bold cyan] Intelligent Compliance Analysis (LLM-based)\n")

            # Progress tracking
            progress_data = {"current": 0, "total": 100, "message": "Starting..."}

            def progress_callback(current, total, message):
                progress_data["current"] = current
                progress_data["total"] = total
                progress_data["message"] = message

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                console=console,
            ) as progress:
                task = progress.add_task("Analyzing...", total=100)

                # Run analysis in a way that updates progress
                if arg1.is_file():
                    result = processor.analyze_compliance(
                        arg1, arg2,
                        progress_callback=lambda c, t, m: (
                            progress_callback(c, t, m),
                            progress.update(task, completed=c, description=m)
                        )
                    )
                else:
                    result = processor.analyze_compliance_folder(
                        arg1, arg2,
                        progress_callback=lambda c, t, m: (
                            progress_callback(c, t, m),
                            progress.update(task, completed=c, description=m)
                        )
                    )

            # Display results
            display_analysis_results(result)

            # Generate and save markdown report
            report_path = ReportGenerator.get_default_report_path(result['requirements_name'])
            ReportGenerator.generate_markdown_report(result, report_path)

            console.print()
            console.print(f"[green]📄 Report saved to:[/green] [cyan]{report_path}[/cyan]")
            console.print(f"[dim]Open with: markdown viewer or text editor[/dim]")

        else:
            # Basic similarity mode
            if arg1.is_file():
                console.print("\n[bold cyan]Mode:[/bold cyan] Single Protocol Match (Vector Similarity)\n")

                with console.status("[bold green]Processing documents..."):
                    result = processor.process_single_match(arg1, arg2)

                # Display results
                console.print(Panel(
                    f"[bold]Protocol:[/bold] {result['protocol_name']}\n"
                    f"[bold]Requirements:[/bold] {result['requirements_name']}\n"
                    f"[bold]Similarity Score:[/bold] {format_score(result['similarity_score'])}",
                    title="Match Results",
                    border_style="green"
                ))

                # Show top matching sections
                if result['top_matches']:
                    console.print("\n[bold]Top Matching Sections:[/bold]\n")
                    for i, match in enumerate(result['top_matches'], 1):
                        score = 1.0 - match.get('distance', 1.0)
                        content = match.get('content', '')[:200] + "..."
                        console.print(f"[cyan]{i}.[/cyan] Score: {format_score(score)}")
                        console.print(f"   {content}\n")

            else:
                console.print("\n[bold cyan]Mode:[/bold cyan] Find Best Protocol from Folder (Vector Similarity)\n")

                with console.status("[bold green]Processing all protocols..."):
                    best_match, all_results = processor.find_best_protocol(arg1, arg2)

                if not best_match:
                    console.print("[red]No matches found![/red]")
                    return

                # Display best match
                console.print(Panel(
                    f"[bold green]✓ Best Matching Protocol:[/bold green]\n\n"
                    f"[bold]File:[/bold] {best_match['protocol_name']}\n"
                    f"[bold]Path:[/bold] {best_match['protocol_path']}\n"
                    f"[bold]Similarity Score:[/bold] {format_score(best_match['similarity_score'])}\n"
                    f"[bold]Max Section Score:[/bold] {format_score(best_match['max_similarity_score'])}\n"
                    f"[bold]Matching Sections:[/bold] {best_match['matching_sections']}",
                    title="Best Match",
                    border_style="green"
                ))

                # Show all results in a table
                if len(all_results) > 1:
                    console.print("\n[bold]All Protocol Rankings:[/bold]\n")

                    table = Table(show_header=True, header_style="bold magenta")
                    table.add_column("Rank", style="cyan", width=6)
                    table.add_column("Protocol Name", style="white")
                    table.add_column("Avg Score", justify="right")
                    table.add_column("Max Score", justify="right")
                    table.add_column("Sections", justify="right")

                    for i, result in enumerate(all_results, 1):
                        table.add_row(
                            f"#{i}",
                            result['protocol_name'],
                            format_score(result['similarity_score']),
                            format_score(result['max_similarity_score']),
                            str(result['matching_sections'])
                        )

                    console.print(table)

    except Exception as e:
        console.print(f"\n[red bold]Error:[/red bold] {str(e)}")
        import traceback
        if "--debug" in flags:
            console.print(traceback.format_exc())
        sys.exit(1)

    console.print("\n[green]✓ Complete![/green]\n")


if __name__ == "__main__":
    main()
