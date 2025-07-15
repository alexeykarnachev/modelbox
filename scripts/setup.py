import os
import subprocess
import time
from pathlib import Path

import gdown
import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

app = typer.Typer(pretty_exceptions_show_locals=False)
console = Console()


def run_command(
    cmd: list[str], check: bool = True, capture_output: bool = False
) -> subprocess.CompletedProcess:
    try:
        if capture_output:
            result = subprocess.run(cmd, check=check, capture_output=True, text=True)
        else:
            result = subprocess.run(cmd, check=check)
        return result
    except subprocess.CalledProcessError as e:
        console.print(f"[red]Command failed: {' '.join(cmd)}[/red]")
        if capture_output and e.stdout:
            console.print(f"[red]STDOUT: {e.stdout}[/red]")
        if capture_output and e.stderr:
            console.print(f"[red]STDERR: {e.stderr}[/red]")
        raise


def check_command_exists(cmd: str) -> bool:
    try:
        subprocess.run([cmd, "--version"], capture_output=True, check=False)
        return True
    except FileNotFoundError:
        return False


def check_docker_running() -> bool:
    try:
        result = run_command(["docker", "info"], capture_output=True, check=False)
        return result.returncode == 0
    except FileNotFoundError:
        return False


def get_docker_compose_cmd() -> str | None:
    if check_command_exists("docker-compose"):
        return "docker-compose"

    try:
        run_command(["docker", "compose", "version"], capture_output=True)
        return "docker compose"
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def get_python_cmd() -> str:
    if os.environ.get("UV_PROJECT_ENVIRONMENT"):
        return "python"

    if check_command_exists("uv"):
        return "uv run python"

    for cmd in ["python3", "python"]:
        if check_command_exists(cmd):
            return cmd

    console.print("Python is not available. Please install Python first.")
    raise typer.Exit(1)


def check_model_exists(model_path: Path) -> bool:
    return model_path.exists() and model_path.stat().st_size > 0


def check_prerequisites() -> tuple[str, str]:
    console.print("[blue]Checking prerequisites...[/blue]")

    if not check_command_exists("docker"):
        console.print(
            "[red]Docker is not installed. Please install Docker first.[/red]"
        )
        raise typer.Exit(1)

    if not check_docker_running():
        console.print("[red]Docker is not running. Please start Docker first.[/red]")
        raise typer.Exit(1)

    docker_compose_cmd = get_docker_compose_cmd()
    if not docker_compose_cmd:
        console.print(
            "[red]Docker Compose is not available. Please install Docker Compose.[/red]"
        )
        raise typer.Exit(1)

    python_cmd = get_python_cmd()

    console.print("[green]✓ Prerequisites checked[/green]")
    return docker_compose_cmd, python_cmd


def download_model_if_needed(model_name: str, model_path: Path, file_id: str) -> None:
    if check_model_exists(model_path):
        console.print(
            f"[green]✓ {model_name} model already exists, skipping download[/green]"
        )
        return

    console.print(f"[blue]Downloading {model_name} model...[/blue]")

    model_path.parent.mkdir(parents=True, exist_ok=True)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task(f"Downloading {model_name}...", total=None)

        try:
            gdown.download(id=file_id, output=str(model_path))
            progress.update(task, completed=True)
        except Exception as e:
            console.print(f"[red]Failed to download {model_name} model: {e}[/red]")
            raise typer.Exit(1) from e

    console.print(f"[green]✓ {model_name} model downloaded successfully[/green]")


def download_models(project_root: Path) -> None:
    console.print("\n[blue]Checking and downloading models...[/blue]")

    models = [
        {
            "name": "Background Removal",
            "path": project_root
            / "model_repository"
            / "bg_removal"
            / "1"
            / "model.onnx",
            "file_id": "17k16xa69f-oBcp2ejrhzuJFRU3u-XmaK",
        },
        {
            "name": "Depth Pro",
            "path": project_root
            / "model_repository"
            / "depth_pro"
            / "1"
            / "model.onnx",
            "file_id": "1OVm9WGqN-YRPcUqdhDEeNAbRYPJzaY6f",
        },
    ]

    for model in models:
        download_model_if_needed(model["name"], model["path"], model["file_id"])


def check_services_running(docker_compose_cmd: str) -> bool:
    try:
        result = run_command(
            [*docker_compose_cmd.split(), "ps", "-q"],
            capture_output=True,
            check=False,
        )
        return bool(result.stdout.strip())
    except subprocess.CalledProcessError:
        return False


def start_services(docker_compose_cmd: str) -> None:
    console.print("\n[blue]Starting services...[/blue]")

    if check_services_running(docker_compose_cmd):
        console.print("[green]✓ Services are already running[/green]")
        return

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("Starting services...", total=None)

        try:
            cmd = [*docker_compose_cmd.split(), "up", "--build", "-d"]
            run_command(cmd)
            progress.update(task, completed=True)
        except subprocess.CalledProcessError as e:
            console.print("[red]Failed to start services[/red]")
            raise typer.Exit(1) from e

    console.print("[green]✓ Services started successfully[/green]")


def check_service_health(
    docker_compose_cmd: str,
    python_cmd: str,
) -> None:
    console.print("\n[blue]Checking service status...[/blue]")
    time.sleep(3)

    try:
        result = run_command([*docker_compose_cmd.split(), "ps"], capture_output=True)
        if "Up" in result.stdout:
            console.print("[green]✓ Services are running![/green]")

            panel_content = f"""
[bold blue]Commands:[/bold blue]
• Stop services: [code]{docker_compose_cmd} down[/code]
• Test inference: [code]{python_cmd} scripts/test_inference.py[/code]"""

            console.print(
                Panel(panel_content, title="ModelBox Services", border_style="green")
            )
        else:
            console.print(
                "[yellow]⚠ Some services may not be running properly[/yellow]"
            )
            console.print(result.stdout)
    except subprocess.CalledProcessError:
        console.print("[yellow]⚠ Could not check service status[/yellow]")


@app.command()
def main(
    force_export: bool = typer.Option(
        False, "--force-download", help="Force re-download of models even if they exist"
    ),
    skip_services: bool = typer.Option(
        False, "--skip-services", help="Skip starting services"
    ),
) -> None:
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)

    console.print(
        Panel.fit(
            "[bold blue]🚀 ModelBox Setup[/bold blue]\n",
            border_style="blue",
        )
    )

    try:
        docker_compose_cmd, python_cmd = check_prerequisites()

        if not force_export:
            download_models(project_root)
        else:
            console.print(
                "\n[yellow]Force download enabled - re-downloading all models[/yellow]"
            )
            bg_model = (
                project_root / "model_repository" / "bg_removal" / "1" / "model.onnx"
            )
            depth_model = (
                project_root / "model_repository" / "depth_pro" / "1" / "model.onnx"
            )
            for model_path in [bg_model, depth_model]:
                if model_path.exists():
                    model_path.unlink()
            download_models(project_root)

        if not skip_services:
            start_services(docker_compose_cmd)
            check_service_health(docker_compose_cmd, python_cmd)
        else:
            console.print("\n[yellow]Skipping service startup as requested[/yellow]")

        console.print("\n[bold green]🎉 Setup completed successfully![/bold green]")
    except KeyboardInterrupt:
        console.print("\n[yellow]Setup interrupted by user[/yellow]")
        raise typer.Exit(1) from None
    except Exception as e:
        console.print(f"\n[red]Setup failed: {e}[/red]")
        raise typer.Exit(1) from e


if __name__ == "__main__":
    app()
