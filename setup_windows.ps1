param(
    [string]$Python = "python",
    [string]$VenvPath = ".venv"
)

$ErrorActionPreference = "Stop"

Write-Host "==> Checking Python..."
try {
    & $Python --version | Out-Host
} catch {
    Write-Host "❌ Python not found. Install Python 3.12+ from https://www.python.org/downloads/"
    exit 1
}

Write-Host "==> Creating virtual environment at $VenvPath..."
& $Python -m venv $VenvPath

$activateScript = Join-Path $VenvPath "Scripts\Activate.ps1"
if (-not (Test-Path $activateScript)) {
    Write-Host "❌ Failed to create virtual environment. Please check your Python installation."
    exit 1
}

Write-Host "==> Activating virtual environment..."
& $activateScript

Write-Host "==> Upgrading pip..."
python -m pip install --upgrade pip

Write-Host "==> Installing dependencies..."
pip install -r requirements.txt

Write-Host "✅ Setup complete."
Write-Host "To activate later: .\$VenvPath\Scripts\Activate.ps1"
