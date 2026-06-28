param(
    [string]$RemoteUrl = "https://github.com/nIBOP/asym-lightgcn-reproducibility.git"
)

$ErrorActionPreference = "Stop"

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    throw "git is not available in PATH"
}

if (-not (Test-Path -LiteralPath ".git")) {
    git init
}

git branch -M main
git add .

$status = git status --short
if ($status) {
    git commit -m "Prepare AsymLightGCN reproducibility package"
} else {
    Write-Host "No changes to commit."
}

$existing = git remote get-url origin 2>$null
if ($LASTEXITCODE -eq 0 -and $existing) {
    git remote set-url origin $RemoteUrl
} else {
    git remote add origin $RemoteUrl
}

git push -u origin main
