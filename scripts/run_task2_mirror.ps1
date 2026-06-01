# Run Task 2 with Hugging Face mirror (recommended in mainland China)
$env:HF_ENDPOINT = "https://hf-mirror.com"
$env:HF_HUB_DOWNLOAD_TIMEOUT = "300"
Set-Location $PSScriptRoot\..
Write-Host "HF_ENDPOINT=$env:HF_ENDPOINT"
python -u scripts/run_task2.py @args
