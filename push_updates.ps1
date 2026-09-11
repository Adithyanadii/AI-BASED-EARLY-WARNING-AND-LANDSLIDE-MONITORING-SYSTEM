Write-Host "====================================="
Write-Host "   NER Landslide AI - GitHub Push"
Write-Host "====================================="

git add .

$commitMessage = Read-Host "Enter update message"

if ([string]::IsNullOrWhiteSpace($commitMessage)) {
    $commitMessage = "Update project"
}

git commit -m "$commitMessage"

if ($LASTEXITCODE -ne 0) {
    Write-Host "No new changes to commit."
    exit
}

git push

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "SUCCESS! Project pushed to GitHub." -ForegroundColor Green
} else {
    Write-Host ""
    Write-Host "Push failed. Check the Git message above." -ForegroundColor Red
}

Read-Host "Press Enter to close"