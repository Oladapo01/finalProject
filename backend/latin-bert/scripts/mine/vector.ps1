# Step 1: Download the file from Google Drive
Write-Host "Downloading file from Google Drive..."
$LinkID = "1zhpyg5vxMT0iSMl7iW7KLW2wZ7phZFKE"
$OutputFile = "latin.200.vectors.txt"

$Confirm = (Invoke-WebRequest -Uri "https://docs.google.com/uc?export=download&id=$LinkID" -UseBasicParsing).Content | Select-String -Pattern 'confirm" value="([0-9A-Za-z_]+)"' | ForEach-Object { $_.Matches.Groups[1].Value }
$UUID = (Invoke-WebRequest -Uri "https://docs.google.com/uc?export=download&id=$LinkID" -UseBasicParsing).Content | Select-String -Pattern 'uuid" value="([0-9A-Za-z_-]+)"' | ForEach-Object { $_.Matches.Groups[1].Value }

Invoke-WebRequest -Uri "https://drive.usercontent.google.com/download?export=download&id=$LinkID&confirm=$Confirm&uuid=$UUID" -OutFile $OutputFile

# Step 2: Move the downloaded file to the data directory
Write-Host "Moving downloaded file to data directory..."
if (-not (Test-Path "data")) {
    New-Item -ItemType Directory -Path "data"
}
Move-Item -Path $OutputFile -Destination "data/"

Write-Host "Done."
