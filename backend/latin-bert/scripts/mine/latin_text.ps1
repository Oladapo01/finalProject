# Step 1: Download the file from Google Drive
Write-Host "Downloading file from Google Drive..."
$LinkID = "1GRe3eFmQBDdF1kIT9T75aPTdquaf8Z8s"
$OutputFile = "latin_library_text.tar.gz"

$Confirm = (Invoke-WebRequest -Uri "https://docs.google.com/uc?export=download&id=$LinkID" -UseBasicParsing).Content | Select-String -Pattern 'confirm" value="([0-9A-Za-z_]+)"' | ForEach-Object { $_.Matches.Groups[1].Value }
$UUID = (Invoke-WebRequest -Uri "https://docs.google.com/uc?export=download&id=$LinkID" -UseBasicParsing).Content | Select-String -Pattern 'uuid" value="([0-9A-Za-z_-]+)"' | ForEach-Object { $_.Matches.Groups[1].Value }

Invoke-WebRequest -Uri "https://drive.usercontent.google.com/download?export=download&id=$LinkID&confirm=$Confirm&uuid=$UUID" -OutFile $OutputFile

# Step 2: Move the downloaded file to the data directory
Write-Host "Moving downloaded file to data directory..."
if (-not (Test-Path "data")) {
    New-Item -ItemType Directory -Path "data"
}
Move-Item -Path $OutputFile -Destination "data/"

# Step 3: Use 7-Zip to extract the .tar file
Write-Host "Using 7-Zip to extract the .tar file..."
$ZipPath = "C:\Program Files\7-Zip\7z.exe"
& $ZipPath x "data\$OutputFile" -o"data" -y

# Step 4: Remove the downloaded .tar.gz file
Remove-Item "data\$OutputFile"

Write-Host "Done."
