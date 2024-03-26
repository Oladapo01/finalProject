$LINK_ID="1Te_14UB-DZ8wYPhHGyDg7LadDTjNzpti"
$OUTPUT_FILE="latin_bert.tar"
$CONFIRM=(Invoke-WebRequest -Uri "https://docs.google.com/uc?export=download&id=$LINK_ID" -UseBasicParsing).Content | Select-String -Pattern 'confirm" value="([0-9A-Za-z_]+)"' | ForEach-Object { $_.Matches.Groups[1].Value }
$UUID=(Invoke-WebRequest -Uri "https://docs.google.com/uc?export=download&id=$LINK_ID" -UseBasicParsing).Content | Select-String -Pattern 'uuid" value="([0-9A-Za-z_-]+)"' | ForEach-Object { $_.Matches.Groups[1].Value }
Invoke-WebRequest -Uri "https://drive.usercontent.google.com/download?export=download&id=$LINK_ID&confirm=$CONFIRM&uuid=$UUID" -OutFile $OUTPUT_FILE
Move-Item -Path $OUTPUT_FILE -Destination "models/"
cd "models/"

# Use 7-Zip to extract the .tar file
& "C:\Program Files\7-Zip\7z.exe" x $OUTPUT_FILE
Remove-Item $OUTPUT_FILE

