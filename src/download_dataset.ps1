# check if phcd.rar is present in src folder, if yes then don't download it
if (-Not (Test-Path "phcd.rar")) {
    Write-Output "phcd.rar not found in src folder - proceeding with download"
    # download the dataset
    Invoke-WebRequest -OutFile phcd.rar https://cs.pollub.pl/phcd/phcd.rar
    Write-Output "Dataset downloaded."
}
else {
    Write-Output "phcd.rar already present"
}

New-Item -ItemType Directory -Path ../data

# copy to ../data/
Move-Item -Path phcd.rar -Destination ../data/phcd.rar

Write-Output "Unpacking ..."

# unpack the archive
Set-Location ../data
Unblock-File phcd.rar
Expand-Archive -Path phcd.rar -DestinationPath .
Unblock-File phsf.rar
Expand-Archive -Path phsf.rar -DestinationPath .

Write-Output "Unpacked. Cleaning up ..."

# copy image folders to their final directory
New-Item -ItemType Directory -Path all_characters
Move-Item -Path znaki/png/* -Destination all_characters
Remove-Item -Recurse -Force *.py, *.pdf, *.rar, ocr_files, inwokacja, znaki

Write-Output "Done!"
