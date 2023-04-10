$path = ".\files\"
If(!(test-path -PathType container $path))
{
      New-Item -ItemType Directory -Path $path
}

Import-Csv ".\csv_file.csv" |
    ForEach-Object -Parallel {
        if (!(Test-Path ".\files\$($_.id).mscz")) {
            try {
                Invoke-WebRequest "https://infura-ipfs.io$($_.ref)" -O ".\files\$($_.id).mscz"
                echo "Downloaded .\files\$($_.id).mscz"
            } catch {
                echo "Error in downloading"
            }
        }
    } -ThrottleLimit 16


echo done