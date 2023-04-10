#!/bin/bash

path=./files/
mkdir -p $path

csvFile=./csv_file.csv

while IFS=, read -r id ref path; do

    # If header of csv
    if [ $id = "id" ]; then
        continue
    fi

    # If we already have file
    if test -f "./files/$id.mscz"; then
        echo "Already downloaded ./files/$id.mscz"
        continue
    fi

    wget -O ./files/$id.mscz https://infura-ipfs.io$ref --connect-timeout=2 --read-timeout=2 --tries=2 --quiet
    echo "Downloaded ./files/$id.mscz"
done < $csvFile