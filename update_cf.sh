#!/bin/bash

# Define base URL
BASE_URL="https://courses.cs.duke.edu/fall24/compsci521"

# Define the directories to download
DIRS=("data-sets" "Mcodes")

# Log file to capture output
LOG_FILE="download.log"

# Set wget options
WGET_OPTS="-r -np -nH --cut-dirs=2 -R 'index.html*'"

# Function to download content
download_content() {
    local url=$1
    echo "Starting download from: $url" | tee -a "$LOG_FILE"
    wget $WGET_OPTS "$url" 2>&1 | tee -a "$LOG_FILE"
    if [[ $? -ne 0 ]]; then
        echo "Error downloading from: $url" | tee -a "$LOG_FILE"
        exit 1
    fi
    echo "Finished downloading from: $url" | tee -a "$LOG_FILE"
}

# Iterate over the directories and download content
for dir in "${DIRS[@]}"; do
    download_content "$BASE_URL/$dir/"
done

echo "All downloads completed successfully." | tee -a "$LOG_FILE"