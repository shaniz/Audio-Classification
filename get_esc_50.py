import os
import requests
import zipfile

# Define dataset path
dataset_path = "./ESC-50"
zip_path = "./ESC-50.zip"
download_url = "https://github.com/karolpiczak/ESC-50/archive/master.zip"

# Create the directory if it doesn't exist
if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)

# Download the dataset if it doesn't exist
if not os.path.exists(zip_path):
    print("Downloading ESC-50 dataset...")
    response = requests.get(download_url, stream=True)
    with open(zip_path, "wb") as file:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                file.write(chunk)
    print("Download complete.")
else:
    print("Dataset ZIP file already exists.")

# Extract the dataset
print("Extracting dataset...")
with zipfile.ZipFile(zip_path, "r") as zip_ref:
    zip_ref.extractall("./")

# Move extracted folder to the correct location
extracted_folder = "ESC-50-master"
if os.path.exists(extracted_folder):
    os.rename(extracted_folder, dataset_path)
    print("Dataset extracted and moved to correct location.")
else:
    print("Extraction failed: folder not found.")

# Remove the zip file to save space
os.remove(zip_path)
print("Cleanup complete.")

# Verify the download
print("Dataset contents:")
print(os.listdir(dataset_path))
