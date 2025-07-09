import os
import zipfile

def extract_missing_zips(dataset_dir='dataset'):
    for entry in os.listdir(dataset_dir):
        if entry.endswith('.zip'):
            zip_path = os.path.join(dataset_dir, entry)
            folder_name = entry[:-4]  # remove '.zip'
            folder_path = os.path.join(dataset_dir, folder_name)

            if not os.path.isdir(folder_path):
                print(f"Extracting {entry} to {folder_name}/...")
                try:
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        zip_ref.extractall(folder_path)
                    print(f"✅ Extracted: {zip_path}")
                except Exception as e:
                    print(f"❌ Failed to extract {zip_path}: {e}")
            else:
                print(f"Skipping {entry} — folder already exists.")

if __name__ == "__main__":
    extract_missing_zips()
