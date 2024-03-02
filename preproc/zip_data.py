from pathlib import Path
import zipfile

def zip_directory(directory, zip_name):
    with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file in directory.rglob('*'):
            # Write the file to the zip, preserving the directory structure
            zipf.write(file, file.relative_to(directory.parent.parent))

base_path = Path('./data/h1')  # replace with your base directory path
out_path = Path('./data/h1')  # replace with your output directory path
broad_categories = ['train', 'minival', 'calibration']
test_categories = ['oracle', 'eval']
tests = ['test_short', 'test_long']

# Create a single zip file for the broad categories
with zipfile.ZipFile(out_path / 'h1.zip', 'w', zipfile.ZIP_DEFLATED) as zipf:
    for category in broad_categories:
        for file in (base_path / category).rglob('*'):
            zipf.write(file, file.relative_to(base_path.parent))

# Zip up other categories ('oracle', 'eval') for each test type
for test in tests:
    for category in test_categories:
        directory_path = base_path / test / category
        zip_name = out_path / f'{test}_{category}.zip'
        zip_directory(directory_path, zip_name)
