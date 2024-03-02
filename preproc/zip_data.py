from pathlib import Path
import zipfile

def zip_files(file_patterns, zip_name):
    with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file_pattern in file_patterns:
            for file_path in base_path.rglob(file_pattern):
                zipf.write(file_path, file_path.relative_to(base_path))

# Define base and output paths
base_path = Path('./data/h1')  # replace with your base directory path
out_path = Path('./data/h1')  # replace with your output directory path

# Define file patterns for each category
h1_categories = ['train/*.nwb', 'minival/*.nwb', 'test_short/*calibration.nwb', 'test_long/*calibration.nwb']
oracle_categories = ['test_short/*oracle.nwb', 'test_long/*oracle.nwb']
eval_categories = ['test_short/*eval.nwb', 'test_long/*eval.nwb']

# Zip up h1 categories
h1_zip_path = out_path / 'h1.zip'
zip_files(h1_categories, h1_zip_path)

# Zip up oracle categories
for pattern in oracle_categories:
    test_type = pattern.split('/')[0]  # e.g., 'test_short'
    zip_path = out_path / f'{test_type}_oracle.zip'
    zip_files([pattern], zip_path)

# Zip up eval categories
for pattern in eval_categories:
    test_type = pattern.split('/')[0]  # e.g., 'test_short'
    zip_path = out_path / f'{test_type}_eval.zip'
    zip_files([pattern], zip_path)
