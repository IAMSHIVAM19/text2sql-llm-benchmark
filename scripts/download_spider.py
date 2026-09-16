"""Download and unpack the Spider text-to-SQL benchmark dataset.

Supports:
1. Automated Google Drive download via gdown (official Yale LILY release).
2. Direct extraction if spider.zip is placed in data/raw/.
3. Integrity verification of tables.json, dev.json, train.json, and database/ directories.
"""

import argparse
import os
import shutil
import zipfile
from pathlib import Path

SPIDER_GDRIVE_ID = "1403EGqzIDoHMdQF4c9Bkyl7dZLZ5Wt6J"
DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def verify_spider_integrity(spider_dir: Path) -> bool:
    """Verify that required Spider files and database directories exist."""
    required_files = ["dev.json", "tables.json"]
    for f in required_files:
        if not (spider_dir / f).exists():
            return False

    db_dir = spider_dir / "database"
    if not db_dir.exists() or not db_dir.is_dir():
        return False

    # Check that at least a few known dev SQLite databases exist
    known_dbs = ["concert_singer", "pets_1", "car_1", "flight_2", "stadium"]
    for db_name in known_dbs:
        db_file = db_dir / db_name / f"{db_name}.sqlite"
        if not db_file.exists():
            return False

    return True


def download_spider(output_dir: Path, force: bool = False) -> Path:
    """Download and extract Spider dataset to output_dir/spider."""
    output_dir.mkdir(parents=True, exist_ok=True)
    spider_dest = output_dir / "spider"

    if spider_dest.exists() and not force:
        if verify_spider_integrity(spider_dest):
            print(f"Spider dataset already exists and verified at: {spider_dest}")
            return spider_dest
        else:
            print(f"Existing Spider directory at {spider_dest} is incomplete. Re-extracting...")

    zip_path = output_dir / "spider.zip"

    # Step 1: Check if local zip exists, if not download with gdown
    if not zip_path.exists():
        print(f"Downloading Spider dataset (official release) via gdown...")
        download_success = False
        gdrive_ids = [SPIDER_GDRIVE_ID, "1_AckYkinAnHQmRmrH0N0hcLNRW0uuAcv"]
        
        try:
            import gdown
            for gid in gdrive_ids:
                try:
                    print(f"Attempting Google Drive ID: {gid}...")
                    url = f"https://drive.google.com/uc?id={gid}"
                    gdown.download(url, str(zip_path), quiet=False)
                    if zip_path.exists() and zip_path.stat().st_size > 1024 * 1024:
                        download_success = True
                        break
                except Exception as inner_e:
                    print(f"Mirror ID {gid} failed: {inner_e}")
        except ImportError:
            print("gdown is not installed. Run: pip install gdown")

        if not download_success or not zip_path.exists():
            print("\n" + "=" * 65)
            print("[NOTICE] Automated download timed out on local network.")
            print("In Google Colab / Kaggle, gdown runs in seconds with high bandwidth:")
            print(f"  !gdown {SPIDER_GDRIVE_ID} -O {zip_path}")
            print("\nAlternatively, download spider.zip directly from:")
            print("  https://yale-lily.github.io/spider")
            print(f"and place the zip file at: {zip_path.resolve()}")
            print("=" * 65 + "\n")
            raise RuntimeError(f"spider.zip not found at {zip_path}")

    # Step 2: Unzip dataset
    print(f"Extracting {zip_path} to {output_dir}...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(output_dir)

    # Some zip archives extract directly as 'spider/', others might have nested folders
    extracted_candidates = [output_dir / "spider", output_dir / "Spider"]
    found_dir = None
    for cand in extracted_candidates:
        if cand.exists():
            found_dir = cand
            break

    if not found_dir:
        raise FileNotFoundError(f"Could not locate extracted spider directory in {output_dir}")

    if found_dir.name != "spider":
        found_dir.rename(spider_dest)

    if not verify_spider_integrity(spider_dest):
        raise RuntimeError(f"Extracted dataset at {spider_dest} failed integrity verification.")

    print(f"Successfully downloaded and verified Spider dataset at: {spider_dest}")
    return spider_dest


def main():
    parser = argparse.ArgumentParser(description="Download and verify Spider dataset.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Target data directory (default: data/raw)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download and extraction even if directory exists",
    )
    args = parser.parse_args()
    download_spider(args.data_dir, force=args.force)


if __name__ == "__main__":
    main()
