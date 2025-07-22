#!/usr/bin/env python3
"""
Skript pro automatické generování job files pro všechny parquet datasety.
"""

import subprocess
from pathlib import Path
import sys

def main():
    # Najdeme všechny parquet soubory
    parquet_files = list(Path(".").glob("*_dataset_dataframe.parquet"))
    
    if not parquet_files:
        print("❌ Nebyl nalezen žádný parquet soubor!")
        print("Ujistěte se, že jste nejdřív spustili create_all_dataframes.py")
        return
    
    print(f"Nalezeno {len(parquet_files)} parquet souborů pro zpracování:")
    for file in sorted(parquet_files):
        print(f"  - {file.name}")
    print()
    
    # Kampaně pro zpracování
    campaigns = ["01", "02"]
    
    # Zpracování každého parquet souboru
    successful = 0
    failed = 0
    
    for parquet_file in sorted(parquet_files):
        dataset_name = parquet_file.stem.replace("_dataset_dataframe", "")
        
        print(f"📊 Zpracovávám {dataset_name}...")
        
        for campaign in campaigns:
            try:
                print(f"  🔄 Kampaň {campaign}...")
                
                # Spustíme příkaz pro generování job file
                result = subprocess.run([
                    sys.executable, "run_fusion.py", "generate-jobfiles",
                    "--parquet-file", str(parquet_file),
                    "--campaign-number", campaign,
                    "--output-dir", "job_files"
                ], capture_output=True, text=True, timeout=60)
                
                if result.returncode == 0:
                    job_file = f"job_files/{dataset_name}_{campaign}_job_file.txt"
                    if Path(job_file).exists():
                        print(f"    ✅ Job file vytvořen: {dataset_name}_{campaign}_job_file.txt")
                        successful += 1
                    else:
                        print(f"    ❌ Job file nebyl vytvořen pro kampaň {campaign}")
                        failed += 1
                else:
                    print(f"    ❌ Chyba při generování job file: {result.stderr}")
                    failed += 1
                    
            except subprocess.TimeoutExpired:
                print(f"    ⏰ Timeout při zpracování kampaně {campaign}")
                failed += 1
            except Exception as e:
                print(f"    ❌ Neočekávaná chyba: {e}")
                failed += 1
    
    # Shrnutí
    print()
    print("=" * 50)
    print("SHRNUTÍ GENEROVÁNÍ JOB FILES")
    print("=" * 50)
    print(f"✅ Úspěšně vytvořeno job files: {successful}")
    print(f"❌ Neúspěšně vytvořeno: {failed}")
    print(f"📊 Celkem parquet souborů: {len(parquet_files)}")
    print(f"🎯 Celkem kampaní: {len(campaigns)}")
    
    # Zobrazení vytvořených job files
    job_files_dir = Path("job_files")
    if job_files_dir.exists():
        job_files = list(job_files_dir.glob("*.txt"))
        if job_files:
            print(f"\n📁 Vytvořené job files ({len(job_files)}):")
            for job_file in sorted(job_files):
                file_size = job_file.stat().st_size
                print(f"  - {job_file.name} ({file_size} bytes)")
                
                # Zobrazíme prvních několik řádků pro kontrolu
                with open(job_file, 'r') as f:
                    lines = f.readlines()[:3]  # První 3 řádky
                    print(f"    Preview: {len(f.readlines()) + 3} řádků celkem")
                    for line in lines:
                        print(f"      {line.strip()}")
                    if len(lines) >= 3:
                        print(f"      ...")
                print()

if __name__ == "__main__":
    main()
