#!/usr/bin/env python3
"""
Skript pro automatické vytváření dataset dataframes pro všechny synchronizované datasety.
"""

import subprocess
from pathlib import Path
import sys

def main():
    # Cesta k synchronizovaným datasetům
    synchronized_data_dir = Path(r"C:\Users\wei0068\Desktop\IT4I\synchronized_data")
    
    if not synchronized_data_dir.exists():
        print(f"❌ Složka {synchronized_data_dir} neexistuje!")
        return
    
    # Najdeme všechny datasety
    datasets = [d for d in synchronized_data_dir.iterdir() if d.is_dir()]
    
    print(f"Nalezeno {len(datasets)} datasetů pro zpracování:")
    for dataset in datasets:
        print(f"  - {dataset.name}")
    print()
    
    # Zpracování každého datasetu
    successful = 0
    failed = 0
    
    for dataset in datasets:
        dataset_name = dataset.name
        output_file = f"{dataset_name}_dataset_dataframe.parquet"
        
        print(f"📊 Zpracovávám {dataset_name}...")
        
        try:
            # Spustíme příkaz pro vytvoření dataframe
            result = subprocess.run([
                sys.executable, "preprocessing.py", "create-dataset-dataframe",
                str(dataset), "--output_path", output_file
            ], capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                # Zkontrolujeme, zda byl soubor vytvořen
                if Path(output_file).exists():
                    file_size = Path(output_file).stat().st_size / 1024  # KB
                    print(f"  ✅ Úspěšně vytvořen: {output_file} ({file_size:.1f} KB)")
                    successful += 1
                else:
                    print(f"  ❌ Soubor nebyl vytvořen: {output_file}")
                    failed += 1
            else:
                print(f"  ❌ Chyba při zpracování: {result.stderr}")
                failed += 1
                
        except subprocess.TimeoutExpired:
            print(f"  ⏰ Timeout při zpracování {dataset_name}")
            failed += 1
        except Exception as e:
            print(f"  ❌ Neočekávaná chyba: {e}")
            failed += 1
    
    # Shrnutí
    print()
    print("=" * 50)
    print("SHRNUTÍ ZPRACOVÁNÍ")
    print("=" * 50)
    print(f"✅ Úspěšně zpracováno: {successful}")
    print(f"❌ Neúspěšně zpracováno: {failed}")
    print(f"📊 Celkem datasetů: {len(datasets)}")
    
    if successful > 0:
        print(f"\n📁 Vytvořené parquet soubory:")
        for parquet_file in Path(".").glob("*_dataset_dataframe.parquet"):
            file_size = parquet_file.stat().st_size / 1024  # KB
            print(f"  - {parquet_file.name} ({file_size:.1f} KB)")

if __name__ == "__main__":
    main()
