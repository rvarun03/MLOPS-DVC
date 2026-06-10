import os

from data_loader import load_raw_data
from splitter import filter_and_split
from data_writer import write_processed_v1,write_processed_v2

RAW_DIR="data/raw"
PROCESSED_DIR = "data/processed"
V1_DIR = "data/processed_v1"
V2_DIR = "data/processed_v2"


def main():

    # STEP1: LOAD THE DATASET 

    dataset=load_raw_data(RAW_DIR)

    # Step 2: filter + split

    splits = filter_and_split(dataset)

    # write_processed_data(
    #     splits=splits,
    #     raw_dir=RAW_DIR,
    #     processed_dir=PROCESSED_DIR    
    # )

    print("Creating processed_v1...")
    write_processed_v1(splits, RAW_DIR, V1_DIR)

    print("Creating processed_v2...")
    write_processed_v2(splits, RAW_DIR, V2_DIR)

    print("DONE 🚀")
    

if __name__ == "__main__":
    main()    