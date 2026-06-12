import os 
import numpy as np
from features_hog import extract_hog_features

INPUT_DIR = "data/processed_v1"
OUTPUT_DIR = "data/features_v1"

def extract_split(split_name):
    X=[]
    y=[]

    split_path=os.path.join(INPUT_DIR,split_name)
    
    for label in os.listdir(split_path):

        class_path=os.path.join(split_path,label)

        if not os.path.isdir(class_path):
            continue

        for image_name in os.listdir(class_path):
            
            image_path=os.path.join(class_path,image_name)

            try:
                features=extract_hog_features(image_path)
                X.append(features)
                y.append(label)
            except Exception as e:
                print(f"Skipping {image_path}: {e}")

    return np.array(X), np.array(y)

def main():

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Processing TRAIN images...")
    X_train, y_train = extract_split("train")

    print("Processing TEST images...")
    X_test, y_test = extract_split("test")

    np.save(os.path.join(OUTPUT_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(OUTPUT_DIR, "y_train.npy"), y_train)

    np.save(os.path.join(OUTPUT_DIR, "X_test.npy"), X_test)
    np.save(os.path.join(OUTPUT_DIR, "y_test.npy"), y_test)

    print("Features saved successfully")

if __name__ == "__main__":
    main()    
    
                