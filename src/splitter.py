from sklearn.model_selection import train_test_split

def filter_and_split(dataset,min_images=3, test_size=0.1, seed=42):
    """
    Input:
        dataset = {class_name: [images]}

    Output:
        splits = {
            class: {
                "train": [...],
                "test": [...]
            }
        }
    """

    splits={}

    for cls,images in dataset.items():

        # ignore small classes
        if len(images) < min_images:
            print(f"Skipping {cls} ({len(images)} images)")
            continue

        train_imgs, test_imgs=train_test_split(
            images,
            test_size=test_size,
            random_state=seed
        )

        splits[cls]={
            "train":train_imgs,
            "test":test_imgs
        }

    return splits    