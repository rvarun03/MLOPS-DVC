import os
import shutil

def create_dir(path):
    os.makedirs(path,exist_ok=True)

def write_processed_data(
        splits,
        raw_dir,
        processed_dir
):
    
    """
    Copy train/test images into processed folder.
    """

    train_root=os.path.join(processed_dir,"train")
    test_root=os.path.join(processed_dir,"test")

    create_dir(train_root)
    create_dir(test_root)

    for classname, split_data in splits.items():

        train_class_dir=os.path.join(
            train_root,
            classname
        )

        test_class_dir=os.path.join(
            test_root,
            classname
        )

        create_dir(train_class_dir)
        create_dir(test_class_dir)

        for image_name in split_data["train"]:

            src=os.path.join(
                raw_dir,
                classname,
                image_name
            )

            dst=os.path.join(
                train_class_dir,
                image_name
            )

            shutil.copy2(src, dst)

        for image_name in split_data["test"]:

            src = os.path.join(
                raw_dir,
                classname,
                image_name
            )

            dst = os.path.join(
                test_class_dir,
                image_name
            )

            shutil.copy2(src, dst)    



