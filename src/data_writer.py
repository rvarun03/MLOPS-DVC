import os
import shutil
from PIL import Image
import random

# -------------------------
# Utility
# -------------------------

def create_dir(path):
    os.makedirs(path, exist_ok=True)

def resize_image(src, dst, size=(224, 224)):
    img = Image.open(src)
    img = img.convert("RGB")
    img = img.resize(size)
    img.save(dst)    

def augment_image(img):
    # random horizontal flip
    if random.random() > 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)

    # random slight rotation
    if random.random() > 0.5:
        img = img.rotate(10)

    return img

# -------------------------
# VERSION 1: CLEAN DATASET
# -------------------------

def write_processed_v1(splits,raw_dir,processed_dir):
    """
    Clean dataset:
    - train/test split
    - resize only
    """
    train_root = os.path.join(processed_dir, "train")
    test_root = os.path.join(processed_dir, "test")

    create_dir(train_root)
    create_dir(test_root)

    for classname, split_data in splits.items():

        train_class_dir= os.path.join(train_root,classname)
        test_class_dir= os.path.join(test_root, classname)

        create_dir(train_class_dir)
        create_dir(test_class_dir)

        # TRAIN
        for image_name in split_data["train"]:
            src = os.path.join(raw_dir, classname, image_name)
            dst = os.path.join(train_class_dir, image_name)
            resize_image(src, dst)

        # TEST
        for image_name in split_data["test"]:
            src = os.path.join(raw_dir, classname, image_name)
            dst = os.path.join(test_class_dir, image_name)
            resize_image(src, dst)

# -------------------------
# VERSION 2: AUGMENTED DATASET
# -------------------------
def write_processed_v2(splits, raw_dir, processed_dir):
    """
    Augmented dataset:
    - train: augmentation + resize
    - test: only resize
    """

    train_root = os.path.join(processed_dir, "train")
    test_root = os.path.join(processed_dir, "test")

    create_dir(train_root)
    create_dir(test_root)

    for classname, split_data in splits.items():

        train_class_dir = os.path.join(train_root, classname)
        test_class_dir = os.path.join(test_root, classname)

        create_dir(train_class_dir)
        create_dir(test_class_dir)

        # TRAIN (AUGMENTED)
        for image_name in split_data["train"]:

            src = os.path.join(raw_dir, classname, image_name)
            dst = os.path.join(train_class_dir, image_name)

            img = Image.open(src).convert("RGB")
            img = augment_image(img)
            img = img.resize((224, 224))
            img.save(dst)

        # TEST (NO AUGMENTATION)
        for image_name in split_data["test"]:

            src = os.path.join(raw_dir, classname, image_name)
            dst = os.path.join(test_class_dir, image_name)

            img = Image.open(src).convert("RGB")
            img = img.resize((224, 224))
            img.save(dst)            


# def create_dir(path):
#     os.makedirs(path,exist_ok=True)

# def write_processed_data(
#         splits,
#         raw_dir,
#         processed_dir
# ):
    
#     """
#     Copy train/test images into processed folder.
#     """

#     train_root=os.path.join(processed_dir,"train")
#     test_root=os.path.join(processed_dir,"test")

#     create_dir(train_root)
#     create_dir(test_root)

#     for classname, split_data in splits.items():

#         train_class_dir=os.path.join(
#             train_root,
#             classname
#         )

#         test_class_dir=os.path.join(
#             test_root,
#             classname
#         )

#         create_dir(train_class_dir)
#         create_dir(test_class_dir)

#         for image_name in split_data["train"]:

#             src=os.path.join(
#                 raw_dir,
#                 classname,
#                 image_name
#             )

#             dst=os.path.join(
#                 train_class_dir,
#                 image_name
#             )

#             shutil.copy2(src, dst)

#         for image_name in split_data["test"]:

#             src = os.path.join(
#                 raw_dir,
#                 classname,
#                 image_name
#             )

#             dst = os.path.join(
#                 test_class_dir,
#                 image_name
#             )

#             shutil.copy2(src, dst)    



