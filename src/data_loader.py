import os

IMG_EXTS = (".jpg", ".jpeg", ".png")

def load_raw_data(raw_dir):
    """
    Reads dataset folder and returns:
    {class_name: [image1, image2, ...]}
    """

    dataset={}

    for cls in os.listdir(raw_dir):
        cls_path=os.path.join(raw_dir,cls)
        if not os.path.isdir(cls_path):
            continue
        images=[
            f for f in os.listdir(cls_path)
            if f.lower().endswith(IMG_EXTS)
        ]

        dataset[cls]= images

    return dataset

