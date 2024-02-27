# custom dataset
import os
from pathlib import Path
from torch.utils.data import Dataset
from PIL import Image

# Make function to find classes in target directory
def get_classes(directory: str):

    # Get class names by scanning the target directory
    classes = os.listdir(directory)
      
    # Create a dictionary of index labels (computers prefer numerical rather than string labels)
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx


# Subclass torch.utils.data.Dataset
class CustomDataset_classification(Dataset):
    
    def __init__(self, targ_dir: str, transform=None):
        
        # Get all image paths
        # Common image extensions
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']

        # Get all image files with common extensions
        self.paths = [file for ext in image_extensions for file in Path(targ_dir).rglob(ext)]

        # Setup transforms
        self.transform = transform
        # Create classes and class_to_idx attributes
        self.classes, self.class_to_idx = get_classes(targ_dir)

    # load images
    def load_image(self, index: int):
        "Opens an image via a path and returns it."
        image_path = self.paths[index]
        return Image.open(image_path).convert("RGB")
    
    # Overwrite the __len__() method
    def __len__(self):
        return len(self.paths)
    
    # Overwrite __getitem__() method
    def __getitem__(self, index: int):
        "Returns one sample of data, data and label (X, y)."
        img = self.load_image(index)
        class_name  = self.paths[index].parent.name # expects path in data_folder/class_name/image.jpeg
        class_idx = self.class_to_idx[class_name]

        # Transform if necessary
        if self.transform:
            return self.transform(img), class_idx # return data, label (X, y)
        else:
            return img, class_idx # return data, label (X, y)
