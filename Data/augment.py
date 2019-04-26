import copy 
from imgaug import augmenters as iaa

def augment(img,augmentationType = "default"):
    """Augments the passed image with the selected augmenter
        
        Args:
            img (np array): images
            augmentationType (str): Defaults to "default". Name of the augmentation setting that shall be used
        
        Returns:
            np array: augmented images (no originals)
    """
    if augmentationType == 'default':
        augmenter = get_default_augmenter()
    else:
        raise ValueError('Augmententation not found.')

    images_aug = augmenter.augment_images(img)  # done by the library
    return images_aug

def get_default_augmenter():
    """ Randomly rotate and flip images
    
    Returns:
        imgaug augmenter: Augmenter that will rotate by 0/90/180/270 degrees and has 50% chance to flip up/down , left/right
    """
    augmenter = iaa.Sequential([
        iaa.geometric.Rot90((1,4)),
        iaa.Flipud(0.5),
        iaa.Fliplr(0.5), # horizontally flip 50% of the images
    ])

    return augmenter
    