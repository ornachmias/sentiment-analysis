from PIL import Image
import os

import configurations


def resize(im, desired_size, path):
    try:
        old_size = im.size
        ratio = float(desired_size) / max(old_size)
        new_size = tuple([int(x * ratio) for x in old_size])
        im = im.resize(new_size, Image.ANTIALIAS)
        new_im = Image.new("RGB", (desired_size, desired_size))
        new_im.paste(im, ((desired_size - new_size[0]) // 2,
                          (desired_size - new_size[1]) // 2))
    except ValueError:
        print("Failed to process image. Image path: " + path)
        ratio = float(desired_size) / min(old_size)
        new_size = tuple([int(x * ratio) for x in old_size])
        im = im.resize(new_size, Image.ANTIALIAS)
        new_im = im.crop((0, 0, desired_size, desired_size))

    return new_im


source_path = "D:\\b-t4sa_imgs\\data"
dst_path = "./Data_resized"

for subdir, dirs, files in os.walk(source_path):
    for file in files:
        current_dir = subdir.replace(source_path, dst_path)
        new_file_path = os.path.join(current_dir, file)
        os.makedirs(current_dir, exist_ok=True)
        im = Image.open(os.path.join(subdir, file))
        resized_im = resize(im, configurations.image_size, os.path.join(subdir, file))
        resized_im.save(new_file_path)