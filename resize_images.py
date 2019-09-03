from PIL import Image, ImageFile
import os
import configurations

ImageFile.LOAD_TRUNCATED_IMAGES = True


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
count = 0
for subdir, dirs, files in os.walk(source_path):
    count += 1
    print(count)
    for file in files:
        current_dir = subdir.replace(source_path, dst_path)
        new_file_path = os.path.join(current_dir, file)
        if os.path.exists(new_file_path):
            continue
        os.makedirs(current_dir, exist_ok=True)
        im = Image.open(os.path.join(subdir, file))
        try:
            resized_im = resize(im, 200, os.path.join(subdir, file))
            resized_im.save(new_file_path)
        except OSError:
            im.save(new_file_path)
