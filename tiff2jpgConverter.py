import os
from PIL import Image, ImageSequence


def main():
    print(f"UNTESTED CODE")
    path_to_tiff: str = r""
    file_name = [file for file in os.listdir(path_to_tiff) if file.endswith("tiff")][0]
    im = Image.open(file_name)

    for i, page in enumerate(ImageSequence.Iterator(im)):
        page.save(f"{i:03}.png")


if __name__ == "__main__":
    main()