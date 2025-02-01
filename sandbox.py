import numpy as np
import cv2 as cv

import sam2_loader  # Import the module

# Load image
img = cv.imread('Data/sample1.png')

def main():
    while True:
        print("\nMenu:")
        print("1. Load SAM2 Model")
        print("2. Exit")

        choice = input("Enter your choice: ")

        if choice == "1":
            sam2, mask_generator = sam2_loader.load_model()
            image = np.array(img.convert("RGB"))
            masks = mask_generator.generate(image)

        elif choice == "2":
            print("Exiting program.")
            break
        else:
            print("Invalid choice. Please enter 1 or 2.")


if __name__ == "__main__":
    main()
