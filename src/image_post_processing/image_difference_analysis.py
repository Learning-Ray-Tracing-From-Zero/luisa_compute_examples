import cv2
import argparse
from skimage.metrics import mean_squared_error as MSE
from skimage.metrics import structural_similarity as SSIM


"""
Mean Squared Error(MSE)
The average square of the difference in pixel values between two images
The smaller the value, the more similar the image
Sensitive to absolute differences in pixel values, but insensitive to structural information
"""

"""
Structural Similarity Index(SSIM)
A metric for measuring the structural similarity between two images,
  based on the comparison of brightness, contrast, and structure
The value range is between [-1, 1],
  and the closer the value is to 1, the more similar the image is
Sensitive to structural information, it can better reflect human visual perception
"""

def compare_images(image1, image2, diff_path=None):
    if len(image1.shape) == 3:
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    if len(image2.shape) == 3:
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    mse_value = MSE(image1, image2)
    ssim_value, _ = SSIM(image1, image2, win_size=7, data_range=255, full=True)
    print(f"MSE: {mse_value}")
    print(f"SSIM: {ssim_value}")
    print(f"1-SSIM: {format(1 - ssim_value, '.10f')}")

    if diff_path:
        diff = cv2.absdiff(image1, image2)
        cv2.imwrite(diff_path, diff)
        print(f"Difference image saved to {diff_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare two images using MSE and SSIM.")
    parser.add_argument("--image1", type=str, required=True, help="Path to the first image.")
    parser.add_argument("--image2", type=str, required=True, help="Path to the second image.")
    parser.add_argument("--diff_path", type=str, help="Path to save the difference image.")
    args = parser.parse_args()

    image1 = cv2.imread(args.image1)
    image2 = cv2.imread(args.image2)
    if image1 is None or image2 is None:
        print("Error: Unable to load images.")
        return

    if image1.shape != image2.shape:
        print("Resizing images to match dimensions...")
        image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))

    compare_images(image1, image2, args.diff_path)


if __name__ == "__main__":
    main()
