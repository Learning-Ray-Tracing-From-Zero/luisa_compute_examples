import cv2
import argparse
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Benchmark: apply Gaussian Blur to an image.")
    parser.add_argument("--input", type=str, required=True, help="Path to the input image.")
    parser.add_argument("--output", type=str, required=True, help="Path to save the output image.")
    parser.add_argument("--radius", type=int, required=True, help="Radius of the Gaussian blur kernel.")
    parser.add_argument("--sigma", type=float, required=True, help="Sigma value for the Gaussian blur.")
    args = parser.parse_args()

    image = cv2.imread(args.input, cv2.IMREAD_COLOR)
    if image is None:
        print(f"Error: Unable to load image from {args.input}")
        return

    cv_blurred = cv2.GaussianBlur(image, (args.radius * 2 + 1, args.radius * 2 + 1), args.sigma)
    cv2.imwrite(args.output, cv_blurred)
    print(f"Output image saved to {args.output}")


if __name__ == "__main__":
    main()
