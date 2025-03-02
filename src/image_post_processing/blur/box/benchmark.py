import cv2
import argparse


def main():
    parser = argparse.ArgumentParser(description="Apply Box Blur to an image.")
    parser.add_argument("--input", type=str, required=True, help="Path to the input image.")
    parser.add_argument("--output", type=str, required=True, help="Path to save the output image.")
    parser.add_argument("--blur_radius", type=int, required=True, help="Radius of the Box blur kernel.")
    args = parser.parse_args()

    print(f"Opencv version: {cv2.__version__}")
    image = cv2.imread(args.input, cv2.IMREAD_COLOR)
    if image is None:
        print(f"Error: Unable to load image from {args.input}")
        return

    # box blur
    blur_kernel_size = 2 * args.blur_radius + 1
    blurred_image = cv2.boxFilter(
        image,
        -1,
        (blur_kernel_size, blur_kernel_size),
        borderType = cv2.BORDER_REFLECT_101,
        normalize = True
    )
    cv2.imwrite(args.output, blurred_image)
    print(f"Output image saved to {args.output}")


if __name__ == "__main__":
    main()
