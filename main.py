import os
import matplotlib.pyplot as plt
from image_processing import ImageProcessing


def plot_images(images, titles, output_dir, cols=3):
    """
    Plot a list of images with corresponding titles.

    Parameters:
    images (list of numpy.ndarray): List of images to plot.
    titles (list of str): List of titles for the images.
    output_dir (str): Directory to save the plotted images.
    cols (int): Number of columns in the plot grid.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    rows = (len(images) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    if rows == 1:
        axes = [axes]
    axes = [ax for sublist in axes for ax in sublist]  # Flatten the list of axes

    for i, (img, title) in enumerate(zip(images, titles)):
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(title)
        axes[i].axis('off')
        plt.imsave(os.path.join(output_dir, f"{title}.png"), img, cmap='gray')

    for ax in axes[len(images):]:
        ax.axis('off')

    plt.tight_layout()
    plt.show()


def main():
    # List of image paths
    image_paths = [
        'data/example2.jpg',  # Bacteria
        'data/lena.jpg',  # Lena
        'data/fingerprint.jpg'  # Fingerprint
    ]

    # Titles for the images
    titles = ['Bacteria', 'Lena', 'Fingerprint']

    # Initialize lists to store images and their processed results
    original_images = []
    sobel_images = []
    log_images = []
    roberts_images = []
    thresholded_images = []

    # Output directory for saving results
    output_dir = 'output'

    # Process each image
    for image_path, title in zip(image_paths, titles):
        processor = ImageProcessing(image_path)

        original_images.append(processor.image)
        sobel_images.append(processor.sobel_edge_detection(ksize=3))  # Adjust kernel size here for tuning
        log_images.append(processor.log_edge_detection(sigma=1.0))  # Adjust sigma here for tuning
        roberts_images.append(processor.roberts_edge_detection())
        thresholded_images.append(processor.global_thresholding())

    # Plot original and processed images
    for i, title in enumerate(titles):
        plot_images(
            [original_images[i], sobel_images[i], log_images[i], roberts_images[i]],
            [title, f'{title} - Sobel', f'{title} - LoG', f'{title} - Roberts'],
            output_dir
        )

    # Plot thresholded images
    plot_images(thresholded_images, [f'{title} - Thresholding' for title in titles], output_dir)


if __name__ == "__main__":
    main()
