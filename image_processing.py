import cv2
import numpy as np


class ImageProcessing:
    def __init__(self, image_path):
        """
        Initialize with the path to the image.

        Parameters:
        image_path (str): Path to the input image.
        """
        self.image = cv2.imread(image_path, 0)
        if self.image is None:
            raise ValueError("Image not found or path is incorrect")

    def sobel_edge_detection(self, ksize=3):
        """
        Apply Sobel edge detection.

        The Sobel operator computes the gradient of the image intensity at each pixel.
        Formula for Sobel operator:
        Gx = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
        Gy = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]

        Gradient magnitude: G = sqrt(Gx^2 + Gy^2)

        Parameters:
        ksize (int): Size of the extended Sobel kernel; must be 1, 3, 5, or 7.

        Returns:
        numpy.ndarray: Image with Sobel edges detected.
        """
        sobel_x = cv2.Sobel(self.image, cv2.CV_64F, 1, 0, ksize=ksize)
        sobel_y = cv2.Sobel(self.image, cv2.CV_64F, 0, 1, ksize=ksize)
        sobel_edges = cv2.magnitude(sobel_x, sobel_y)
        return np.uint8(sobel_edges)

    def log_edge_detection(self, sigma=1.0):
        """
        Apply Laplacian of Gaussian (LoG) edge detection.

        The LoG operator applies a Gaussian filter followed by the Laplacian filter.
        Formula for LoG:
        LoG(x, y) = (x^2 + y^2 - 2*sigma^2) * exp(-(x^2 + y^2) / (2*sigma^2)) / (2*pi*sigma^4)

        Parameters:
        sigma (float): Standard deviation of the Gaussian kernel.

        Returns:
        numpy.ndarray: Image with LoG edges detected.
        """
        ksize = int(6 * sigma + 1)
        if ksize % 2 == 0:
            ksize += 1
        blurred_image = cv2.GaussianBlur(self.image, (ksize, ksize), sigma)
        log_edges = cv2.Laplacian(blurred_image, cv2.CV_64F)
        return np.uint8(log_edges)

    def roberts_edge_detection(self):
        """
        Apply Roberts edge detection.

        The Roberts operator computes the gradient using 2x2 convolution kernels.
        Formula for Roberts operator:
        Gx = [[1, 0], [0, -1]]
        Gy = [[0, 1], [-1, 0]]

        Gradient magnitude: G = sqrt(Gx^2 + Gy^2)

        Returns:
        numpy.ndarray: Image with Roberts edges detected.
        """
        kernel_x = np.array([[1, 0], [0, -1]], dtype=int)
        kernel_y = np.array([[0, 1], [-1, 0]], dtype=int)
        edges_x = cv2.filter2D(self.image, cv2.CV_16S, kernel_x)
        edges_y = cv2.filter2D(self.image, cv2.CV_16S, kernel_y)
        roberts_edges = cv2.addWeighted(cv2.convertScaleAbs(edges_x), 0.5, cv2.convertScaleAbs(edges_y), 0.5, 0)
        return roberts_edges

    def global_thresholding(self):
        """
        Apply global thresholding using Otsu's method.

        Otsu's method chooses the threshold to minimize the intra-class variance.
        Formula for Otsu's thresholding:
        Total variance = (class1_variance * class1_weight) + (class2_variance * class2_weight)

        Returns:
        numpy.ndarray: Binary image after thresholding.
        """
        _, thresholded_image = cv2.threshold(self.image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return thresholded_image
