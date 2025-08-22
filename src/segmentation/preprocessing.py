import cv2
import numpy as np

def _deskew(image: np.ndarray) -> np.ndarray:
    """Deskew an image using moments."""
    # Convert to grayscale if it's a color image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

    # Threshold to get a binary image. This is crucial for moment calculation.
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Compute moments to find the skew angle
    moments = cv2.moments(thresh)
    if abs(moments['mu02']) < 1e-2:
        # No significant skew detected
        return image

    # Calculate skew based on central moments
    skew = moments['mu11'] / moments['mu02']
    # Create the transformation matrix
    M = np.float32([[1, skew, -0.5 * image.shape[0] * skew], [0, 1, 0]])

    # Apply the affine transformation to deskew the original image (not the thresholded one)
    deskewed = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
    return deskewed

def preprocess_for_hpp(
    image: np.ndarray,
    light_denoising: bool = True,
    enhance_contrast: bool = True,
    noise_reduction: bool = False,
    use_adaptive_threshold: bool = False,
    deskew: bool = True,
) -> np.ndarray:
    """
    Preprocesses an image for HPP segmentation with various options.

    Args:
        image: Input image (expected to be BGR or Grayscale).
        light_denoising: Apply light denoising, good for preserving text features.
        enhance_contrast: Enhance contrast using CLAHE.
        noise_reduction: Apply more aggressive noise reduction (e.g., median blur).
        use_adaptive_threshold: Use adaptive thresholding for binarization, good for uneven lighting.
        deskew: Correct for skew in the image, important for accurate projections.

    Returns:
        A processed binary image (grayscale, 0-255) where text is white and background is black.
    """
    # Deskew the image first, as it affects all subsequent steps
    if deskew:
        image = _deskew(image)

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

    # Denoising options
    if light_denoising:
        # Fast Non-Local Means Denoising is effective for this kind of noise
        gray = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)

    if noise_reduction:
        # Median blur is good for salt-and-pepper noise
        gray = cv2.medianBlur(gray, 3)

    # Contrast Enhancement
    if enhance_contrast:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

    # Binarization and Inversion (so text is white)
    if use_adaptive_threshold:
        # Adaptive thresholding can be better for varying illumination
        processed_image = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )
    else:
        # OTSU's method is a global thresholding technique that works well on bimodal images
        _, processed_image = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    return processed_image


def preprocess_pid_for_hpp(image: np.ndarray) -> np.ndarray:
    """
    P&ID optimized preprocessing for HPP.
    This configuration is tailored for typical P&ID documents, applying light denoising
    and contrast enhancement without aggressive thresholding.
    """
    return preprocess_for_hpp(
        image,
        light_denoising=True,
        enhance_contrast=True,
        noise_reduction=False,
        use_adaptive_threshold=False,
        deskew=True
    )


def enhanced_preprocess_for_hpp(image: np.ndarray) -> np.ndarray:
    """
    Advanced preprocessing for HPP.
    This uses a more aggressive approach suitable for lower quality images,
    including stronger noise reduction and adaptive thresholding.
    """
    return preprocess_for_hpp(
        image,
        light_denoising=True,
        enhance_contrast=True,
        noise_reduction=True,
        use_adaptive_threshold=True,
        deskew=True
    )
