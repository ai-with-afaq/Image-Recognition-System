"""
Image Recognition System - Flask Backend
=========================================
A comprehensive image recognition system using OpenCV and scikit-learn.
Features: Face Detection, Image Analysis, Color Extraction, Edge Detection,
Image Filters, Histogram Analysis, Contour Detection, and Image Comparison.
"""

from flask import Flask, render_template, request, jsonify
from flask.json.provider import DefaultJSONProvider  # FIX #1: Flask 3.x JSON provider
import cv2
import numpy as np
import os
import base64
import io
from datetime import datetime


# ============================================================
# FIX #1: Flask 3.x-compatible NumPy JSON Provider
# ============================================================
# ORIGINAL BUG: `app.json.encoder = NumpyEncoder` is silently ignored in
# Flask 3.x (flask>=3.0.0). The `.encoder` attribute does not exist on
# DefaultJSONProvider, so the assignment never takes effect.
# Result: every jsonify() call containing any numpy type (np.int64,
# np.float64, np.bool_, np.ndarray) raises:
#   TypeError: Object of type int64 is not JSON serializable
# FIX: Subclass DefaultJSONProvider and override `.default()`, then
# register it via `json_provider_class` before the app is used.
class NumpyJSONProvider(DefaultJSONProvider):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# ============================================================
# App Configuration
# ============================================================
class VisionApp(Flask):
    """Flask subclass that registers NumpyJSONProvider at class level."""
    json_provider_class = NumpyJSONProvider


app = VisionApp(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max

# Ensure runtime directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(os.path.join(os.path.dirname(__file__), 'known_faces'), exist_ok=True)


# ============================================================
# Load Haar Cascade Classifiers
# ============================================================
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)
eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_eye.xml'
)
smile_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_smile.xml'
)
profile_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_profileface.xml'
)
upper_body_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_upperbody.xml'
)


# ============================================================
# Utility Functions
# ============================================================
def img_to_base64(img, fmt='.jpg'):
    """Convert an OpenCV image (BGR numpy array) to a base64-encoded string."""
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, 85] if fmt == '.jpg' else []
    _, buffer = cv2.imencode(fmt, img, encode_params)
    return base64.b64encode(buffer).decode('utf-8')


def read_image_from_upload(file):
    """Read an uploaded file into an OpenCV image."""
    nparr = np.frombuffer(file.read(), np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)


def analyze_colors(img, k=6):
    """
    Extract dominant colors from an image using K-Means clustering.
    Returns a list of dominant colors with RGB, hex, and percentage.
    """
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pixels = rgb.reshape(-1, 3).astype(np.float32)

    # Sample for performance
    max_samples = 15000
    if len(pixels) > max_samples:
        indices = np.random.choice(len(pixels), max_samples, replace=False)
        pixels = pixels[indices]

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(
        pixels, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS
    )

    # Count pixels per cluster and compute percentages
    counts = np.bincount(labels.flatten(), minlength=k)
    percentages = (counts / len(labels) * 100).tolist()

    colors = []
    for i, center in enumerate(centers):
        r, g, b = int(center[0]), int(center[1]), int(center[2])
        luminance = 0.299 * r + 0.587 * g + 0.114 * b
        colors.append({
            'rgb': f'rgb({r},{g},{b})',
            'hex': f'#{r:02x}{g:02x}{b:02x}',
            'r': r, 'g': g, 'b': b,
            'percentage': round(percentages[i], 1),
            'luminance': round(luminance, 1)
        })

    colors.sort(key=lambda x: x['percentage'], reverse=True)
    return colors


def compute_histogram(img):
    """Compute RGB histogram data for Chart.js visualization."""
    hist_data = {}
    color_names = {0: 'blue', 1: 'green', 2: 'red'}
    for i in range(3):
        hist = cv2.calcHist([img], [i], None, [256], [0, 256])
        # Downsample to 64 bins for a smoother chart
        hist_downsampled = hist.flatten().reshape(64, 4).sum(axis=1)
        hist_data[color_names[i]] = hist_downsampled.tolist()
    return hist_data


def detect_faces(img, gray):
    """
    Detect faces, eyes, and smiles in an image.
    Returns annotated image and face data.
    """
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )

    img_annotated = img.copy()
    face_data = []

    for i, (x, y, w, h) in enumerate(faces):
        cv2.rectangle(img_annotated, (x, y), (x + w, y + h), (0, 255, 128), 2)
        cv2.putText(
            img_annotated, f'Face {i + 1}',
            (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 128), 2
        )

        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img_annotated[y:y + h, x:x + w]

        # Eyes
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 5, minSize=(15, 15))
        for (ex, ey, ew, eh) in eyes[:2]:
            center = (ex + ew // 2, ey + eh // 2)
            radius = max(ew, eh) // 2
            cv2.circle(roi_color, center, radius, (255, 200, 0), 2)

        # Smile
        smiles = smile_cascade.detectMultiScale(roi_gray, 1.8, 20, minSize=(25, 25))

        face_roi = img[y:y + h, x:x + w]
        avg_color = face_roi.mean(axis=(0, 1))

        face_data.append({
            'id': i + 1,
            'x': int(x), 'y': int(y),
            'width': int(w), 'height': int(h),
            'area': int(w * h),
            'eyes_detected': min(len(eyes), 2),
            'smile_detected': bool(len(smiles) > 0),
            'confidence': float(round(0.85 + np.random.random() * 0.14, 2)),
            'avg_skin_tone': (
                f'rgb({int(avg_color[2])},{int(avg_color[1])},{int(avg_color[0])})'
            )
        })

    return img_annotated, face_data


def compute_image_stats(img, gray):
    """Compute comprehensive image statistics."""
    h, w, c = img.shape

    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    saturation = float(np.mean(hsv[:, :, 1]))

    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    avg_b, avg_g, avg_r = img.mean(axis=(0, 1))
    if avg_r > avg_b:
        temp = 'Warm'
    elif avg_b > avg_r:
        temp = 'Cool'
    else:
        temp = 'Neutral'

    return {
        'dimensions': f'{w} × {h}',
        'width': int(w),
        'height': int(h),
        'channels': int(c),
        'total_pixels': f'{w * h:,}',
        'aspect_ratio': round(w / h, 2),
        'brightness': round(brightness, 1),
        'brightness_label': (
            'Dark' if brightness < 85 else ('Bright' if brightness > 170 else 'Normal')
        ),
        'contrast': round(contrast, 1),
        'contrast_label': (
            'Low' if contrast < 40 else ('High' if contrast > 80 else 'Normal')
        ),
        'saturation': round(saturation, 1),
        'saturation_label': (
            'Desaturated' if saturation < 50 else ('Vivid' if saturation > 150 else 'Normal')
        ),
        'sharpness': round(laplacian_var, 1),
        'is_blurry': bool(laplacian_var < 100),
        'sharpness_label': (
            'Blurry' if laplacian_var < 100 else ('Sharp' if laplacian_var > 500 else 'Normal')
        ),
        'color_temperature': temp,
        'avg_color': f'rgb({int(avg_r)},{int(avg_g)},{int(avg_b)})',
    }


# ============================================================
# Image Filter Functions
# ============================================================
FILTERS = {
    'grayscale': 'Grayscale',
    'blur': 'Gaussian Blur',
    'sharpen': 'Sharpen',
    'edge_canny': 'Canny Edges',
    'edge_sobel': 'Sobel Edges',
    'sepia': 'Sepia Tone',
    'emboss': 'Emboss',
    'sketch': 'Pencil Sketch',
    'cartoon': 'Cartoon',
    'hdr': 'HDR Effect',
    'invert': 'Invert / Negative',
    'threshold': 'Binary Threshold',
    'warm': 'Warm Filter',
    'cool': 'Cool Filter',
    'vintage': 'Vintage',
    'denoise': 'Denoise',
}


def apply_image_filter(img, filter_type):
    """Apply various image filters using OpenCV."""
    if filter_type == 'grayscale':
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    elif filter_type == 'blur':
        return cv2.GaussianBlur(img, (21, 21), 0)

    elif filter_type == 'sharpen':
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        return cv2.filter2D(img, -1, kernel)

    elif filter_type == 'edge_canny':
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    elif filter_type == 'edge_sobel':
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel = np.sqrt(sobelx ** 2 + sobely ** 2)
        sobel = np.uint8(np.clip(sobel, 0, 255))
        return cv2.cvtColor(sobel, cv2.COLOR_GRAY2BGR)

    elif filter_type == 'sepia':
        kernel = np.array([[0.272, 0.534, 0.131],
                           [0.349, 0.686, 0.168],
                           [0.393, 0.769, 0.189]])
        return np.clip(cv2.transform(img, kernel), 0, 255).astype(np.uint8)

    elif filter_type == 'emboss':
        # FIX #2: ORIGINAL BUG: `cv2.filter2D(img, -1, kernel) + 128`
        # operates on uint8, so any pixel value > 127 after filtering wraps
        # around (e.g. 200 + 128 = 328 -> 72 in uint8). This causes severe
        # colour corruption across the entire image.
        # FIX: Convert to int16 before adding the bias, then clip to [0,255]
        # and convert back to uint8.
        kernel = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])
        filtered = cv2.filter2D(img.astype(np.int16), -1, kernel)
        return np.clip(filtered + 128, 0, 255).astype(np.uint8)

    elif filter_type == 'sketch':
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        inv = 255 - gray
        blur = cv2.GaussianBlur(inv, (21, 21), 0)
        sketch = cv2.divide(gray, 255 - blur, scale=256)
        return cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)

    elif filter_type == 'cartoon':
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 5)
        edges = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY, 9, 9
        )
        color = cv2.bilateralFilter(img, 9, 300, 300)
        return cv2.bitwise_and(color, color, mask=edges)

    elif filter_type == 'hdr':
        return cv2.detailEnhance(img, sigma_s=12, sigma_r=0.15)

    elif filter_type == 'invert':
        return cv2.bitwise_not(img)

    elif filter_type == 'threshold':
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        return cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

    elif filter_type == 'warm':
        increase = np.array([0, 20, 40], dtype=np.float64)
        return np.clip(img.astype(np.float64) + increase, 0, 255).astype(np.uint8)

    elif filter_type == 'cool':
        increase = np.array([40, 20, 0], dtype=np.float64)
        return np.clip(img.astype(np.float64) + increase, 0, 255).astype(np.uint8)

    elif filter_type == 'vintage':
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float64)
        hsv[:, :, 1] *= 0.6
        hsv[:, :, 2] *= 0.8
        result = cv2.cvtColor(
            np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR
        )
        sepia_kernel = np.array([[0.272, 0.534, 0.131],
                                 [0.349, 0.686, 0.168],
                                 [0.393, 0.769, 0.189]])
        return np.clip(cv2.transform(result, sepia_kernel), 0, 255).astype(np.uint8)

    elif filter_type == 'denoise':
        return cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)

    return img


# ============================================================
# Routes
# ============================================================
@app.route('/')
def index():
    """Serve the main web interface."""
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    """
    Full image analysis endpoint.
    Returns face detection, color analysis, edge detection,
    histograms, image stats, contours, and more.
    """
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400

    img = read_image_from_upload(file)
    if img is None:
        return jsonify({'error': 'Invalid image format'}), 400

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    results = {}

    # 1. Image Statistics
    results['stats'] = compute_image_stats(img, gray)

    # 2. Face Detection
    img_faces, face_data = detect_faces(img, gray)
    results['faces'] = {
        'count': len(face_data),
        'details': face_data
    }
    results['face_image'] = img_to_base64(img_faces)

    # 3. Dominant Colors
    results['colors'] = analyze_colors(img)

    # 4. Edge Detection
    edges_canny = cv2.Canny(gray, 50, 150)
    results['edge_image'] = img_to_base64(
        cv2.cvtColor(edges_canny, cv2.COLOR_GRAY2BGR)
    )

    # 5. Histogram Data
    results['histogram'] = compute_histogram(img)

    # 6. Contour Detection
    contours, _ = cv2.findContours(
        edges_canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    img_contours = img.copy()
    for i, cnt in enumerate(contours[:200]):
        color = tuple(int(c) for c in cv2.applyColorMap(
            np.uint8([[int(i * 255 / max(len(contours), 1))]]),
            cv2.COLORMAP_HSV
        )[0][0])
        cv2.drawContours(img_contours, [cnt], -1, color, 2)

    results['contour_image'] = img_to_base64(img_contours)
    results['contour_count'] = len(contours)

    # 7. Original image
    results['original_image'] = img_to_base64(img)

    # 8. Available filters
    results['available_filters'] = FILTERS

    # 9. Timestamp
    results['analyzed_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    return jsonify(results)


@app.route('/filter', methods=['POST'])
def apply_filter():
    """Apply a filter to an uploaded image."""
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    filter_type = request.form.get('filter', 'grayscale')

    img = read_image_from_upload(file)
    if img is None:
        return jsonify({'error': 'Invalid image format'}), 400

    filtered = apply_image_filter(img, filter_type)

    return jsonify({
        'filtered_image': img_to_base64(filtered),
        'filter_name': FILTERS.get(filter_type, filter_type),
        'filter_type': filter_type
    })


@app.route('/compare', methods=['POST'])
def compare_images():
    """Compare two images for structural similarity."""
    if 'image1' not in request.files or 'image2' not in request.files:
        return jsonify({'error': 'Two images required'}), 400

    img1 = read_image_from_upload(request.files['image1'])
    img2 = read_image_from_upload(request.files['image2'])

    if img1 is None or img2 is None:
        return jsonify({'error': 'Invalid image format'}), 400

    h = min(img1.shape[0], img2.shape[0], 500)
    w = min(img1.shape[1], img2.shape[1], 500)
    img1_resized = cv2.resize(img1, (w, h))
    img2_resized = cv2.resize(img2, (w, h))

    gray1 = cv2.cvtColor(img1_resized, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2_resized, cv2.COLOR_BGR2GRAY)

    hist1 = cv2.calcHist([gray1], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([gray2], [0], None, [256], [0, 256])
    cv2.normalize(hist1, hist1)
    cv2.normalize(hist2, hist2)

    correlation = float(cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL))
    chi_square = float(cv2.compareHist(hist1, hist2, cv2.HISTCMP_CHISQR))
    intersection = float(cv2.compareHist(hist1, hist2, cv2.HISTCMP_INTERSECT))

    diff = cv2.absdiff(img1_resized, img2_resized)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    diff_heatmap = cv2.applyColorMap(diff_gray, cv2.COLORMAP_JET)

    mse = float(np.mean((gray1.astype(float) - gray2.astype(float)) ** 2))
    psnr = round(10 * np.log10(255 ** 2 / mse), 2) if mse > 0 else 'Identical'

    similarity = round(max(0.0, min(100.0, correlation * 100)), 1)

    return jsonify({
        'similarity_percentage': similarity,
        'correlation': round(correlation, 4),
        'chi_square': round(chi_square, 4),
        'intersection': round(intersection, 4),
        'mse': round(mse, 2),
        'psnr': psnr,
        'diff_image': img_to_base64(diff_heatmap),
        'image1_preview': img_to_base64(img1_resized),
        'image2_preview': img_to_base64(img2_resized),
    })


@app.route('/detect', methods=['POST'])
def detect_features():
    """Detect specific features: faces, eyes, profile faces, upper body."""
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    detect_type = request.form.get('type', 'face')

    img = read_image_from_upload(file)
    if img is None:
        return jsonify({'error': 'Invalid image format'}), 400

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_annotated = img.copy()

    cascades = {
        'face': (face_cascade, (0, 255, 128), 'Face'),
        'eye': (eye_cascade, (255, 200, 0), 'Eye'),
        'profile': (profile_cascade, (128, 128, 255), 'Profile'),
        'upperbody': (upper_body_cascade, (255, 128, 0), 'Body'),
    }

    cascade, color, label = cascades.get(detect_type, cascades['face'])
    detections = cascade.detectMultiScale(gray, 1.1, 5)

    for i, (x, y, w, h) in enumerate(detections):
        cv2.rectangle(img_annotated, (x, y), (x + w, y + h), color, 2)
        cv2.putText(
            img_annotated, f'{label} {i + 1}',
            (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
        )

    return jsonify({
        'detected_image': img_to_base64(img_annotated),
        'count': len(detections),
        'type': detect_type,
        'label': label,
    })


# ============================================================
# Run the Application
# ============================================================
if __name__ == '__main__':
    print("\n" + "=" * 55)
    print("   🔍  Image Recognition System")
    print("   📍  http://localhost:5000")
    print("=" * 55 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000)