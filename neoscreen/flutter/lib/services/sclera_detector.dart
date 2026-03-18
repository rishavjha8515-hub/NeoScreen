import 'dart:io';

import 'package:image/image.dart' as img_lib;

/// Native Dart sclera detection using HSV thresholding.
/// Mirrors the Python OpenCV pipeline from ml/sclera_detection.py.
class ScleraDetector {
  /// Returns a 224×224 cropped sclera image, or null if not detected.
  static Future<img_lib.Image?> detect(String imagePath) async {
    final bytes = await File(imagePath).readAsBytes();
    img_lib.Image? image = img_lib.decodeImage(bytes);
    if (image == null) return null;

    // Resize for faster processing
    image = img_lib.copyResize(image, width: 640);

    // Apply CLAHE-equivalent: increase contrast via histogram equalisation
    image = _enhanceContrast(image);

    // White balance correction
    image = _whiteBalance(image);

    // HSV-based sclera mask
    final roi = _findScleraROI(image);
    if (roi == null) return null;

    return img_lib.copyResize(roi, width: 224, height: 224);
  }

  static img_lib.Image _enhanceContrast(img_lib.Image src) {
    // Simple contrast stretch (approximates CLAHE behaviour)
    return img_lib.adjustColor(src, contrast: 1.3, brightness: 0.0);
  }

  static img_lib.Image _whiteBalance(img_lib.Image src) {
    // Grey-world white balance
    double sumR = 0, sumG = 0, sumB = 0;
    final n = src.width * src.height;

    for (final pixel in src) {
      sumR += pixel.r;
      sumG += pixel.g;
      sumB += pixel.b;
    }

    final avgR = sumR / n;
    final avgG = sumG / n;
    final avgB = sumB / n;
    final avg = (avgR + avgG + avgB) / 3;

    final scaleR = avg / (avgR + 1e-6);
    final scaleG = avg / (avgG + 1e-6);
    final scaleB = avg / (avgB + 1e-6);

    final result = src.clone();
    for (int y = 0; y < result.height; y++) {
      for (int x = 0; x < result.width; x++) {
        final p = result.getPixel(x, y);
        result.setPixel(
          x,
          y,
          result.getColor(
            (p.r * scaleR).clamp(0, 255).toInt(),
            (p.g * scaleG).clamp(0, 255).toInt(),
            (p.b * scaleB).clamp(0, 255).toInt(),
          ),
        );
      }
    }
    return result;
  }

  static img_lib.Image? _findScleraROI(img_lib.Image src) {
    int minX = src.width, minY = src.height, maxX = 0, maxY = 0;
    int whitePixelCount = 0;

    for (int y = 0; y < src.height; y++) {
      for (int x = 0; x < src.width; x++) {
        final p = src.getPixel(x, y);
        final r = p.r.toDouble();
        final g = p.g.toDouble();
        final b = p.b.toDouble();

        // Convert to HSV and check for white/near-white sclera
        final max = [r, g, b].reduce((a, b) => a > b ? a : b);
        final min = [r, g, b].reduce((a, b) => a < b ? a : b);
        final v = max / 255.0;
        final s = max == 0 ? 0.0 : (max - min) / max;

        // White region: high V, low S (similar to HSV bounds in Python)
        if (v >= 0.70 && s <= 0.15) {
          whitePixelCount++;
          if (x < minX) minX = x;
          if (y < minY) minY = y;
          if (x > maxX) maxX = x;
          if (y > maxY) maxY = y;
        }
      }
    }

    // Reject if region too small (noise)
    if (whitePixelCount < 500 || (maxX - minX) < 20 || (maxY - minY) < 20) {
      return null;
    }

    return img_lib.copyCrop(
      src,
      x: minX,
      y: minY,
      width: maxX - minX,
      height: maxY - minY,
    );
  }
}
