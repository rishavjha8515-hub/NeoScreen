import 'dart:typed_data';

import 'package:image/image.dart' as img_lib;
import 'package:tflite_flutter/tflite_flutter.dart';

import '../models/risk_result.dart';

class InferenceService {
  static Interpreter? _interpreter;

  static Future<Interpreter> _getInterpreter() async {
    _interpreter ??= await Interpreter.fromAsset('assets/models/neoscreen_v1.tflite');
    return _interpreter!;
  }

  /// Run inference on a 224×224 sclera image.
  static Future<RiskResult> classify(img_lib.Image sclera, {String lang = 'hi'}) async {
    final interpreter = await _getInterpreter();

    // Normalise pixels to [0, 1] and build input tensor (1, 224, 224, 3)
    final input = _imageToFloat32(sclera);
    final output = List.filled(3, 0.0).reshape([1, 3]);

    interpreter.run(input, output);

    final probs = (output[0] as List).cast<double>();
    return RiskResult.fromProbabilities(probs);
  }

  static List<List<List<List<double>>>> _imageToFloat32(img_lib.Image image) {
    final resized = img_lib.copyResize(image, width: 224, height: 224);
    return [
      List.generate(224, (y) {
        return List.generate(224, (x) {
          final pixel = resized.getPixel(x, y);
          return [
            pixel.r / 255.0,
            pixel.g / 255.0,
            pixel.b / 255.0,
          ];
        });
      }),
    ];
  }
}