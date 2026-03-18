import 'dart:io';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:geolocator/geolocator.dart';
import 'package:image/image.dart' as img_lib;

import '../models/risk_result.dart';
import '../services/inference_service.dart';
import '../services/phc_referral_service.dart';
import '../services/sclera_detector.dart';
import '../utils/locale_utils.dart';
import 'result_screen.dart';

class MainScreen extends StatefulWidget {
  const MainScreen({super.key});

  @override
  State<MainScreen> createState() => _MainScreenState();
}

class _MainScreenState extends State<MainScreen> {
  CameraController? _controller;
  List<CameraDescription> _cameras = [];
  bool _isLoading = false;
  bool _cameraReady = false;

  // ASHA worker ID — in production, read from login / shared prefs
  final String _ashaId = 'MH-ASHA-DEMO-001';

  // Baby age in hours — in production, collected from intake form
  int _babyAgeHours = 24;

  @override
  void initState() {
    super.initState();
    _initCamera();
  }

  Future<void> _initCamera() async {
    _cameras = await availableCameras();
    if (_cameras.isEmpty) return;

    _controller = CameraController(
      _cameras.first,
      ResolutionPreset.high,
      enableAudio: false,
    );
    await _controller!.initialize();
    if (mounted) setState(() => _cameraReady = true);
  }

  @override
  void dispose() {
    _controller?.dispose();
    super.dispose();
  }

  Future<void> _analyseImage() async {
    if (_controller == null || !_controller!.value.isInitialized) return;
    setState(() => _isLoading = true);

    try {
      // ── Step 1: Capture ──────────────────────────────────────────────
      final XFile imageFile = await _controller!.takePicture();

      // ── Step 2: Sclera detection ─────────────────────────────────────
      final img_lib.Image? sclera = await ScleraDetector.detect(imageFile.path);
      if (sclera == null) {
        _showRetryDialog('Could not detect eye sclera.\nPlease retake — ensure eye is visible and well-lit.');
        return;
      }

      // ── Step 3: TF Lite inference ────────────────────────────────────
      final String lang = LocaleUtils.getLanguageCode(context);
      final RiskResult result = await InferenceService.classify(sclera, lang: lang);

      // ── Step 4: PHC referral if HIGH risk ────────────────────────────
      if (result.risk == RiskLevel.high) {
        final Position pos = await _getLocation();
        await PHCReferralService.sendAlert(
          babyAgeHours: _babyAgeHours,
          ashaId: _ashaId,
          lat: pos.latitude,
          lon: pos.longitude,
        );
      }

      // ── Step 5: Navigate to result screen ───────────────────────────
      if (mounted) {
        Navigator.push(
          context,
          MaterialPageRoute(builder: (_) => ResultScreen(result: result)),
        );
      }
    } catch (e) {
      _showError('An error occurred: $e');
    } finally {
      if (mounted) setState(() => _isLoading = false);
    }
  }

  Future<Position> _getLocation() async {
    LocationPermission perm = await Geolocator.checkPermission();
    if (perm == LocationPermission.denied) {
      perm = await Geolocator.requestPermission();
    }
    return await Geolocator.getCurrentPosition(
      desiredAccuracy: LocationAccuracy.medium,
    );
  }

  void _showRetryDialog(String message) {
    showDialog(
      context: context,
      builder: (_) => AlertDialog(
        title: const Text('Retry Needed'),
        content: Text(message),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('OK'),
          ),
        ],
      ),
    );
  }

  void _showError(String message) {
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(content: Text(message), backgroundColor: Colors.red),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.black,
      appBar: AppBar(
        backgroundColor: const Color(0xFF1B5E20),
        title: const Text(
          'NeoScreen',
          style: TextStyle(color: Colors.white, fontWeight: FontWeight.bold),
        ),
        subtitle: const Text(
          'Newborn Jaundice Detection',
          style: TextStyle(color: Colors.white70, fontSize: 12),
        ),
      ),
      body: Stack(
        children: [
          // Camera preview
          if (_cameraReady && _controller != null)
            Center(child: CameraPreview(_controller!))
          else
            const Center(
              child: CircularProgressIndicator(color: Colors.white),
            ),

          // Overlay: eye-target guide
          Center(
            child: Container(
              width: 220,
              height: 140,
              decoration: BoxDecoration(
                border: Border.all(color: Colors.white60, width: 2),
                borderRadius: BorderRadius.circular(12),
              ),
              child: const Center(
                child: Text(
                  'Align baby\'s eye here',
                  style: TextStyle(color: Colors.white70, fontSize: 13),
                  textAlign: TextAlign.center,
                ),
              ),
            ),
          ),

          // Loading overlay
          if (_isLoading)
            Container(
              color: Colors.black54,
              child: const Center(
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    CircularProgressIndicator(color: Colors.white),
                    SizedBox(height: 16),
                    Text(
                      'Analysing...',
                      style: TextStyle(color: Colors.white, fontSize: 16),
                    ),
                  ],
                ),
              ),
            ),
        ],
      ),

      // Capture button
      floatingActionButton: FloatingActionButton.large(
        onPressed: _isLoading ? null : _analyseImage,
        backgroundColor: const Color(0xFF1B5E20),
        child: _isLoading
            ? const CircularProgressIndicator(color: Colors.white)
            : const Icon(Icons.camera_alt, color: Colors.white, size: 36),
      ),
      floatingActionButtonLocation: FloatingActionButtonLocation.centerFloat,
    );
  }
}
