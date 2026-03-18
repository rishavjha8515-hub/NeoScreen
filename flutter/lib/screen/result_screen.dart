import 'package:flutter/material.dart';

import '../models/risk_result.dart';

class ResultScreen extends StatelessWidget {
  final RiskResult result;

  const ResultScreen({super.key, required this.result});

  Color get _bgColor {
    switch (result.risk) {
      case RiskLevel.high:
        return const Color(0xFFB71C1C);
      case RiskLevel.medium:
        return const Color(0xFFF57F17);
      case RiskLevel.low:
        return const Color(0xFF1B5E20);
    }
  }

  IconData get _icon {
    switch (result.risk) {
      case RiskLevel.high:
        return Icons.warning_rounded;
      case RiskLevel.medium:
        return Icons.info_rounded;
      case RiskLevel.low:
        return Icons.check_circle_rounded;
    }
  }

  String get _riskLabel {
    switch (result.risk) {
      case RiskLevel.high:
        return 'HIGH RISK';
      case RiskLevel.medium:
        return 'MEDIUM RISK';
      case RiskLevel.low:
        return 'LOW RISK';
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: _bgColor,
      body: SafeArea(
        child: Padding(
          padding: const EdgeInsets.all(24),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Icon(_icon, size: 96, color: Colors.white),
              const SizedBox(height: 24),
              Text(
                _riskLabel,
                style: const TextStyle(
                  color: Colors.white,
                  fontSize: 32,
                  fontWeight: FontWeight.bold,
                  letterSpacing: 1.5,
                ),
              ),
              const SizedBox(height: 32),

              // Hindi message
              Container(
                width: double.infinity,
                padding: const EdgeInsets.all(20),
                decoration: BoxDecoration(
                  color: Colors.white24,
                  borderRadius: BorderRadius.circular(16),
                ),
                child: Column(
                  children: [
                    Text(
                      result.messageHi,
                      style: const TextStyle(
                        color: Colors.white,
                        fontSize: 22,
                        fontWeight: FontWeight.w600,
                      ),
                      textAlign: TextAlign.center,
                    ),
                    const SizedBox(height: 12),
                    Text(
                      result.messageEn,
                      style: const TextStyle(
                        color: Colors.white70,
                        fontSize: 16,
                      ),
                      textAlign: TextAlign.center,
                    ),
                  ],
                ),
              ),

              const SizedBox(height: 24),

              // Confidence scores
              _ConfidenceBar(label: 'Low', value: result.probLow, color: Colors.green.shade200),
              const SizedBox(height: 8),
              _ConfidenceBar(label: 'Medium', value: result.probMedium, color: Colors.orange.shade200),
              const SizedBox(height: 8),
              _ConfidenceBar(label: 'High', value: result.probHigh, color: Colors.red.shade200),

              if (result.risk == RiskLevel.high) ...[
                const SizedBox(height: 24),
                Container(
                  padding: const EdgeInsets.all(14),
                  decoration: BoxDecoration(
                    color: Colors.white30,
                    borderRadius: BorderRadius.circular(12),
                  ),
                  child: const Row(
                    children: [
                      Icon(Icons.sms, color: Colors.white),
                      SizedBox(width: 12),
                      Expanded(
                        child: Text(
                          'PHC alert sent automatically',
                          style: TextStyle(color: Colors.white, fontSize: 14),
                        ),
                      ),
                    ],
                  ),
                ),
              ],

              const SizedBox(height: 40),
              ElevatedButton.icon(
                onPressed: () => Navigator.pop(context),
                icon: const Icon(Icons.camera_alt),
                label: const Text('Screen Another Baby'),
                style: ElevatedButton.styleFrom(
                  backgroundColor: Colors.white,
                  foregroundColor: _bgColor,
                  padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 14),
                  textStyle: const TextStyle(fontSize: 16, fontWeight: FontWeight.bold),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

class _ConfidenceBar extends StatelessWidget {
  final String label;
  final double value;
  final Color color;

  const _ConfidenceBar({
    required this.label,
    required this.value,
    required this.color,
  });

  @override
  Widget build(BuildContext context) {
    return Row(
      children: [
        SizedBox(
          width: 60,
          child: Text(label, style: const TextStyle(color: Colors.white70, fontSize: 13)),
        ),
        Expanded(
          child: ClipRRect(
            borderRadius: BorderRadius.circular(4),
            child: LinearProgressIndicator(
              value: value,
              backgroundColor: Colors.white24,
              valueColor: AlwaysStoppedAnimation(color),
              minHeight: 10,
            ),
          ),
        ),
        const SizedBox(width: 8),
        Text(
          '${(value * 100).toStringAsFixed(1)}%',
          style: const TextStyle(color: Colors.white70, fontSize: 12),
        ),
      ],
    );
  }
}