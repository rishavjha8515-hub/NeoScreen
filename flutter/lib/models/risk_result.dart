enum RiskLevel { low, medium, high }

class RiskResult {
  final RiskLevel risk;
  final double probLow;
  final double probMedium;
  final double probHigh;
  final String messageHi;
  final String messageEn;
  final DateTime timestamp;

  const RiskResult({
    required this.risk,
    required this.probLow,
    required this.probMedium,
    required this.probHigh,
    required this.messageHi,
    required this.messageEn,
    required this.timestamp,
  });

  static const _messages = {
    RiskLevel.high: {
      'hi': 'Shishu ko turant PHC le jaayein',
      'en': 'Take baby to PHC immediately',
    },
    RiskLevel.medium: {
      'hi': 'Kal dobara check karein',
      'en': 'Recheck tomorrow',
    },
    RiskLevel.low: {
      'hi': 'Koi khatra nahi',
      'en': 'No immediate risk',
    },
  };

  static RiskResult fromProbabilities(List<double> probs) {
    final pLow = probs[0];
    final pMed = probs[1];
    final pHigh = probs[2];

    RiskLevel risk;
    if (pHigh >= 0.35) {
      risk = RiskLevel.high;
    } else if (pLow >= 0.60) {
      risk = RiskLevel.low;
    } else {
      risk = RiskLevel.medium;
    }

    return RiskResult(
      risk: risk,
      probLow: pLow,
      probMedium: pMed,
      probHigh: pHigh,
      messageHi: _messages[risk]!['hi']!,
      messageEn: _messages[risk]!['en']!,
      timestamp: DateTime.now(),
    );
  }
}