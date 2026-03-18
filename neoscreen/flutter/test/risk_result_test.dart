import 'package:flutter_test/flutter_test.dart';
import 'package:neoscreen/models/risk_result.dart';

void main() {
  group('RiskResult.fromProbabilities', () {
    test('P(High) >= 0.35 → HIGH', () {
      final r = RiskResult.fromProbabilities([0.30, 0.35, 0.35]);
      expect(r.risk, RiskLevel.high);
    });

    test('P(Low) >= 0.60 → LOW', () {
      final r = RiskResult.fromProbabilities([0.65, 0.20, 0.15]);
      expect(r.risk, RiskLevel.low);
    });

    test('Neither threshold → MEDIUM', () {
      final r = RiskResult.fromProbabilities([0.40, 0.40, 0.20]);
      expect(r.risk, RiskLevel.medium);
    });

    test('High threshold wins over low threshold', () {
      // Both could fire — High takes precedence
      final r = RiskResult.fromProbabilities([0.61, 0.00, 0.39]);
      expect(r.risk, RiskLevel.high);
    });

    test('Hindi messages are present for all risk levels', () {
      for (final probs in [
        [0.10, 0.10, 0.80],
        [0.40, 0.40, 0.20],
        [0.70, 0.20, 0.10],
      ]) {
        final r = RiskResult.fromProbabilities(probs.cast<double>());
        expect(r.messageHi, isNotEmpty);
        expect(r.messageEn, isNotEmpty);
      }
    });

    test('Probabilities are stored accurately', () {
      final r = RiskResult.fromProbabilities([0.1, 0.3, 0.6]);
      expect(r.probLow, closeTo(0.1, 0.001));
      expect(r.probMedium, closeTo(0.3, 0.001));
      expect(r.probHigh, closeTo(0.6, 0.001));
    });
  });
}
