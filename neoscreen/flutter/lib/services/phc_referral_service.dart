import 'dart:io';
import 'dart:math';

import 'package:flutter_dotenv/flutter_dotenv.dart';
import 'package:http/http.dart' as http;
import 'package:sqflite/sqflite.dart';
import 'package:path/path.dart' as p;
import 'package:flutter/services.dart' show rootBundle;

class _PHC {
  final String name;
  final String phone;
  final double lat;
  final double lon;

  const _PHC(this.name, this.phone, this.lat, this.lon);
}

class PHCReferralService {
  static Database? _db;

  static Future<Database> _getDb() async {
    if (_db != null) return _db!;

    final dbPath = p.join(await getDatabasesPath(), 'phc_maharashtra.db');

    // Copy bundled DB from assets on first run
    if (!File(dbPath).existsSync()) {
      final data = await rootBundle.load('assets/data/phc_maharashtra.db');
      final bytes = data.buffer.asUint8List();
      await File(dbPath).writeAsBytes(bytes, flush: true);
    }

    _db = await openDatabase(dbPath, readOnly: true);
    return _db!;
  }

  static Future<_PHC> _getNearestPHC(double lat, double lon) async {
    final db = await _getDb();
    final rows = await db.rawQuery('SELECT name, phone, lat, lon FROM phcs');

    _PHC? nearest;
    double minDist = double.infinity;

    for (final row in rows) {
      final phcLat = row['lat'] as double;
      final phcLon = row['lon'] as double;
      final dist = _haversineKm(lat, lon, phcLat, phcLon);
      if (dist < minDist) {
        minDist = dist;
        nearest = _PHC(
          row['name'] as String,
          row['phone'] as String,
          phcLat,
          phcLon,
        );
      }
    }

    if (nearest == null) throw Exception('PHC database is empty');
    return nearest;
  }

  static double _haversineKm(double lat1, double lon1, double lat2, double lon2) {
    const r = 6371.0;
    final dLat = _deg2rad(lat2 - lat1);
    final dLon = _deg2rad(lon2 - lon1);
    final a = sin(dLat / 2) * sin(dLat / 2) +
        cos(_deg2rad(lat1)) * cos(_deg2rad(lat2)) * sin(dLon / 2) * sin(dLon / 2);
    return r * 2 * atan2(sqrt(a), sqrt(1 - a));
  }

  static double _deg2rad(double deg) => deg * pi / 180;

  /// Send Twilio SMS to nearest PHC. Only fires on HIGH risk.
  static Future<void> sendAlert({
    required int babyAgeHours,
    required String ashaId,
    required double lat,
    required double lon,
  }) async {
    final phc = await _getNearestPHC(lat, lon);

    final sid = dotenv.env['TWILIO_SID'] ?? '';
    final token = dotenv.env['TWILIO_TOKEN'] ?? '';
    final from = dotenv.env['TWILIO_FROM'] ?? '';

    if (sid.isEmpty || token.isEmpty || from.isEmpty) {
      throw Exception('Twilio credentials missing in .env');
    }

    final body =
        'NeoScreen HIGH RISK ALERT | Baby age: ${babyAgeHours}h | '
        'ASHA: $ashaId | Location: ${lat.toStringAsFixed(5)},${lon.toStringAsFixed(5)} | '
        'PHC: ${phc.name}';

    final response = await http.post(
      Uri.parse('https://api.twilio.com/2010-04-01/Accounts/$sid/Messages.json'),
      headers: {
        'Authorization':
            'Basic ${base64Encode('$sid:$token')}',
        'Content-Type': 'application/x-www-form-urlencoded',
      },
      body: {'From': from, 'To': phc.phone, 'Body': body},
    );

    if (response.statusCode != 201) {
      throw Exception('Twilio SMS failed: ${response.body}');
    }
  }

  static String base64Encode(String input) {
    return _base64(input.codeUnits);
  }

  static String _base64(List<int> bytes) {
    const chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/';
    final buf = StringBuffer();
    for (int i = 0; i < bytes.length; i += 3) {
      final b0 = bytes[i];
      final b1 = i + 1 < bytes.length ? bytes[i + 1] : 0;
      final b2 = i + 2 < bytes.length ? bytes[i + 2] : 0;
      buf.write(chars[(b0 >> 2) & 0x3F]);
      buf.write(chars[((b0 << 4) | (b1 >> 4)) & 0x3F]);
      buf.write(i + 1 < bytes.length ? chars[((b1 << 2) | (b2 >> 6)) & 0x3F] : '=');
      buf.write(i + 2 < bytes.length ? chars[b2 & 0x3F] : '=');
    }
    return buf.toString();
  }
}
