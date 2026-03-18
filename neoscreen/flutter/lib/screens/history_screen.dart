import 'package:flutter/material.dart';
import 'package:sqflite/sqflite.dart';
import 'package:path/path.dart' as p;

import '../models/risk_result.dart';

/// Local screening history — stores last 10 results, no PII.
class HistoryService {
  static Database? _db;

  static Future<Database> _getDb() async {
    if (_db != null) return _db!;
    final dbPath = p.join(await getDatabasesPath(), 'neoscreen_history.db');
    _db = await openDatabase(
      dbPath,
      version: 1,
      onCreate: (db, _) => db.execute('''
        CREATE TABLE history (
          id         INTEGER PRIMARY KEY AUTOINCREMENT,
          timestamp  TEXT    NOT NULL,
          risk       TEXT    NOT NULL,
          prob_low   REAL    NOT NULL,
          prob_med   REAL    NOT NULL,
          prob_high  REAL    NOT NULL,
          baby_age_h INTEGER NOT NULL,
          asha_id    TEXT    NOT NULL
        )
      '''),
    );
    return _db!;
  }

  static Future<void> save(RiskResult r, int babyAgeHours, String ashaId) async {
    final db = await _getDb();
    await db.insert('history', {
      'timestamp':  r.timestamp.toIso8601String(),
      'risk':       r.risk.name,
      'prob_low':   r.probLow,
      'prob_med':   r.probMedium,
      'prob_high':  r.probHigh,
      'baby_age_h': babyAgeHours,
      'asha_id':    ashaId,
    });
    // Keep only last 10
    await db.rawDelete('''
      DELETE FROM history WHERE id NOT IN (
        SELECT id FROM history ORDER BY id DESC LIMIT 10
      )
    ''');
  }

  static Future<List<Map<String, dynamic>>> load() async {
    final db = await _getDb();
    return db.query('history', orderBy: 'id DESC', limit: 10);
  }
}

/// History screen showing last 10 screenings.
class HistoryScreen extends StatefulWidget {
  const HistoryScreen({super.key});

  @override
  State<HistoryScreen> createState() => _HistoryScreenState();
}

class _HistoryScreenState extends State<HistoryScreen> {
  List<Map<String, dynamic>> _records = [];
  bool _loading = true;

  @override
  void initState() {
    super.initState();
    _load();
  }

  Future<void> _load() async {
    final records = await HistoryService.load();
    setState(() { _records = records; _loading = false; });
  }

  Color _riskColor(String risk) {
    switch (risk) {
      case 'high':   return const Color(0xFFB71C1C);
      case 'medium': return const Color(0xFFF57F17);
      default:       return const Color(0xFF1B5E20);
    }
  }

  IconData _riskIcon(String risk) {
    switch (risk) {
      case 'high':   return Icons.warning_rounded;
      case 'medium': return Icons.info_rounded;
      default:       return Icons.check_circle_rounded;
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        backgroundColor: const Color(0xFF1B5E20),
        title: const Text('Screening History',
            style: TextStyle(color: Colors.white, fontWeight: FontWeight.bold)),
        iconTheme: const IconThemeData(color: Colors.white),
      ),
      body: _loading
          ? const Center(child: CircularProgressIndicator())
          : _records.isEmpty
              ? const Center(
                  child: Column(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      Icon(Icons.history, size: 64, color: Colors.grey),
                      SizedBox(height: 16),
                      Text('No screenings yet',
                          style: TextStyle(color: Colors.grey, fontSize: 16)),
                    ],
                  ),
                )
              : ListView.builder(
                  padding: const EdgeInsets.all(16),
                  itemCount: _records.length,
                  itemBuilder: (_, i) {
                    final r = _records[i];
                    final risk = r['risk'] as String;
                    final ts = DateTime.parse(r['timestamp'] as String);
                    final age = r['baby_age_h'] as int;
                    final pHigh = (r['prob_high'] as double) * 100;

                    return Card(
                      margin: const EdgeInsets.only(bottom: 12),
                      shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(12)),
                      child: ListTile(
                        leading: CircleAvatar(
                          backgroundColor: _riskColor(risk),
                          child: Icon(_riskIcon(risk),
                              color: Colors.white, size: 20),
                        ),
                        title: Text(
                          '${risk.toUpperCase()} RISK',
                          style: TextStyle(
                            fontWeight: FontWeight.bold,
                            color: _riskColor(risk),
                          ),
                        ),
                        subtitle: Text(
                          'Baby age: ${age}h  •  P(High): ${pHigh.toStringAsFixed(1)}%\n'
                          '${ts.day}/${ts.month}/${ts.year}  ${ts.hour}:${ts.minute.toString().padLeft(2, '0')}',
                        ),
                        isThreeLine: true,
                      ),
                    );
                  },
                ),
    );
  }
}
