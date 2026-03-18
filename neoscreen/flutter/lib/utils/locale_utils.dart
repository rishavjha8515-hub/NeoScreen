import 'package:flutter/material.dart';

class LocaleUtils {
  static String getLanguageCode(BuildContext context) {
    final code = Localizations.localeOf(context).languageCode;
    return code == 'hi' ? 'hi' : 'en';
  }
}
