package io.github.mtbarr.rinha.service;

import jakarta.inject.Singleton;
import java.nio.charset.StandardCharsets;

@Singleton
public final class FraudRequestParser {

  private static final int FEATURE_VECTOR_DIMENSIONS = 14;

  private static final float MAX_AMOUNT = 10_000f;
  private static final float MAX_INSTALLMENTS = 12f;
  private static final float RATIO_CAP = 10f;
  private static final float MAX_MINUTES = 1440f;
  private static final float MAX_KM = 1000f;
  private static final float MAX_TX_24H = 20f;
  private static final float MAX_MERCH_AVG = 10_000f;

  private static final int[] DOW_TABLE = {0, 3, 2, 5, 0, 3, 5, 1, 4, 6, 2, 4};

  private static final String K_TRANSACTION = "\"transaction\"";
  private static final String K_CUSTOMER = "\"customer\"";
  private static final String K_MERCHANT = "\"merchant\"";
  private static final String K_TERMINAL = "\"terminal\"";
  private static final String K_AMOUNT = "\"amount\"";
  private static final String K_INSTALLMENTS = "\"installments\"";
  private static final String K_REQUESTED_AT = "\"requested_at\"";
  private static final String K_AVG_AMOUNT = "\"avg_amount\"";
  private static final String K_TX_COUNT = "\"tx_count_24h\"";
  private static final String K_ID = "\"id\"";
  private static final String K_MCC = "\"mcc\"";
  private static final String K_ONLINE = "\"is_online\"";
  private static final String K_CARD = "\"card_present\"";
  private static final String K_KM_HOME = "\"km_from_home\"";
  private static final String K_LAST_TX = "\"last_transaction\"";
  private static final String K_TIMESTAMP = "\"timestamp\"";
  private static final String K_KM_CURRENT = "\"km_from_current\"";
  private static final String K_KNOWN = "\"known_merchants\"";

  private final ThreadLocal<float[]> fvBuf = ThreadLocal.withInitial(
    () -> new float[FEATURE_VECTOR_DIMENSIONS]
  );

  public float[] extractFeatureVector(final byte[] payload) {
    final String j = new String(payload, StandardCharsets.UTF_8);
    return extract(j);
  }

  private float[] extract(final String j) {
    final float[] fv = fvBuf.get();

    final int txS = sectionStart(j, 0, K_TRANSACTION);
    final float txAmt = extractFloat(j, txS, K_AMOUNT);
    final int txInst = (int) extractFloat(j, txS, K_INSTALLMENTS);
    final String reqAt = extractStr(j, txS, K_REQUESTED_AT);
    final int txHour = tsHour(reqAt);
    final int txDow = tsDow(reqAt);
    final long txEpoch = tsEpoch(reqAt);

    final int custS = sectionStart(j, 0, K_CUSTOMER);
    final float custAvg = extractFloat(j, custS, K_AVG_AMOUNT);
    final int custTxN = (int) extractFloat(j, custS, K_TX_COUNT);

    final int merchS = sectionStart(j, 0, K_MERCHANT);
    final String merchId = extractStr(j, merchS, K_ID);
    final int mcc = extractIntStr(j, merchS, K_MCC);
    final float merchAvg = extractFloat(j, merchS, K_AVG_AMOUNT);

    final boolean unknownMerchant = !merchantIsKnown(j, custS, K_KNOWN, merchId);

    final int termS = sectionStart(j, 0, K_TERMINAL);
    final boolean isOnline = extractBool(j, termS, K_ONLINE);
    final boolean cardPresent = extractBool(j, termS, K_CARD);
    final float kmHome = extractFloat(j, termS, K_KM_HOME);

    long minsSinceLast = -1L;
    float kmFromCurrent = -1f;
    final int ltIdx = j.indexOf(K_LAST_TX);
    if (ltIdx >= 0) {
      int colon = j.indexOf(':', ltIdx) + 1;
      char c;
      while ((c = j.charAt(colon)) == ' ' || c == '\n' || c == '\r') colon++;
      if (j.charAt(colon) == '{') {
        final long lastEpoch = tsEpoch(extractStr(j, colon, K_TIMESTAMP));
        kmFromCurrent = extractFloat(j, colon, K_KM_CURRENT);
        minsSinceLast = Math.max(0L, txEpoch - lastEpoch) / 60L;
      }
    }

    fv[0] = clamp(txAmt / MAX_AMOUNT);
    fv[1] = clamp(txInst / MAX_INSTALLMENTS);
    fv[2] = custAvg > 0f ? clamp((txAmt / custAvg) / RATIO_CAP) : 0f;
    fv[3] = txHour / 23f;
    fv[4] = (txDow - 1) / 6f;

    if (ltIdx >= 0 && minsSinceLast >= 0) {
      fv[5] = clamp(minsSinceLast / MAX_MINUTES);
      fv[6] = clamp(kmFromCurrent / MAX_KM);
    } else {
      fv[5] = -1f;
      fv[6] = -1f;
    }

    fv[7] = clamp(kmHome / MAX_KM);
    fv[8] = clamp(custTxN / MAX_TX_24H);
    fv[9] = isOnline ? 1f : 0f;
    fv[10] = cardPresent ? 1f : 0f;
    fv[11] = unknownMerchant ? 1f : 0f;
    fv[12] = mccRisk(mcc);
    fv[13] = clamp(merchAvg / MAX_MERCH_AVG);

    return fv;
  }

  private static int sectionStart(final String j, final int from, final String key) {
    final int k = j.indexOf(key, from);
    final int colon = j.indexOf(':', k);
    return j.indexOf('{', colon);
  }

  private static String extractStr(final String j, final int from, final String key) {
    final int k = j.indexOf(key, from);
    final int colon = j.indexOf(':', k);
    final int q1 = j.indexOf('"', colon + 1);
    final int q2 = j.indexOf('"', q1 + 1);
    return j.substring(q1 + 1, q2);
  }

  private static float extractFloat(final String j, final int from, final String key) {
    final int k = j.indexOf(key, from);
    int start = j.indexOf(':', k) + 1;
    char c;
    while ((c = j.charAt(start)) == ' ' || c == '\n' || c == '\r') start++;
    int end = start;
    while (end < j.length() && (c = j.charAt(end)) != ',' && c != '}' && c != '\n' && c != '\r') end++;
    return parseFloat(j, start, end);
  }

  private static int extractIntStr(final String j, final int from, final String key) {
    final int k = j.indexOf(key, from);
    final int q1 = j.indexOf('"', j.indexOf(':', k) + 1) + 1;
    int v = 0;
    int i = q1;
    while (i < j.length() && j.charAt(i) != '"') {
      v = v * 10 + (j.charAt(i) - '0');
      i++;
    }
    return v;
  }

  private static boolean extractBool(final String j, final int from, final String key) {
    final int k = j.indexOf(key, from);
    int start = j.indexOf(':', k) + 1;
    while (start < j.length() && j.charAt(start) == ' ') start++;
    return j.charAt(start) == 't';
  }

  private static boolean merchantIsKnown(final String j, final int from, final String key, final String targetId) {
    final int k = j.indexOf(key, from);
    if (k < 0) return false;
    final int bracket = j.indexOf('[', k);
    final int end = j.indexOf(']', bracket);
    final int tLen = targetId.length();
    int pos = bracket + 1;
    while (pos < end) {
      final int q1 = j.indexOf('"', pos);
      if (q1 < 0 || q1 >= end) break;
      final int q2 = j.indexOf('"', q1 + 1);
      if (q2 - q1 - 1 == tLen && j.regionMatches(q1 + 1, targetId, 0, tLen)) return true;
      pos = q2 + 1;
    }
    return false;
  }

  private static int tsHour(final String s) {
    return (s.charAt(11) - '0') * 10 + (s.charAt(12) - '0');
  }

  private static int tsDow(final String s) {
    final int y = digits(s, 0, 4);
    final int m = digits(s, 5, 7);
    final int d = digits(s, 8, 10);
    final int adjY = m < 3 ? y - 1 : y;
    final int adjM = m < 3 ? m + 9 : m - 3;
    final int era = adjY >= 0 ? adjY / 400 : (adjY - 399) / 400;
    final int yoe = adjY - era * 400;
    final int doy = (153 * adjM + 2) / 5 + d - 1;
    final int doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    int dow = era * 146097 + doe - 719468;
    dow %= 7;
    return dow <= 0 ? dow + 7 : dow;
  }

  private static long tsEpoch(final String s) {
    final int y = digits(s, 0, 4);
    final int mo = digits(s, 5, 7);
    final int d = digits(s, 8, 10);
    final int h = digits(s, 11, 13);
    final int min = digits(s, 14, 16);
    final int sec = digits(s, 17, 19);
    int m = mo;
    int yr = y;
    if (m <= 2) { yr--; m += 9; } else { m -= 3; }
    final long era = (yr >= 0 ? yr : yr - 399) / 400;
    final int yoe = (int) (yr - era * 400);
    final int doy = (153 * m + 2) / 5 + d - 1;
    final int doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    final long days = era * 146097L + doe - 719468L;
    return days * 86400L + h * 3600L + min * 60L + sec;
  }

  private static int digits(final String s, final int start, final int end) {
    int v = 0;
    for (int i = start; i < end; i++) v = v * 10 + (s.charAt(i) - '0');
    return v;
  }

  private static float parseFloat(final String s, int start, final int end) {
    float result = 0f;
    boolean neg = false;
    char c = s.charAt(start);
    if (c == '-') { neg = true; start++; }
    while (start < end && (c = s.charAt(start)) >= '0' && c <= '9') {
      result = result * 10f + (c - '0');
      start++;
    }
    if (start < end && s.charAt(start) == '.') {
      start++;
      float frac = 0f;
      float div = 1f;
      while (start < end && (c = s.charAt(start)) >= '0' && c <= '9') {
        frac = frac * 10f + (c - '0');
        div *= 10f;
        start++;
      }
      result += frac / div;
    }
    return neg ? -result : result;
  }

  private static float clamp(final float v) {
    return v < 0f ? 0f : (v > 1f ? 1f : v);
  }

  public static float dayOfWeek(final int year, final int month, final int day) {
    int y = year;
    if (month < 3) y--;
    final int dow = (y + y / 4 - y / 100 + y / 400 + DOW_TABLE[month - 1] + day) % 7;
    return (dow + 6) % 7;
  }

  private static float mccRisk(final int mcc) {
    return switch (mcc) {
      case 5411 -> 0.15f;
      case 5812 -> 0.30f;
      case 5912 -> 0.20f;
      case 5944 -> 0.45f;
      case 7801 -> 0.80f;
      case 7802 -> 0.75f;
      case 7995 -> 0.85f;
      case 4511 -> 0.35f;
      case 5311 -> 0.25f;
      default -> 0.50f;
    };
  }
}
