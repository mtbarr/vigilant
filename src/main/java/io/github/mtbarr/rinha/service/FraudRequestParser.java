package io.github.mtbarr.rinha.service;

import jakarta.inject.Singleton;

@Singleton
public final class FraudRequestParser {

  private static final int FEATURE_VECTOR_DIMENSIONS = 14;

  private static final float MAX_TRANSACTION_AMOUNT = 10_000f;
  private static final float MAX_INSTALLMENT_COUNT = 12f;
  private static final float AMOUNT_TO_AVG_RATIO_CAP = 10f;
  private static final float MAX_MINUTES_SINCE_LAST_TX = 1440f;
  private static final float MAX_DISTANCE_KM = 1000f;
  private static final float MAX_TRANSACTIONS_24H = 20f;
  private static final float MAX_MERCHANT_AVG_AMOUNT = 10_000f;

  private static final int[] DAY_OF_WEEK_MONTH_TABLE = {0, 3, 2, 5, 0, 3, 5, 1, 4, 6, 2, 4};

  private final ThreadLocal<float[]> featureVectorBuffer = ThreadLocal.withInitial(
    () -> new float[FEATURE_VECTOR_DIMENSIONS]
  );

  public float[] extractFeatureVector(final byte[] jsonPayload) {
    final String j = new String(jsonPayload, java.nio.charset.StandardCharsets.UTF_8);
    return extractFeatureVector(j);
  }

  public float[] extractFeatureVector(final String j) {
    final float[] featureVector = featureVectorBuffer.get();

    final int txStart = sectionStart(j, 0, "transaction");
    final float txAmount = extractFloat(j, txStart, "amount");
    final int txInstallments = extractInt(j, txStart, "installments");
    final String reqAtStr = extractStr(j, txStart, "requested_at");
    final int txHour = tsHour(reqAtStr);
    final int txDow = tsDayOfWeek(reqAtStr);
    final long txEpoch = tsEpochSeconds(reqAtStr);

    final int custStart = sectionStart(j, 0, "customer");
    final float custAvgAmount = extractFloat(j, custStart, "avg_amount");
    final int custTxCount = extractInt(j, custStart, "tx_count_24h");

    final int merchStart = sectionStart(j, 0, "merchant");
    final String merchId = extractStr(j, merchStart, "id");
    final int mccCode = extractIntStr(j, merchStart, "mcc");
    final float merchAvg = extractFloat(j, merchStart, "avg_amount");

    final boolean unknownMerchant = !merchantIsKnown(j, custStart, "known_merchants", merchId);

    final int termStart = sectionStart(j, 0, "terminal");
    final boolean isOnline = extractBool(j, termStart, "is_online");
    final boolean cardPresent = extractBool(j, termStart, "card_present");
    final float kmFromHome = extractFloat(j, termStart, "km_from_home");

    long minutesSinceLastTx = -1L;
    float distanceFromCurrentKm = -1f;

    int ltIdx = j.indexOf("\"last_transaction\"");
    if (ltIdx >= 0) {
      int colon = j.indexOf(':', ltIdx) + 1;
      while (j.charAt(colon) == ' ' || j.charAt(colon) == '\n' || j.charAt(colon) == '\r') {
        colon++;
      }
      if (j.charAt(colon) == '{') {
        long lastEpoch = tsEpochSeconds(extractStr(j, colon, "timestamp"));
        distanceFromCurrentKm = extractFloat(j, colon, "km_from_current");
        minutesSinceLastTx = Math.max(0L, txEpoch - lastEpoch) / 60L;
      }
    }

    featureVector[0] = clampToUnitRange(txAmount / MAX_TRANSACTION_AMOUNT);
    featureVector[1] = clampToUnitRange(txInstallments / MAX_INSTALLMENT_COUNT);
    featureVector[2] = custAvgAmount > 0f
                       ? clampToUnitRange((txAmount / custAvgAmount) / AMOUNT_TO_AVG_RATIO_CAP)
                       : 0f;
    featureVector[3] = txHour / 23f;
    featureVector[4] = (txDow - 1) / 6f;

    if (ltIdx >= 0 && minutesSinceLastTx >= 0) {
      featureVector[5] = clampToUnitRange(minutesSinceLastTx / MAX_MINUTES_SINCE_LAST_TX);
      featureVector[6] = clampToUnitRange(distanceFromCurrentKm / MAX_DISTANCE_KM);
    } else {
      featureVector[5] = -1f;
      featureVector[6] = -1f;
    }

    featureVector[7] = clampToUnitRange(kmFromHome / MAX_DISTANCE_KM);
    featureVector[8] = clampToUnitRange(custTxCount / MAX_TRANSACTIONS_24H);
    featureVector[9] = isOnline ? 1f : 0f;
    featureVector[10] = cardPresent ? 1f : 0f;
    featureVector[11] = unknownMerchant ? 1f : 0f;
    featureVector[12] = computeMccRiskScore(mccCode);
    featureVector[13] = clampToUnitRange(merchAvg / MAX_MERCHANT_AVG_AMOUNT);

    return featureVector;
  }

  private static int sectionStart(String j, int from, String key) {
    int k = j.indexOf('"' + key + '"', from);
    int colon = j.indexOf(':', k);
    return j.indexOf('{', colon);
  }

  private static String extractStr(String j, int from, String key) {
    int k = j.indexOf('"' + key + '"', from);
    int colon = j.indexOf(':', k);
    int q1 = j.indexOf('"', colon + 1);
    int q2 = j.indexOf('"', q1 + 1);
    return j.substring(q1 + 1, q2);
  }

  private static float extractFloat(String j, int from, String key) {
    int k = j.indexOf('"' + key + '"', from);
    int start = j.indexOf(':', k) + 1;
    while (j.charAt(start) == ' ') {
      start++;
    }
    int end = start;
    char c;
    while (end < j.length() && (c = j.charAt(end)) != ',' && c != '}' && c != '\n' && c != '\r') {
      end++;
    }
    return Float.parseFloat(j.substring(start, end).trim());
  }

  private static int extractInt(String j, int from, String key) {
    return (int) extractFloat(j, from, key);
  }

  private static int extractIntStr(String j, int from, String key) {
    int k = j.indexOf('"' + key + '"', from);
    int q1 = j.indexOf('"', j.indexOf(':', k) + 1) + 1;
    int v = 0;
    char c;
    while ((c = j.charAt(q1++)) != '"') {
      v = v * 10 + (c - '0');
    }
    return v;
  }

  private static boolean extractBool(String j, int from, String key) {
    int k = j.indexOf('"' + key + '"', from);
    int start = j.indexOf(':', k) + 1;
    while (j.charAt(start) == ' ') {
      start++;
    }
    return j.charAt(start) == 't';
  }

  private static boolean merchantIsKnown(String j, int from, String key, String targetId) {
    int k = j.indexOf('"' + key + '"', from);
    if (k < 0) {
      return false;
    }
    int bracket = j.indexOf('[', k);
    int end = j.indexOf(']', bracket);
    int pos = bracket + 1;
    int tLen = targetId.length();
    while (pos < end) {
      int q1 = j.indexOf('"', pos);
      if (q1 < 0 || q1 >= end) {
        break;
      }
      int q2 = j.indexOf('"', q1 + 1);
      if (q2 - q1 - 1 == tLen && j.regionMatches(q1 + 1, targetId, 0, tLen)) {
        return true;
      }
      pos = q2 + 1;
    }
    return false;
  }

  private static int tsHour(String s) {
    return (s.charAt(11) - '0') * 10 + (s.charAt(12) - '0');
  }

  private static int tsDayOfWeek(String s) {
    int y = digits(s, 0, 4);
    int m = digits(s, 5, 7);
    int d = digits(s, 8, 10);
    if (m < 3) {
      y--;
    }
    int dow = (y + y / 4 - y / 100 + y / 400 + DAY_OF_WEEK_MONTH_TABLE[m - 1] + d) % 7;
    return dow == 0 ? 7 : dow;
  }

  private static long tsEpochSeconds(String s) {
    int y = digits(s, 0, 4);
    int m = digits(s, 5, 7);
    int d = digits(s, 8, 10);
    int h = digits(s, 11, 13);
    int min = digits(s, 14, 16);
    int sec = digits(s, 17, 19);
    if (m <= 2) {
      y--;
      m += 9;
    } else {
      m -= 3;
    }
    long era = (y >= 0 ? y : y - 399) / 400;
    int yoe = (int) (y - era * 400);
    int doy = (153 * m + 2) / 5 + d - 1;
    int doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    long days = era * 146097L + doe - 719468L;
    return days * 86400L + h * 3600L + min * 60L + sec;
  }

  private static int digits(String s, int start, int end) {
    int v = 0;
    for (int i = start; i < end; i++) {
      v = v * 10 + (s.charAt(i) - '0');
    }
    return v;
  }

  public static float dayOfWeek(final int year, final int month, final int day) {
    int adjustedYear = year;
    if (month < 3) {
      adjustedYear--;
    }
    final int rawDayOfWeek = (adjustedYear
                              + adjustedYear / 4
                              - adjustedYear / 100
                              + adjustedYear / 400
                              + DAY_OF_WEEK_MONTH_TABLE[month - 1]
                              + day) % 7;
    return (rawDayOfWeek + 6) % 7;
  }

  private static float clampToUnitRange(float value) {
    return value < 0f ? 0f : (value > 1f ? 1f : value);
  }

  private static float computeMccRiskScore(int merchantCategoryCode) {
    return switch (merchantCategoryCode) {
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
