package io.github.mtbarr.rinha.service;

import jakarta.inject.Singleton;
import java.nio.charset.StandardCharsets;

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

  private static final byte[] K_TRANSACTION = "\"transaction\"".getBytes(StandardCharsets.US_ASCII);
  private static final byte[] K_CUSTOMER = "\"customer\"".getBytes(StandardCharsets.US_ASCII);
  private static final byte[] K_MERCHANT = "\"merchant\"".getBytes(StandardCharsets.US_ASCII);
  private static final byte[] K_TERMINAL = "\"terminal\"".getBytes(StandardCharsets.US_ASCII);
  private static final byte[] K_AMOUNT = "\"amount\"".getBytes(StandardCharsets.US_ASCII);
  private static final byte[] K_INSTALLMENTS = "\"installments\"".getBytes(StandardCharsets.US_ASCII);
  private static final byte[] K_REQUESTED_AT = "\"requested_at\"".getBytes(StandardCharsets.US_ASCII);
  private static final byte[] K_AVG_AMOUNT = "\"avg_amount\"".getBytes(StandardCharsets.US_ASCII);
  private static final byte[] K_TX_COUNT_24H = "\"tx_count_24h\"".getBytes(StandardCharsets.US_ASCII);
  private static final byte[] K_ID = "\"id\"".getBytes(StandardCharsets.US_ASCII);
  private static final byte[] K_MCC = "\"mcc\"".getBytes(StandardCharsets.US_ASCII);
  private static final byte[] K_IS_ONLINE = "\"is_online\"".getBytes(StandardCharsets.US_ASCII);
  private static final byte[] K_CARD_PRESENT = "\"card_present\"".getBytes(StandardCharsets.US_ASCII);
  private static final byte[] K_KM_FROM_HOME = "\"km_from_home\"".getBytes(StandardCharsets.US_ASCII);
  private static final byte[] K_LAST_TRANSACTION = "\"last_transaction\"".getBytes(StandardCharsets.US_ASCII);
  private static final byte[] K_TIMESTAMP = "\"timestamp\"".getBytes(StandardCharsets.US_ASCII);
  private static final byte[] K_KM_FROM_CURRENT = "\"km_from_current\"".getBytes(StandardCharsets.US_ASCII);
  private static final byte[] K_KNOWN_MERCHANTS = "\"known_merchants\"".getBytes(StandardCharsets.US_ASCII);

  private final ThreadLocal<float[]> featureVectorBuffer = ThreadLocal.withInitial(
    () -> new float[FEATURE_VECTOR_DIMENSIONS]
  );

  public float[] extractFeatureVector(final byte[] j) {
    final float[] fv = featureVectorBuffer.get();

    final int tx = sectionStart(j, 0, K_TRANSACTION);
    final float txAmt = extractFloat(j, tx, K_AMOUNT);
    final int txInst = (int) extractFloat(j, tx, K_INSTALLMENTS);
    final int tsIdx = keyPos(j, tx, K_REQUESTED_AT) + K_REQUESTED_AT.length;
    final int tsColon = colonPos(j, tsIdx);
    final int tsEnd = valueEndStr(j, tsColon + 1);
    final int txHour = digits(j, tsColon + 2, 2);
    final int txDow = tsDayOfWeek(j, tsColon + 2);
    final long txEpoch = tsEpochSeconds(j, tsColon + 2);

    final int cust = sectionStart(j, 0, K_CUSTOMER);
    final float custAvg = extractFloat(j, cust, K_AVG_AMOUNT);
    final int custTxN = (int) extractFloat(j, cust, K_TX_COUNT_24H);

    final int merch = sectionStart(j, 0, K_MERCHANT);
    final int merchIdStart = strStart(j, merch, K_ID) + 1;
    final int merchIdEnd = strEnd(j, merch, K_ID);
    final int mcc = extractIntStr(j, merch, K_MCC);
    final float merchAvg = extractFloat(j, merch, K_AVG_AMOUNT);

    final int term = sectionStart(j, 0, K_TERMINAL);
    final boolean isOnline = extractBool(j, term, K_IS_ONLINE);
    final boolean cardPresent = extractBool(j, term, K_CARD_PRESENT);
    final float kmHome = extractFloat(j, term, K_KM_FROM_HOME);

    final int ltIdx = byteIndexOf(j, 0, K_LAST_TRANSACTION);
    long minsSinceLast = -1L;
    float kmFromCurrent = -1f;
    if (ltIdx >= 0) {
      final int colon = colonPos(j, ltIdx + K_LAST_TRANSACTION.length);
      final int brace = skipWs(j, colon + 1);
      if (j[brace] == '{') {
        final long lastEpoch = tsEpochSeconds(j, strStart(j, brace, K_TIMESTAMP) + K_TIMESTAMP.length);
        kmFromCurrent = extractFloat(j, brace, K_KM_FROM_CURRENT);
        minsSinceLast = Math.max(0L, txEpoch - lastEpoch) / 60L;
      }
    }

    fv[0] = clamp(txAmt / MAX_TRANSACTION_AMOUNT);
    fv[1] = clamp(txInst / MAX_INSTALLMENT_COUNT);
    fv[2] = custAvg > 0f ? clamp((txAmt / custAvg) / AMOUNT_TO_AVG_RATIO_CAP) : 0f;
    fv[3] = txHour / 23f;
    fv[4] = (txDow - 1) / 6f;

    if (ltIdx >= 0 && minsSinceLast >= 0) {
      fv[5] = clamp(minsSinceLast / MAX_MINUTES_SINCE_LAST_TX);
      fv[6] = clamp(kmFromCurrent / MAX_DISTANCE_KM);
    } else {
      fv[5] = -1f;
      fv[6] = -1f;
    }

    fv[7] = clamp(kmHome / MAX_DISTANCE_KM);
    fv[8] = clamp(custTxN / MAX_TRANSACTIONS_24H);
    fv[9] = isOnline ? 1f : 0f;
    fv[10] = cardPresent ? 1f : 0f;
    fv[11] = merchantIsKnown(j, cust, K_KNOWN_MERCHANTS, merchIdStart, merchIdEnd) ? 0f : 1f;
    fv[12] = mccRisk(mcc);
    fv[13] = clamp(merchAvg / MAX_MERCHANT_AVG_AMOUNT);

    return fv;
  }

  private static int keyPos(final byte[] j, final int from, final byte[] key) {
    return byteIndexOf(j, from, key);
  }

  private static int colonPos(final byte[] j, final int from) {
    int i = from;
    while (j[i] != ':') {
      i++;
    }
    return i;
  }

  private static int skipWs(final byte[] j, int pos) {
    byte c;
    while ((c = j[pos]) == ' ' || c == '\n' || c == '\r') {
      pos++;
    }
    return pos;
  }

  private static int sectionStart(final byte[] j, final int from, final byte[] key) {
    final int k = byteIndexOf(j, from, key);
    if (k < 0) {
      return -1;
    }
    final int colon = colonPos(j, k + key.length);
    return skipWs(j, colon + 1);
  }

  private static int valueEndStr(final byte[] j, int pos) {
    pos = skipWs(j, pos);
    if (j[pos] != '"') {
      return pos;
    }
    int i = pos + 1;
    while (i < j.length && j[i] != '"') {
      i++;
    }
    return i;
  }

  private static int strStart(final byte[] j, final int from, final byte[] key) {
    final int k = byteIndexOf(j, from, key);
    if (k < 0) {
      return -1;
    }
    int pos = skipWs(j, colonPos(j, k + key.length) + 1);
    return j[pos] == '"' ? pos : pos - 1;
  }

  private static int strEnd(final byte[] j, final int from, final byte[] key) {
    final int k = byteIndexOf(j, from, key);
    if (k < 0) {
      return -1;
    }
    final int colon = colonPos(j, k + key.length);
    return valueEndStr(j, colon + 1);
  }

  private static float extractFloat(final byte[] j, int from, final byte[] key) {
    final int k = byteIndexOf(j, from, key);
    if (k < 0) {
      return 0f;
    }
    int start = skipWs(j, colonPos(j, k + key.length) + 1);
    int end = start;
    byte c;
    while (end < j.length && (c = j[end]) != ',' && c != '}' && c != '\n' && c != '\r') {
      end++;
    }
    return parseFloat(j, start, end);
  }

  private static int extractIntStr(final byte[] j, int from, final byte[] key) {
    final int k = byteIndexOf(j, from, key);
    if (k < 0) {
      return 0;
    }
    int q1 = skipWs(j, colonPos(j, k + key.length) + 1);
    if (j[q1] == '"') {
      q1++;
    }
    int v = 0;
    while (q1 < j.length && j[q1] != '"') {
      v = v * 10 + (j[q1] - '0');
      q1++;
    }
    return v;
  }

  private static boolean extractBool(final byte[] j, int from, final byte[] key) {
    final int k = byteIndexOf(j, from, key);
    if (k < 0) {
      return false;
    }
    return j[skipWs(j, colonPos(j, k + key.length) + 1)] == 't';
  }

  private static boolean merchantIsKnown(final byte[] j, int from, final byte[] key, final int idStart, final int idEnd) {
    final int k = byteIndexOf(j, from, key);
    if (k < 0) {
      return false;
    }
    final int bracket = byteIndexOf(j, k + key.length, new byte[]{'['});
    if (bracket < 0) {
      return false;
    }
    final int close = byteIndexOf(j, bracket + 1, new byte[]{']'});
    final int idLen = idEnd - idStart;
    int pos = bracket + 1;
    while (pos < close) {
      int q1 = byteIndexOf(j, pos, new byte[]{'"'});
      if (q1 < 0 || q1 >= close) {
        break;
      }
      int q2 = byteIndexOf(j, q1 + 1, new byte[]{'"'});
      if (q2 - q1 - 1 == idLen && bytesEqual(j, q1 + 1, j, idStart, idLen)) {
        return true;
      }
      pos = q2 + 1;
    }
    return false;
  }

  private static int tsDayOfWeek(final byte[] j, final int tsStart) {
    final int y = digits(j, tsStart, 4);
    final int m = digits(j, tsStart + 5, 2);
    final int d = digits(j, tsStart + 8, 2);
    final int adjY = m < 3 ? y - 1 : y;
    final int adjM = m < 3 ? m + 9 : m - 3;
    final int era = adjY >= 0 ? adjY / 400 : (adjY - 399) / 400;
    final int yoe = adjY - era * 400;
    final int doy = (153 * adjM + 2) / 5 + d - 1;
    final int doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    final int days = era * 146097 + doe - 719468;
    int dow = days % 7;
    if (dow <= 0) {
      dow += 7;
    }
    return dow;
  }

  private static long tsEpochSeconds(final byte[] j, final int tsStart) {
    final int y = digits(j, tsStart, 4);
    final int mo = digits(j, tsStart + 5, 2);
    final int d = digits(j, tsStart + 8, 2);
    final int h = digits(j, tsStart + 11, 2);
    final int min = digits(j, tsStart + 14, 2);
    final int sec = digits(j, tsStart + 17, 2);
    int m = mo;
    int yr = y;
    if (m <= 2) {
      yr--;
      m += 9;
    } else {
      m -= 3;
    }
    final long era = (yr >= 0 ? yr : yr - 399) / 400;
    final int yoe = (int) (yr - era * 400);
    final int doy = (153 * m + 2) / 5 + d - 1;
    final int doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    final long days = era * 146097L + doe - 719468L;
    return days * 86400L + h * 3600L + min * 60L + sec;
  }

  private static int digits(final byte[] j, int start, final int count) {
    int v = 0;
    final int end = start + count;
    for (int i = start; i < end; i++) {
      v = v * 10 + (j[i] - '0');
    }
    return v;
  }

  private static float parseFloat(final byte[] j, int start, final int end) {
    float result = 0f;
    boolean neg = false;
    if (start < end && j[start] == '-') {
      neg = true;
      start++;
    }
    while (start < end && j[start] >= '0' && j[start] <= '9') {
      result = result * 10f + (j[start] - '0');
      start++;
    }
    if (start < end && j[start] == '.') {
      start++;
      float frac = 0f;
      float div = 1f;
      while (start < end && j[start] >= '0' && j[start] <= '9') {
        frac = frac * 10f + (j[start] - '0');
        div *= 10f;
        start++;
      }
      result += frac / div;
    }
    return neg ? -result : result;
  }

  private static int byteIndexOf(final byte[] data, int from, final byte[] pattern) {
    final int limit = data.length - pattern.length;
    for (int i = from; i <= limit; i++) {
      boolean match = true;
      for (int j = 0; j < pattern.length; j++) {
        if (data[i + j] != pattern[j]) {
          match = false;
          break;
        }
      }
      if (match) {
        return i;
      }
    }
    return -1;
  }

  private static boolean bytesEqual(final byte[] a, final int aOff, final byte[] b, final int bOff, final int len) {
    for (int i = 0; i < len; i++) {
      if (a[aOff + i] != b[bOff + i]) {
        return false;
      }
    }
    return true;
  }

  private static float clamp(final float v) {
    return v < 0f ? 0f : (v > 1f ? 1f : v);
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
