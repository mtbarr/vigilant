package io.github.mtbarr.rinha.service;

import jakarta.enterprise.context.ApplicationScoped;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;

@ApplicationScoped
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

  private static final byte[] KEY_AMOUNT = encodeJsonKey("amount");
  private static final byte[] KEY_INSTALLMENTS = encodeJsonKey("installments");
  private static final byte[] KEY_REQUESTED_AT = encodeJsonKey("requested_at");
  private static final byte[] KEY_CUSTOMER = encodeJsonKey("customer");
  private static final byte[] KEY_AVG_AMOUNT = encodeJsonKey("avg_amount");
  private static final byte[] KEY_TX_COUNT_24H = encodeJsonKey("tx_count_24h");
  private static final byte[] KEY_MERCHANT = encodeJsonKey("merchant");
  private static final byte[] KEY_MERCHANT_ID = encodeJsonKey("id");
  private static final byte[] KEY_MCC = encodeJsonKey("mcc");
  private static final byte[] KEY_TERMINAL = encodeJsonKey("terminal");
  private static final byte[] KEY_IS_ONLINE = encodeJsonKey("is_online");
  private static final byte[] KEY_CARD_PRESENT = encodeJsonKey("card_present");
  private static final byte[] KEY_KM_FROM_HOME = encodeJsonKey("km_from_home");
  private static final byte[] KEY_KNOWN_MERCHANTS = encodeJsonKey("known_merchants");
  private static final byte[] KEY_LAST_TRANSACTION = encodeJsonKey("last_transaction");
  private static final byte[] KEY_TIMESTAMP = encodeJsonKey("timestamp");
  private static final byte[] KEY_KM_FROM_CURRENT = encodeJsonKey("km_from_current");

  private static final byte CHAR_QUOTE = '"';
  private static final byte CHAR_COLON = ':';
  private static final byte CHAR_OPEN_BRACE = '{';
  private static final byte CHAR_OPEN_BRACKET = '[';
  private static final byte CHAR_CLOSE_BRACKET = ']';
  private static final byte CHAR_MINUS = '-';
  private static final byte CHAR_DOT = '.';
  private static final byte CHAR_DIGIT_ZERO = '0';
  private static final byte CHAR_DIGIT_NINE = '9';

  private final ThreadLocal<float[]> featureVectorBuffer = ThreadLocal.withInitial(
    () -> new float[FEATURE_VECTOR_DIMENSIONS]
  );

  public float[] extractFeatureVector(final byte[] jsonPayload) {
    final float[] featureVector = featureVectorBuffer.get();

    final int amountValueStart = findNumericValueStart(jsonPayload, KEY_AMOUNT, 0);
    final float transactionAmount = parseFloatValue(jsonPayload, amountValueStart);

    final int installmentsValueStart = findNumericValueStart(
      jsonPayload,
      KEY_INSTALLMENTS,
      amountValueStart
    );
    final int installmentCount = parseIntegerValue(jsonPayload, installmentsValueStart);

    final int requestedAtValueStart = findStringValueStart(
      jsonPayload,
      KEY_REQUESTED_AT,
      installmentsValueStart
    );
    final int requestedAtValueEnd = findStringValueEnd(jsonPayload, requestedAtValueStart);
    final int transactionHour = extractHourFromTimestamp(jsonPayload, requestedAtValueStart);
    final int transactionDayOfWeek = extractDayOfWeekFromTimestamp(jsonPayload, requestedAtValueStart);
    final long requestedAtEpochSeconds = parseEpochSecondsFromTimestamp(jsonPayload, requestedAtValueStart);

    final int customerSectionStart = findObjectSectionStart(
      jsonPayload,
      KEY_CUSTOMER,
      requestedAtValueEnd
    );
    final int customerAvgAmountStart = findNumericValueStart(
      jsonPayload,
      KEY_AVG_AMOUNT,
      customerSectionStart
    );
    final float customerAverageAmount = parseFloatValue(jsonPayload, customerAvgAmountStart);
    final int customerTxCount24hStart = findNumericValueStart(
      jsonPayload,
      KEY_TX_COUNT_24H,
      customerSectionStart
    );
    final int customerTransactionCount24h = parseIntegerValue(
      jsonPayload,
      customerTxCount24hStart
    );

    final int merchantSectionStart = findObjectSectionStart(
      jsonPayload,
      KEY_MERCHANT,
      customerSectionStart
    );
    final int merchantIdValueStart = findStringValueStart(
      jsonPayload,
      KEY_MERCHANT_ID,
      merchantSectionStart
    );
    final int merchantIdValueEnd = findStringValueEnd(jsonPayload, merchantIdValueStart);
    final int mccValueStart = findNumericValueStart(jsonPayload, KEY_MCC, merchantSectionStart);
    final int merchantCategoryCode = parseMccValue(jsonPayload, mccValueStart);
    final int merchantAvgAmountStart = findNumericValueStart(
      jsonPayload,
      KEY_AVG_AMOUNT,
      merchantSectionStart
    );
    final float merchantAverageAmount = parseFloatValue(jsonPayload, merchantAvgAmountStart);

    final int terminalSectionStart = findObjectSectionStart(
      jsonPayload,
      KEY_TERMINAL,
      merchantSectionStart
    );
    final int isOnlineValueStart = findNumericValueStart(
      jsonPayload,
      KEY_IS_ONLINE,
      terminalSectionStart
    );
    final boolean isOnlineTransaction = parseBooleanValue(jsonPayload, isOnlineValueStart);
    final int cardPresentValueStart = findNumericValueStart(
      jsonPayload,
      KEY_CARD_PRESENT,
      terminalSectionStart
    );
    final boolean isCardPresent = parseBooleanValue(jsonPayload, cardPresentValueStart);
    final int kmFromHomeValueStart = findNumericValueStart(
      jsonPayload,
      KEY_KM_FROM_HOME,
      terminalSectionStart
    );
    final float distanceFromHomeKm = parseFloatValue(jsonPayload, kmFromHomeValueStart);

    final boolean isUnknownMerchant = !isMerchantInKnownList(
      jsonPayload,
      customerSectionStart,
      merchantIdValueStart,
      merchantIdValueEnd
    );

    final int lastTransactionKeyPos = findByteArrayIndex(
      jsonPayload,
      KEY_LAST_TRANSACTION,
      terminalSectionStart
    );
    final int lastTransactionSectionStart = lastTransactionKeyPos >= 0
      ? findSingleByteIndex(jsonPayload, CHAR_OPEN_BRACE, lastTransactionKeyPos)
      : -1;
    final boolean hasLastTransaction = lastTransactionSectionStart > 0;

    final long minutesSinceLastTransaction;
    final float distanceFromCurrentKm;

    if (hasLastTransaction) {
      final int lastTimestampValueStart = findStringValueStart(
        jsonPayload,
        KEY_TIMESTAMP,
        lastTransactionSectionStart
      );
      final long lastTransactionEpochSeconds = parseEpochSecondsFromTimestamp(
        jsonPayload,
        lastTimestampValueStart
      );
      minutesSinceLastTransaction = Math.max(
        0L,
        requestedAtEpochSeconds - lastTransactionEpochSeconds
      ) / 60L;
      final int kmFromCurrentValueStart = findNumericValueStart(
        jsonPayload,
        KEY_KM_FROM_CURRENT,
        lastTransactionSectionStart
      );
      distanceFromCurrentKm = parseFloatValue(jsonPayload, kmFromCurrentValueStart);
    } else {
      minutesSinceLastTransaction = -1L;
      distanceFromCurrentKm = -1f;
    }

    featureVector[0] = clampToUnitRange(transactionAmount / MAX_TRANSACTION_AMOUNT);
    featureVector[1] = clampToUnitRange(installmentCount / MAX_INSTALLMENT_COUNT);
    featureVector[2] = clampToUnitRange(
      (transactionAmount / customerAverageAmount) / AMOUNT_TO_AVG_RATIO_CAP
    );
    featureVector[3] = transactionHour / 23f;
    featureVector[4] = transactionDayOfWeek / 6f;

    if (hasLastTransaction) {
      featureVector[5] = clampToUnitRange(minutesSinceLastTransaction / MAX_MINUTES_SINCE_LAST_TX);
      featureVector[6] = clampToUnitRange(distanceFromCurrentKm / MAX_DISTANCE_KM);
    } else {
      featureVector[5] = -1f;
      featureVector[6] = -1f;
    }

    featureVector[7] = clampToUnitRange(distanceFromHomeKm / MAX_DISTANCE_KM);
    featureVector[8] = clampToUnitRange(customerTransactionCount24h / MAX_TRANSACTIONS_24H);
    featureVector[9] = isOnlineTransaction ? 1f : 0f;
    featureVector[10] = isCardPresent ? 1f : 0f;
    featureVector[11] = isUnknownMerchant ? 1f : 0f;
    featureVector[12] = computeMccRiskScore(merchantCategoryCode);
    featureVector[13] = clampToUnitRange(merchantAverageAmount / MAX_MERCHANT_AVG_AMOUNT);

    return featureVector;
  }

  private static int findObjectSectionStart(
    final byte[] jsonPayload,
    final byte[] jsonKey,
    final int searchFrom
  ) {
    final int keyPosition = findByteArrayIndex(jsonPayload, jsonKey, searchFrom);
    final int colonPosition = findSingleByteIndex(jsonPayload, CHAR_COLON, keyPosition);
    return findSingleByteIndex(jsonPayload, CHAR_OPEN_BRACE, colonPosition);
  }

  private static int findNumericValueStart(
    final byte[] jsonPayload,
    final byte[] jsonKey,
    final int searchFrom
  ) {
    final int keyPosition = findByteArrayIndex(jsonPayload, jsonKey, searchFrom);
    final int colonPosition = findSingleByteIndex(jsonPayload, CHAR_COLON, keyPosition) + 1;
    int valuePosition = colonPosition;
    while (jsonPayload[valuePosition] <= ' ') {
      valuePosition++;
    }
    return valuePosition;
  }

  private static int findStringValueStart(
    final byte[] jsonPayload,
    final byte[] jsonKey,
    final int searchFrom
  ) {
    final int keyPosition = findByteArrayIndex(jsonPayload, jsonKey, searchFrom);
    final int colonPosition = findSingleByteIndex(jsonPayload, CHAR_COLON, keyPosition) + 1;
    int valuePosition = colonPosition;
    while (jsonPayload[valuePosition] <= ' ') {
      valuePosition++;
    }
    return valuePosition + 1;
  }

  private static int findStringValueEnd(
    final byte[] jsonPayload,
    final int valueStart
  ) {
    return findSingleByteIndex(jsonPayload, CHAR_QUOTE, valueStart);
  }

  private static float parseFloatValue(final byte[] jsonPayload, final int valueStart) {
    int position = valueStart;
    final boolean isNegative = jsonPayload[position] == CHAR_MINUS;
    if (isNegative) {
      position++;
    }

    float integerPart = 0f;
    while (jsonPayload[position] >= CHAR_DIGIT_ZERO
           && jsonPayload[position] <= CHAR_DIGIT_NINE) {
      integerPart = integerPart * 10f + (jsonPayload[position] - CHAR_DIGIT_ZERO);
      position++;
    }

    if (jsonPayload[position] == CHAR_DOT) {
      position++;
      float fractionalPart = 0f;
      float fractionalDivisor = 1f;
      while (jsonPayload[position] >= CHAR_DIGIT_ZERO
             && jsonPayload[position] <= CHAR_DIGIT_NINE) {
        fractionalPart = fractionalPart * 10f + (jsonPayload[position] - CHAR_DIGIT_ZERO);
        fractionalDivisor *= 10f;
        position++;
      }
      integerPart += fractionalPart / fractionalDivisor;
    }

    return isNegative ? -integerPart : integerPart;
  }

  private static int parseIntegerValue(final byte[] jsonPayload, final int valueStart) {
    int position = valueStart;
    int result = 0;
    while (jsonPayload[position] >= CHAR_DIGIT_ZERO
           && jsonPayload[position] <= CHAR_DIGIT_NINE) {
      result = result * 10 + (jsonPayload[position] - CHAR_DIGIT_ZERO);
      position++;
    }
    return result;
  }

  private static int parseMccValue(final byte[] jsonPayload, final int valueStart) {
    int position = valueStart;
    if (jsonPayload[position] == CHAR_QUOTE) {
      position++;
    }
    int result = 0;
    while (jsonPayload[position] >= CHAR_DIGIT_ZERO
           && jsonPayload[position] <= CHAR_DIGIT_NINE) {
      result = result * 10 + (jsonPayload[position] - CHAR_DIGIT_ZERO);
      position++;
    }
    return result;
  }

  private static boolean parseBooleanValue(final byte[] jsonPayload, final int valueStart) {
    return jsonPayload[valueStart] == 't';
  }

  private static boolean isMerchantInKnownList(
    final byte[] jsonPayload,
    final int searchFrom,
    final int targetIdStart,
    final int targetIdEnd
  ) {
    final int knownMerchantsKeyPos = findByteArrayIndex(
      jsonPayload,
      KEY_KNOWN_MERCHANTS,
      searchFrom
    );
    if (knownMerchantsKeyPos < 0) {
      return false;
    }
    final int arrayOpenBracketPos = findSingleByteIndex(
      jsonPayload,
      CHAR_OPEN_BRACKET,
      knownMerchantsKeyPos
    );
    if (arrayOpenBracketPos < 0) {
      return false;
    }
    final int arrayCloseBracketPos = findSingleByteIndex(
      jsonPayload,
      CHAR_CLOSE_BRACKET,
      arrayOpenBracketPos
    );
    if (arrayCloseBracketPos < 0) {
      return false;
    }

    final int targetIdLength = targetIdEnd - targetIdStart;
    int scanPosition = arrayOpenBracketPos + 1;
    while (scanPosition < arrayCloseBracketPos) {
      final int quoteStartPosition = findSingleByteIndex(
        jsonPayload,
        CHAR_QUOTE,
        scanPosition
      );
      if (quoteStartPosition < 0 || quoteStartPosition >= arrayCloseBracketPos) {
        break;
      }
      final int quoteEndPosition = findSingleByteIndex(
        jsonPayload,
        CHAR_QUOTE,
        quoteStartPosition + 1
      );
      if (quoteEndPosition < 0) {
        break;
      }
      final int candidateLength = quoteEndPosition - quoteStartPosition - 1;
      if (candidateLength == targetIdLength
          && byteRegionMatches(
            jsonPayload,
            quoteStartPosition + 1,
            jsonPayload,
            targetIdStart,
            targetIdLength
          )) {
        return true;
      }
      scanPosition = quoteEndPosition + 1;
    }
    return false;
  }

  private static int extractHourFromTimestamp(
    final byte[] timestampBytes,
    final int timestampStart
  ) {
    return (timestampBytes[timestampStart + 11] - CHAR_DIGIT_ZERO) * 10
      + (timestampBytes[timestampStart + 12] - CHAR_DIGIT_ZERO);
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

  private static int extractDayOfWeekFromTimestamp(
    final byte[] timestampBytes,
    final int timestampStart
  ) {
    final int year = parseDigitSequence(timestampBytes, timestampStart, 4);
    final int month = parseDigitSequence(timestampBytes, timestampStart + 5, 2);
    final int day = parseDigitSequence(timestampBytes, timestampStart + 8, 2);
    return (int) dayOfWeek(year, month, day);
  }

  private static long parseEpochSecondsFromTimestamp(
    final byte[] timestampBytes,
    final int timestampStart
  ) {
    final int year = parseDigitSequence(timestampBytes, timestampStart, 4);
    int month = parseDigitSequence(timestampBytes, timestampStart + 5, 2);
    final int day = parseDigitSequence(timestampBytes, timestampStart + 8, 2);
    final int hour = parseDigitSequence(timestampBytes, timestampStart + 11, 2);
    final int minute = parseDigitSequence(timestampBytes, timestampStart + 14, 2);
    final int second = parseDigitSequence(timestampBytes, timestampStart + 17, 2);

    int adjustedYear = year;
    int adjustedMonth = month;
    if (month <= 2) {
      adjustedYear--;
      adjustedMonth += 9;
    } else {
      adjustedMonth -= 3;
    }

    final long era = (adjustedYear >= 0 ? adjustedYear : adjustedYear - 399) / 400L;
    final int yearOfEra = adjustedYear - (int) era * 400;
    final int dayOfYear = (153 * adjustedMonth + 2) / 5 + day - 1;
    final int dayOfEra = yearOfEra * 365
      + yearOfEra / 4
      - yearOfEra / 100
      + dayOfYear;
    final long daysSinceEpoch = era * 146097L + dayOfEra - 719468L;

    return daysSinceEpoch * 86400L
      + hour * 3600L
      + minute * 60L
      + second;
  }

  private static int parseDigitSequence(
    final byte[] textBytes,
    final int sequenceStart,
    final int digitCount
  ) {
    int value = 0;
    for (int i = 0; i < digitCount; i++) {
      value = value * 10 + (textBytes[sequenceStart + i] - CHAR_DIGIT_ZERO);
    }
    return value;
  }

  private static float clampToUnitRange(final float value) {
    return value < 0f ? 0f : (value > 1f ? 1f : value);
  }

  private static float computeMccRiskScore(final int merchantCategoryCode) {
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

  private static int findSingleByteIndex(
    final byte[] data,
    final int targetByte,
    final int fromIndex
  ) {
    for (int i = fromIndex; i < data.length; i++) {
      if (data[i] == targetByte) {
        return i;
      }
    }
    return -1;
  }

  private static int findByteArrayIndex(
    final byte[] data,
    final byte[] pattern,
    final int fromIndex
  ) {
    final int patternLength = pattern.length;
    final int searchLimit = data.length - patternLength;
    for (int i = fromIndex; i <= searchLimit; i++) {
      if (data[i] == pattern[0] && Arrays.equals(data, i, i + patternLength, pattern, 0, patternLength)) {
        return i;
      }
    }
    return -1;
  }

  private static boolean byteRegionMatches(
    final byte[] sourceArray,
    final int sourceOffset,
    final byte[] targetArray,
    final int targetOffset,
    final int matchLength
  ) {
    return Arrays.equals(sourceArray, sourceOffset, sourceOffset + matchLength,
                         targetArray, targetOffset, targetOffset + matchLength);
  }

  private static byte[] encodeJsonKey(final String keyName) {
    final byte[] encodedKey = new byte[keyName.length() + 2];
    encodedKey[0] = CHAR_QUOTE;
    System.arraycopy(
      keyName.getBytes(StandardCharsets.US_ASCII),
      0,
      encodedKey,
      1,
      keyName.length()
    );
    encodedKey[encodedKey.length - 1] = CHAR_QUOTE;
    return encodedKey;
  }
}
