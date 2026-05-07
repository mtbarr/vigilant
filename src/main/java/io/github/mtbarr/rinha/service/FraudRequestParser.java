package io.github.mtbarr.rinha.service;

import jakarta.enterprise.context.ApplicationScoped;
import java.nio.charset.StandardCharsets;

@ApplicationScoped
public class FraudRequestParser {

  private static final byte[] FIELD_AMOUNT = encode("amount");
  private static final byte[] FIELD_INSTALLMENTS = encode("installments");
  private static final byte[] FIELD_REQUESTED_AT = encode("requested_at");
  private static final byte[] FIELD_AVG_AMOUNT = encode("avg_amount");
  private static final byte[] FIELD_TX_COUNT_24H = encode("tx_count_24h");
  private static final byte[] FIELD_KNOWN_MERCHANTS = encode("known_merchants");
  private static final byte[] FIELD_MCC = encode("mcc");
  private static final byte[] FIELD_ID = encode("id");
  private static final byte[] FIELD_IS_ONLINE = encode("is_online");
  private static final byte[] FIELD_CARD_PRESENT = encode("card_present");
  private static final byte[] FIELD_KM_FROM_HOME = encode("km_from_home");
  private static final byte[] FIELD_TIMESTAMP = encode("timestamp");
  private static final byte[] FIELD_KM_FROM_CURRENT = encode("km_from_current");

  private static final float NORMALIZATION_MAX_AMOUNT = 10_000f;
  private static final float NORMALIZATION_MAX_INSTALLMENTS = 12f;
  private static final float NORMALIZATION_AMOUNT_VS_AVERAGE_RATIO = 10f;
  private static final float NORMALIZATION_MAX_MINUTES = 1440f;
  private static final float NORMALIZATION_MAX_DISTANCE = 1000f;
  private static final float NORMALIZATION_MAX_TRANSACTION_COUNT = 20f;
  private static final float NORMALIZATION_MERCHANT_AMOUNT_DIVISOR = 10_000f;

  private static byte[] encode(final String text) {
    return text.getBytes(StandardCharsets.US_ASCII);
  }

  public float[] extractVector(final byte[] requestBody) {
    final ByteCursor cursor = new ByteCursor(requestBody);

    cursor.skipFieldValue();

    cursor.skipToField(FIELD_AMOUNT);
    final float transactionAmount = cursor.parseFloat();

    cursor.skipToField(FIELD_INSTALLMENTS);
    final int installmentCount = cursor.parseInt();

    cursor.skipToField(FIELD_REQUESTED_AT);
    final int transactionYear = cursor.parseYear();
    final int transactionMonth = cursor.parseTwoDigits();
    cursor.advance();
    final int transactionDay = cursor.parseTwoDigits();
    cursor.advance();
    final int transactionHour = cursor.parseTwoDigits();
    cursor.advance();
    final int transactionMinute = cursor.parseTwoDigits();
    cursor.skipStringEnd();

    cursor.skipToField(FIELD_AVG_AMOUNT);
    final float customerAverageAmount = cursor.parseFloat();

    cursor.skipToField(FIELD_TX_COUNT_24H);
    final int recentTransactionCount = cursor.parseInt();

    cursor.skipToField(FIELD_KNOWN_MERCHANTS);
    cursor.skipToArrayStart();
    final int merchantListStart = cursor.position();
    cursor.skipToArrayEnd();
    final int merchantListEnd = cursor.position();

    cursor.skipToField(FIELD_ID);
    final byte[] currentMerchantId = cursor.parseString();

    cursor.skipToField(FIELD_MCC);
    final int merchantCategoryCode = cursor.parseQuotedInt();

    cursor.skipToField(FIELD_AVG_AMOUNT);
    final float merchantAverageAmount = cursor.parseFloat();

    cursor.skipToField(FIELD_IS_ONLINE);
    final boolean isOnlineTransaction = cursor.parseBoolean();

    cursor.skipToField(FIELD_CARD_PRESENT);
    final boolean isCardPresent = cursor.parseBoolean();

    cursor.skipToField(FIELD_KM_FROM_HOME);
    final float homeDistanceKilometers = cursor.parseFloat();

    final boolean hasPreviousTransaction;
    final int minutesSincePrevious;
    final float distanceFromPrevious;

    if (cursor.skipToOptionalField(FIELD_TIMESTAMP)) {
      parseTransactionTimestamp(cursor);
    } else {
      cursor.skipPastNull();
    }
    hasPreviousTransaction = cursor.hasValue();

    if (hasPreviousTransaction) {
      final int previousYear = cursor.parseYear();
      final int previousMonth = cursor.parseTwoDigits();
      cursor.advance();
      final int previousDay = cursor.parseTwoDigits();
      cursor.advance();
      final int previousHour = cursor.parseTwoDigits();
      cursor.advance();
      final int previousMinute = cursor.parseTwoDigits();
      cursor.skipStringEnd();

      cursor.skipToField(FIELD_KM_FROM_CURRENT);
      distanceFromPrevious = cursor.parseFloat();

      minutesSincePrevious = computeMinutesBetween(
        previousYear, previousMonth, previousDay, previousHour, previousMinute,
        transactionYear, transactionMonth, transactionDay, transactionHour, transactionMinute
      );
    } else {
      minutesSincePrevious = 0;
      distanceFromPrevious = 0f;
    }

    final boolean isUnknownMerchant = !isMerchantInList(
      requestBody, merchantListStart, merchantListEnd, currentMerchantId);

    return assembleFeatureVector(
      transactionAmount, installmentCount, customerAverageAmount,
      transactionHour, dayOfWeek(transactionYear, transactionMonth, transactionDay),
      hasPreviousTransaction, minutesSincePrevious, distanceFromPrevious,
      homeDistanceKilometers, recentTransactionCount,
      isOnlineTransaction, isCardPresent, isUnknownMerchant,
      merchantCategoryCode, merchantAverageAmount
    );
  }

  private void parseTransactionTimestamp(final ByteCursor cursor) {
    cursor.markHasValue();
  }

  private static float[] assembleFeatureVector(
    final float amount, final int installments, final float customerAverage,
    final int hour, final float dayOfWeek,
    final boolean hasPrevious, final int minutesSincePrevious, final float kilometersFromPrevious,
    final float kilometersFromHome, final int transactionCount24h,
    final boolean isOnline, final boolean isCardPresent,
    final boolean isUnknownMerchant, final int merchantCategoryCode,
    final float merchantAverageAmount) {

    final float[] featureVector = new float[14];

    featureVector[0] = clamp(amount / NORMALIZATION_MAX_AMOUNT);
    featureVector[1] = clamp(installments / NORMALIZATION_MAX_INSTALLMENTS);
    featureVector[2] = clamp((amount / customerAverage) / NORMALIZATION_AMOUNT_VS_AVERAGE_RATIO);
    featureVector[3] = clamp(hour / 23f);
    featureVector[4] = clamp(dayOfWeek / 6f);

    if (hasPrevious) {
      featureVector[5] = clamp(minutesSincePrevious / NORMALIZATION_MAX_MINUTES);
      featureVector[6] = clamp(kilometersFromPrevious / NORMALIZATION_MAX_DISTANCE);
    } else {
      featureVector[5] = -1f;
      featureVector[6] = -1f;
    }

    featureVector[7] = clamp(kilometersFromHome / NORMALIZATION_MAX_DISTANCE);
    featureVector[8] = clamp(transactionCount24h / NORMALIZATION_MAX_TRANSACTION_COUNT);
    featureVector[9] = isOnline ? 1f : 0f;
    featureVector[10] = isCardPresent ? 1f : 0f;
    featureVector[11] = isUnknownMerchant ? 1f : 0f;
    featureVector[12] = mccRiskFor(merchantCategoryCode);
    featureVector[13] = clamp(merchantAverageAmount / NORMALIZATION_MERCHANT_AMOUNT_DIVISOR);

    return featureVector;
  }


  private static boolean isMerchantInList(final byte[] source,
                                          final int rangeStart, final int rangeEnd,
                                          final byte[] targetMerchant) {
    int position = rangeStart;
    while (position < rangeEnd) {
      if (source[position] == '"') {
        position++;
        final int valueStart = position;
        while (position < rangeEnd && source[position] != '"') {
          position++;
        }
        if (bytesMatch(source, valueStart, position, targetMerchant)) {
          return true;
        }
        position++;
      } else {
        position++;
      }
    }
    return false;
  }

  private static boolean bytesMatch(final byte[] source, final int sourceStart,
                                    final int sourceEnd, final byte[] target) {
    final int length = sourceEnd - sourceStart;
    if (length != target.length) {
      return false;
    }
    for (int index = 0; index < length; index++) {
      if (source[sourceStart + index] != target[index]) {
        return false;
      }
    }
    return true;
  }


  private static float clamp(final float value) {
    if (value < 0f) {
      return 0f;
    }
    if (value > 1f) {
      return 1f;
    }
    return value;
  }

  static float mccRiskFor(final int mccCode) {
    return switch (mccCode) {
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


  static float dayOfWeek(final int year, final int month, final int day) {
    final int yearOffset = month < 3 ? year - 1 : year;
    final int monthOffset = month < 3 ? month + 12 : month;
    final int century = yearOffset / 100;
    final int yearInCentury = yearOffset % 100;
    final int zellerResult = (day + (13 * (monthOffset + 1)) / 5 + yearInCentury
                              + yearInCentury / 4 + century / 4 - 2 * century) % 7;
    return (zellerResult + 6) % 7;
  }

  static int computeMinutesBetween(
    final int year1, final int month1, final int day1,
    final int hour1, final int minute1,
    final int year2, final int month2, final int day2,
    final int hour2, final int minute2) {
    final long epochDays1 = daysSinceEpoch(year1, month1, day1);
    final long epochDays2 = daysSinceEpoch(year2, month2, day2);
    final long totalMinutes1 = epochDays1 * 1440 + hour1 * 60L + minute1;
    final long totalMinutes2 = epochDays2 * 1440 + hour2 * 60L + minute2;
    final long difference = totalMinutes2 - totalMinutes1;
    return difference > 0 ? (int) difference : 0;
  }

  private static long daysSinceEpoch(final int year, final int month, final int day) {
    final int yearOffset = month <= 2 ? year - 1 : year;
    final int era = yearOffset >= 0 ? yearOffset / 400 : (yearOffset - 399) / 400;
    final int yearOfEra = yearOffset - era * 400;
    final int monthOffset = month > 2 ? month - 3 : month + 9;
    final int dayOfYear = (153 * monthOffset + 2) / 5 + day - 1;
    final int dayOfEra = yearOfEra * 365 + yearOfEra / 4 - yearOfEra / 100 + dayOfYear;
    return (long) era * 146097 + dayOfEra - 719468;
  }


  private static final class ByteCursor {
    private static final byte QUOTE = '"';
    private static final byte COLON = ':';
    private static final byte COMMA = ',';
    private static final byte MINUS = '-';
    private static final byte DOT = '.';
    private static final byte OPEN_BRACKET = '[';
    private static final byte CLOSE_BRACKET = ']';
    private static final byte NULL_LOWER = 'n';
    private static final byte NULL_UPPER = 'N';
    private static final byte TRUE = 't';

    private final byte[] source;
    private int offset;
    private boolean hasValue;

    ByteCursor(final byte[] source) {
      this.source = source;
    }

    int position() {
      return offset;
    }

    void advance() {
      offset++;
    }

    void markHasValue() {
      hasValue = true;
    }

    boolean hasValue() {
      return hasValue;
    }

    void skipPastNull() {
      hasValue = false;
      if (offset < source.length) {
        if (source[offset] == NULL_LOWER || source[offset] == NULL_UPPER) {
          offset += 4;
        }
      }
    }


    void skipToField(final byte[] fieldName) {
      while (offset < source.length) {
        if (source[offset] == QUOTE) {
          offset++;
          if (startsWith(offset, fieldName)) {
            offset += fieldName.length;
            skipTo(COLON);
            offset++;
            skipWhitespace();
            return;
          }
          skipStringContents();
        } else {
          offset++;
        }
      }
    }

    boolean skipToOptionalField(final byte[] fieldName) {
      int savedPosition = offset;
      while (offset < source.length) {
        if (source[offset] == QUOTE) {
          offset++;
          if (startsWith(offset, fieldName)) {
            offset += fieldName.length;
            offset++;
            skipWhitespace();
            return true;
          }
          skipStringContents();
        } else if (source[offset] == NULL_LOWER || source[offset] == NULL_UPPER) {
          offset = savedPosition;
          return false;
        } else {
          offset++;
        }
      }
      offset = savedPosition;
      return false;
    }

    void skipFieldValue() {
      skipTo(QUOTE);
      offset++;
      skipStringContents();
      offset++;
    }

    void skipStringEnd() {
      skipTo(QUOTE);
      offset++;
    }

    void skipToArrayStart() {
      skipTo(OPEN_BRACKET);
      offset++;
    }

    void skipToArrayEnd() {
      while (offset < source.length && source[offset] != CLOSE_BRACKET) {
        offset++;
      }
    }

    void skipStringContents() {
      while (offset < source.length && source[offset] != QUOTE) {
        offset++;
      }
    }


    int parseInt() {
      skipWhitespace();
      boolean isNegative = source[offset] == MINUS;
      if (isNegative) {
        offset++;
      }
      int value = 0;
      while (offset < source.length && isDigit(source[offset])) {
        value = value * 10 + (source[offset] - '0');
        offset++;
      }
      return isNegative ? -value : value;
    }

    float parseFloat() {
      skipWhitespace();
      boolean isNegative = source[offset] == MINUS;
      if (isNegative) {
        offset++;
      }

      long integerPart = 0;
      while (offset < source.length && isDigit(source[offset])) {
        integerPart = integerPart * 10 + (source[offset] - '0');
        offset++;
      }

      float result = integerPart;

      if (offset < source.length && source[offset] == DOT) {
        offset++;
        long fractionPart = 0;
        int fractionDigits = 0;
        while (offset < source.length && isDigit(source[offset])) {
          fractionPart = fractionPart * 10 + (source[offset] - '0');
          fractionDigits++;
          offset++;
        }
        float divisor = 1f;
        for (int power = 0; power < fractionDigits; power++) {
          divisor *= 10f;
        }
        result += fractionPart / divisor;
      }

      return isNegative ? -result : result;
    }

    boolean parseBoolean() {
      skipWhitespace();
      if (source[offset] == TRUE) {
        offset += 4;
        return true;
      }
      offset += 5;
      return false;
    }

    byte[] parseString() {
      skipWhitespace();
      if (source[offset] == QUOTE) {
        offset++;
      }
      final int start = offset;
      skipStringContents();
      final byte[] result = java.util.Arrays.copyOfRange(source, start, offset);
      if (offset < source.length && source[offset] == QUOTE) {
        offset++;
      }
      return result;
    }

    int parseYear() {
      skipWhitespace();
      if (source[offset] == QUOTE) {
        offset++;
      }
      return readDigits(4);
    }

    int parseTwoDigits() {
      return readDigits(2);
    }

    int parseQuotedInt() {
      skipWhitespace();
      if (source[offset] == QUOTE) {
        offset++;
      }
      int value = 0;
      while (offset < source.length && isDigit(source[offset])) {
        value = value * 10 + (source[offset] - '0');
        offset++;
      }
      if (offset < source.length && source[offset] == QUOTE) {
        offset++;
      }
      return value;
    }


    private boolean startsWith(final int start, final byte[] pattern) {
      if (start + pattern.length > source.length) {
        return false;
      }
      for (int index = 0; index < pattern.length; index++) {
        if (source[start + index] != pattern[index]) {
          return false;
        }
      }
      return true;
    }

    private int readDigits(final int count) {
      int value = 0;
      final int limit = Math.min(offset + count, source.length);
      while (offset < limit && isDigit(source[offset])) {
        value = value * 10 + (source[offset] - '0');
        offset++;
      }
      return value;
    }

    private void skipTo(final byte target) {
      while (offset < source.length && source[offset] != target) {
        offset++;
      }
    }

    private void skipWhitespace() {
      while (offset < source.length && isWhitespace(source[offset])) {
        offset++;
      }
    }
  }

  private static boolean isWhitespace(final byte value) {
    return value == ' ' || value == '\t' || value == '\n' || value == '\r';
  }

  private static boolean isDigit(final byte value) {
    return value >= '0' && value <= '9';
  }
}
