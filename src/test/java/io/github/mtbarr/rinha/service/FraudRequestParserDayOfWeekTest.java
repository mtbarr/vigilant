package io.github.mtbarr.rinha.service;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.assertEquals;

class FraudRequestParserDayOfWeekTest {

  private final FraudRequestParser parser = new FraudRequestParser();

  // Monday=0, Tuesday=1, Wednesday=2, Thursday=3, Friday=4, Saturday=5, Sunday=6

  @Test
  void mondayShouldBeZero() {
    // 2026-03-09 is a Monday
    assertEquals(0f, FraudRequestParser.dayOfWeek(2026, 3, 9));
    // 2026-01-05 is a Monday
    assertEquals(0f, FraudRequestParser.dayOfWeek(2026, 1, 5));
  }

  @Test
  void tuesdayShouldBeOne() {
    // 2026-03-10 is a Tuesday
    assertEquals(1f, FraudRequestParser.dayOfWeek(2026, 3, 10));
    // 2026-01-06 is a Tuesday
    assertEquals(1f, FraudRequestParser.dayOfWeek(2026, 1, 6));
  }

  @Test
  void wednesdayShouldBeTwo() {
    // 2026-03-11 is a Wednesday
    assertEquals(2f, FraudRequestParser.dayOfWeek(2026, 3, 11));
  }

  @Test
  void thursdayShouldBeThree() {
    // 2026-03-12 is a Thursday
    assertEquals(3f, FraudRequestParser.dayOfWeek(2026, 3, 12));
  }

  @Test
  void fridayShouldBeFour() {
    // 2026-03-13 is a Friday
    assertEquals(4f, FraudRequestParser.dayOfWeek(2026, 3, 13));
  }

  @Test
  void saturdayShouldBeFive() {
    // 2026-03-14 is a Saturday
    assertEquals(5f, FraudRequestParser.dayOfWeek(2026, 3, 14));
  }

  @Test
  void sundayShouldBeSix() {
    // 2026-03-08 is a Sunday
    assertEquals(6f, FraudRequestParser.dayOfWeek(2026, 3, 8));
    // 2026-01-04 is a Sunday
    assertEquals(6f, FraudRequestParser.dayOfWeek(2026, 1, 4));
  }

  @Test
  void january1st2026ShouldBeThursday() {
    // 2026-01-01 is a Thursday
    assertEquals(3f, FraudRequestParser.dayOfWeek(2026, 1, 1));
  }

  @Test
  void december31st2026ShouldBeThursday() {
    // 2026-12-31 is a Thursday
    assertEquals(3f, FraudRequestParser.dayOfWeek(2026, 12, 31));
  }

  @Test
  void leapYear2024February29ShouldBeThursday() {
    // 2024-02-29 is a Thursday
    assertEquals(3f, FraudRequestParser.dayOfWeek(2024, 2, 29));
  }

  @Test
  void march1st2026ShouldBeSunday() {
    // 2026-03-01 is a Sunday
    assertEquals(6f, FraudRequestParser.dayOfWeek(2026, 3, 1));
  }
}
