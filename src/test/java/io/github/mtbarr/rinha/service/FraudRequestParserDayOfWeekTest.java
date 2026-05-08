package io.github.mtbarr.rinha.service;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.assertEquals;

class FraudRequestParserDayOfWeekTest {

  private final FraudRequestParser parser = new FraudRequestParser();

  

  @Test
  void mondayShouldBeZero() {
    
    assertEquals(0f, FraudRequestParser.dayOfWeek(2026, 3, 9));
    
    assertEquals(0f, FraudRequestParser.dayOfWeek(2026, 1, 5));
  }

  @Test
  void tuesdayShouldBeOne() {
    
    assertEquals(1f, FraudRequestParser.dayOfWeek(2026, 3, 10));
    
    assertEquals(1f, FraudRequestParser.dayOfWeek(2026, 1, 6));
  }

  @Test
  void wednesdayShouldBeTwo() {
    
    assertEquals(2f, FraudRequestParser.dayOfWeek(2026, 3, 11));
  }

  @Test
  void thursdayShouldBeThree() {
    
    assertEquals(3f, FraudRequestParser.dayOfWeek(2026, 3, 12));
  }

  @Test
  void fridayShouldBeFour() {
    
    assertEquals(4f, FraudRequestParser.dayOfWeek(2026, 3, 13));
  }

  @Test
  void saturdayShouldBeFive() {
    
    assertEquals(5f, FraudRequestParser.dayOfWeek(2026, 3, 14));
  }

  @Test
  void sundayShouldBeSix() {
    
    assertEquals(6f, FraudRequestParser.dayOfWeek(2026, 3, 8));
    
    assertEquals(6f, FraudRequestParser.dayOfWeek(2026, 1, 4));
  }

  @Test
  void january1st2026ShouldBeThursday() {
    
    assertEquals(3f, FraudRequestParser.dayOfWeek(2026, 1, 1));
  }

  @Test
  void december31st2026ShouldBeThursday() {
    
    assertEquals(3f, FraudRequestParser.dayOfWeek(2026, 12, 31));
  }

  @Test
  void leapYear2024February29ShouldBeThursday() {
    
    assertEquals(3f, FraudRequestParser.dayOfWeek(2024, 2, 29));
  }

  @Test
  void march1st2026ShouldBeSunday() {
    
    assertEquals(6f, FraudRequestParser.dayOfWeek(2026, 3, 1));
  }
}
