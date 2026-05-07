package io.github.mtbarr.rinha.index;

import java.util.Arrays;

public final class NearestNeighborsBuffer {

  private static final int K_NEIGHBORS = 5;

  private final long[] distances = new long[K_NEIGHBORS];
  private final byte[] fraudLabels = new byte[K_NEIGHBORS];
  private final int[] originalIds = new int[K_NEIGHBORS];
  private int worstIndex = 0;

  public NearestNeighborsBuffer() {
    Arrays.fill(distances, Long.MAX_VALUE);
  }

  /**
   * Reseta o buffer para reutilização via ThreadLocal. Deve ser chamado no início de cada request antes do primeiro
   * {@link #tryInsert}.
   */
  public void reset() {
    Arrays.fill(distances, Long.MAX_VALUE);
    Arrays.fill(fraudLabels, (byte) 0); // necessário: evita labels obsoletos de requests anteriores
    worstIndex = 0;
  }

  public void tryInsert(
    final long candidateDistance,
    final byte fraudLabel,
    final int originalId
  ) {

    final long worstDistance = distances[worstIndex];
    final int worstOriginalId = originalIds[worstIndex];

    if (candidateDistance > worstDistance) {
      return;
    }
    if (candidateDistance == worstDistance && originalId >= worstOriginalId) {
      return;
    }

    distances[worstIndex] = candidateDistance;
    fraudLabels[worstIndex] = fraudLabel;
    originalIds[worstIndex] = originalId;
    updateWorstIndex();
  }

  private void updateWorstIndex() {
    worstIndex = 0;
    for (int i = 1; i < K_NEIGHBORS; i++) {
      if ((distances[i] > distances[worstIndex]) ||
          (distances[i] == distances[worstIndex] &&
           originalIds[i] > originalIds[worstIndex])) {
        worstIndex = i;
      }
    }
  }

  public int countFraudVotes() {
    int votes = 0;
    for (final byte label : fraudLabels) {
      if (label == 1) {
        votes++;
      }
    }
    return votes;
  }

  public long worstDistance() {
    return distances[worstIndex];
  }
}