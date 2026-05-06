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

  public void tryInsert(final long candidateDistance, final byte fraudLabel, final int originalId) {
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
    for (int position = 1; position < K_NEIGHBORS; position++) {
      if (distances[position] > distances[worstIndex] ||
          (distances[position] == distances[worstIndex] && originalIds[position] > originalIds[worstIndex])) {
        worstIndex = position;
      }
    }
  }

  public int countFraudVotes() {
    int voteCount = 0;
    for (final byte fraudLabel : fraudLabels) {
      if (fraudLabel == 1) {
        voteCount++;
      }
    }
    return voteCount;
  }

  public long worstDistance() {
    return distances[worstIndex];
  }
}
