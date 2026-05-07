package io.github.mtbarr.rinha.index;

import jakarta.annotation.PostConstruct;
import jakarta.enterprise.context.ApplicationScoped;
import java.io.File;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.channels.FileChannel.MapMode;
import java.util.Arrays;

@ApplicationScoped
public class InvertedFileIndex {

  private static final int NUM_CLUSTERS = 512;
  private static final int NUM_DIMENSIONS = 14;
  private static final int SQ8_LEVELS = 256;
  private static final int NUM_PROBE_CLUSTERS = 64;
  private static final int NUM_CANDIDATES = 50;
  private static final int NUM_NEIGHBORS = 5;

  private static final long HEADER_SIZE = 56L;
  private static final long CLUSTER_TABLE_ENTRY_SIZE = 8L;

  private static final ValueLayout.OfInt INT_LE = ValueLayout.JAVA_INT.withOrder(ByteOrder.LITTLE_ENDIAN);
  private static final ValueLayout.OfLong LONG_LE = ValueLayout.JAVA_LONG.withOrder(ByteOrder.LITTLE_ENDIAN);
  private static final ValueLayout.OfFloat FLOAT_LE = ValueLayout.JAVA_FLOAT.withOrder(ByteOrder.LITTLE_ENDIAN);
  private static final ValueLayout.OfByte BYTE_LE = ValueLayout.JAVA_BYTE;

  private MemorySegment indexSegment;
  private float[] ivfCentroidsFlat;
  private float[] sq8ScaledMinValues;
  private float[] sq8InverseStep;
  private byte[] reorderedFraudLabels;
  private long vectorsSectionOffset;
  private long clusterTableOffset;
  private volatile boolean isIndexReady = false;

  private final ThreadLocal<float[]> scaledQueryBuffer = ThreadLocal.withInitial(
    () -> new float[NUM_DIMENSIONS]
  );
  private final ThreadLocal<float[]> centroidDistanceBuffer = ThreadLocal.withInitial(
    () -> new float[NUM_CLUSTERS]
  );
  private final ThreadLocal<int[]> centroidOrderBuffer = ThreadLocal.withInitial(
    () -> new int[NUM_CLUSTERS]
  );
  private final ThreadLocal<int[]> candidateIdBuffer = ThreadLocal.withInitial(
    () -> new int[NUM_CANDIDATES]
  );
  private final ThreadLocal<float[]> candidateDistanceBuffer = ThreadLocal.withInitial(
    () -> new float[NUM_CANDIDATES]
  );

  @PostConstruct
  void initialize() {
    final String indexPath = System.getenv().getOrDefault(
      "INDEX_PATH",
      "/data/index.bin"
    );
    try {
      loadIndexFromFile(indexPath);
      runWarmupQueries();
      isIndexReady = true;
    } catch (final IOException exception) {
      System.err.println("Index not loaded: " + exception.getMessage());
      isIndexReady = false;
    }
  }

  private void loadIndexFromFile(final String filePath) throws IOException {
    final File indexFile = new File(filePath);
    try (final var randomAccessFile = new RandomAccessFile(indexFile, "r");
         final var fileChannel = randomAccessFile.getChannel()) {
      indexSegment = fileChannel.map(
        MapMode.READ_ONLY,
        0,
        indexFile.length(),
        Arena.global()
      );

      final int magicNumber = indexSegment.get(INT_LE, 0);
      if (magicNumber != 0x52494E44) {
        throw new IOException("Bad magic: " + Integer.toHexString(magicNumber));
      }
      final int versionNumber = indexSegment.get(INT_LE, 4);
      if (versionNumber != 1) {
        throw new IOException("Bad version: " + versionNumber);
      }
      final int storedClusters = indexSegment.get(INT_LE, 8);
      if (storedClusters != NUM_CLUSTERS) {
        throw new IOException(
          "Expected K=" + NUM_CLUSTERS + " got " + storedClusters
        );
      }
      final int totalVectorCount = indexSegment.get(INT_LE, 12);
      final long centroidsSectionOffset = indexSegment.get(LONG_LE, 16);
      final long quantizationParamsOffset = indexSegment.get(LONG_LE, 24);
      clusterTableOffset = indexSegment.get(LONG_LE, 32);
      vectorsSectionOffset = indexSegment.get(LONG_LE, 40);
      final long labelsSectionOffset = indexSegment.get(LONG_LE, 48);
      final MemorySegment centroidsSegment = indexSegment.asSlice(
        centroidsSectionOffset,
        (long) NUM_CLUSTERS * NUM_DIMENSIONS * 4L
      );
      for (int i = 0; i < NUM_CLUSTERS * NUM_DIMENSIONS; i++) {
        ivfCentroidsFlat[i] = centroidsSegment.get(FLOAT_LE, (long) i * 4L);
      }

      sq8ScaledMinValues = new float[NUM_DIMENSIONS];
      sq8InverseStep = new float[NUM_DIMENSIONS];
      final MemorySegment paramsSegment = indexSegment.asSlice(
        quantizationParamsOffset,
        (long) NUM_DIMENSIONS * 2L * 4L
      );
      for (int d = 0; d < NUM_DIMENSIONS; d++) {
        final float minValue = paramsSegment.get(FLOAT_LE, (long) (d * 2) * 4L);
        final float maxValue = paramsSegment.get(FLOAT_LE, (long) (d * 2 + 1) * 4L);
        sq8ScaledMinValues[d] = minValue;
        sq8InverseStep[d] = (maxValue - minValue) / (SQ8_LEVELS - 1);
      }

      reorderedFraudLabels = new byte[totalVectorCount];
      final MemorySegment labelsSegment = indexSegment.asSlice(
        labelsSectionOffset,
        totalVectorCount
      );
      labelsSegment.asByteBuffer().get(reorderedFraudLabels);
    }
  }

  private void runWarmupQueries() {
    final float[] warmupQuery = new float[NUM_DIMENSIONS];
    final int[] warmupNeighbors = new int[NUM_NEIGHBORS];
    final float[] warmupDistances = new float[NUM_NEIGHBORS];
    for (int i = 0; i < 200; i++) {
      for (int d = 0; d < NUM_DIMENSIONS; d++) {
        warmupQuery[d] = (i * 7 + d) % 100 / 100f;
      }
      searchNearestNeighbors(warmupQuery, warmupNeighbors, warmupDistances);
    }
  }

  public int searchNearestNeighbors(
    final float[] queryVector,
    final int[] neighborIds,
    final float[] neighborDistances
  ) {
    final float[] scaledQuery = scaledQueryBuffer.get();
    for (int d = 0; d < NUM_DIMENSIONS; d++) {
      scaledQuery[d] = (queryVector[d] - sq8ScaledMinValues[d]) / sq8InverseStep[d];
    }

    final float[] centroidDistances = centroidDistanceBuffer.get();
    final int[] centroidOrder = centroidOrderBuffer.get();
    for (int clusterIndex = 0; clusterIndex < NUM_CLUSTERS; clusterIndex++) {
      float clusterDistance = 0;
      final int centroidBase = clusterIndex * NUM_DIMENSIONS;
      for (int dimIndex = 0; dimIndex < NUM_DIMENSIONS; dimIndex++) {
        final float delta = queryVector[dimIndex] - ivfCentroidsFlat[centroidBase + dimIndex];
        clusterDistance += delta * delta;
      }
      centroidDistances[clusterIndex] = clusterDistance;
      centroidOrder[clusterIndex] = clusterIndex;
    }
    partialSortCentroids(centroidOrder, centroidDistances, NUM_PROBE_CLUSTERS);

    final int[] candidateIds = candidateIdBuffer.get();
    final float[] candidateDists = candidateDistanceBuffer.get();
    Arrays.fill(candidateDists, Float.MAX_VALUE);
    Arrays.fill(candidateIds, -1);

    for (int probeIndex = 0; probeIndex < NUM_PROBE_CLUSTERS; probeIndex++) {
      final int clusterIndex = centroidOrder[probeIndex];
      final long tableEntryOffset = clusterTableOffset
                                    + (long) clusterIndex * CLUSTER_TABLE_ENTRY_SIZE;
      final int clusterBase = indexSegment.get(INT_LE, tableEntryOffset);
      final int clusterSize = indexSegment.get(INT_LE, tableEntryOffset + 4L);
      final long clusterVectorStart = vectorsSectionOffset
                                      + (long) clusterBase * NUM_DIMENSIONS;
      final MemorySegment clusterSegment = indexSegment.asSlice(
        clusterVectorStart,
        (long) clusterSize * NUM_DIMENSIONS
      );

      for (int vectorIndex = 0; vectorIndex < clusterSize; vectorIndex++) {
        final long vectorOffset = (long) vectorIndex * NUM_DIMENSIONS;
        final int globalIndex = clusterBase + vectorIndex;
        final float approxDistance = computeSq8Distance(scaledQuery, clusterSegment, vectorOffset);
        if (approxDistance < candidateDists[NUM_CANDIDATES - 1]) {
          insertIntoSortedArray(
            candidateIds,
            candidateDists,
            NUM_CANDIDATES,
            globalIndex,
            approxDistance
          );
        }
      }
    }

    Arrays.fill(neighborIds, -1);
    Arrays.fill(neighborDistances, Float.MAX_VALUE);
    for (int candidateIndex = 0; candidateIndex < NUM_CANDIDATES; candidateIndex++) {
      if (candidateIds[candidateIndex] >= 0
          && candidateDists[candidateIndex] < neighborDistances[NUM_NEIGHBORS - 1]) {
        insertIntoSortedArray(
          neighborIds,
          neighborDistances,
          NUM_NEIGHBORS,
          candidateIds[candidateIndex],
          candidateDists[candidateIndex]
        );
      }
    }

    int fraudVoteCount = 0;
    for (int neighborIndex = 0; neighborIndex < NUM_NEIGHBORS; neighborIndex++) {
      final int globalIndex = neighborIds[neighborIndex];
      if (globalIndex >= 0 && reorderedFraudLabels[globalIndex] == 1) {
        fraudVoteCount++;
      }
    }
    return fraudVoteCount;
  }

  private float computeSq8Distance(
    final float[] scaledQuery,
    final MemorySegment clusterSegment,
    final long vectorOffset
  ) {
    float sum = 0;
    for (int d = 0; d < NUM_DIMENSIONS; d++) {
      final float diff = scaledQuery[d] - (clusterSegment.getAtIndex(
        BYTE_LE,
        vectorOffset + d
      ) & 0xFF);
      sum += diff * diff;
    }
    return sum;
  }

  private static void insertIntoSortedArray(
    final int[] neighborIds,
    final float[] neighborDistances,
    final int maxNeighbors,
    final int newId,
    final float newDistance
  ) {
    int insertPosition = maxNeighbors - 1;
    while (insertPosition > 0 && neighborDistances[insertPosition - 1] > newDistance) {
      neighborDistances[insertPosition] = neighborDistances[insertPosition - 1];
      neighborIds[insertPosition] = neighborIds[insertPosition - 1];
      insertPosition--;
    }
    neighborDistances[insertPosition] = newDistance;
    neighborIds[insertPosition] = newId;
  }

  private static void partialSortCentroids(
    final int[] centroidOrder,
    final float[] centroidDistances,
    final int topCount
  ) {
    quickselectCentroids(
      centroidOrder,
      centroidDistances,
      0,
      centroidOrder.length - 1,
      topCount
    );
    for (int i = 1; i < topCount; i++) {
      final int currentOrder = centroidOrder[i];
      final float currentDistance = centroidDistances[currentOrder];
      int j = i - 1;
      while (j >= 0 && centroidDistances[centroidOrder[j]] > currentDistance) {
        centroidOrder[j + 1] = centroidOrder[j];
        j--;
      }
      centroidOrder[j + 1] = currentOrder;
    }
  }

  private static void quickselectCentroids(
    final int[] centroidOrder,
    final float[] centroidDistances,
    final int leftBoundary,
    final int rightBoundary,
    final int targetRank
  ) {
    if (leftBoundary >= rightBoundary) {
      return;
    }
    final int pivotIndex = partitionCentroids(
      centroidOrder,
      centroidDistances,
      leftBoundary,
      rightBoundary
    );
    final int pivotRank = pivotIndex - leftBoundary + 1;
    if (pivotRank == targetRank) {
      return;
    }
    if (targetRank < pivotRank) {
      quickselectCentroids(
        centroidOrder,
        centroidDistances,
        leftBoundary,
        pivotIndex - 1,
        targetRank
      );
    } else {
      quickselectCentroids(
        centroidOrder,
        centroidDistances,
        pivotIndex + 1,
        rightBoundary,
        targetRank - pivotRank
      );
    }
  }

  private static int partitionCentroids(
    final int[] centroidOrder,
    final float[] centroidDistances,
    final int leftBoundary,
    final int rightBoundary
  ) {
    final float pivotDistance = centroidDistances[centroidOrder[rightBoundary]];
    int swapIndex = leftBoundary - 1;
    for (int scanIndex = leftBoundary; scanIndex < rightBoundary; scanIndex++) {
      if (centroidDistances[centroidOrder[scanIndex]] <= pivotDistance) {
        swapIndex++;
        final int tempOrder = centroidOrder[swapIndex];
        centroidOrder[swapIndex] = centroidOrder[scanIndex];
        centroidOrder[scanIndex] = tempOrder;
      }
    }
    final int tempOrder = centroidOrder[swapIndex + 1];
    centroidOrder[swapIndex + 1] = centroidOrder[rightBoundary];
    centroidOrder[rightBoundary] = tempOrder;
    return swapIndex + 1;
  }

  public boolean isReady() {
    return isIndexReady;
  }
}
