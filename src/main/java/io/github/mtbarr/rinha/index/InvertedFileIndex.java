package io.github.mtbarr.rinha.index;

import jakarta.annotation.PostConstruct;
import jakarta.enterprise.context.ApplicationScoped;
import java.io.File;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
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

  private MappedByteBuffer entireFileBuffer;
  private float[] ivfCentroidsFlat;
  private float[] sq8MinValues;
  private float[] sq8MaxValues;
  private float[] sq8DimensionRanges;
  private float[] sq8InverseLevels;
  private byte[] reorderedFraudLabels;
  private long vectorsSectionOffset;
  private long labelsSectionOffset;
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
      entireFileBuffer = fileChannel.map(
        FileChannel.MapMode.READ_ONLY,
        0,
        indexFile.length()
      );

      final int magicNumber = entireFileBuffer.getInt(0);
      if (magicNumber != 0x52494E44) {
        throw new IOException("Bad magic: " + Integer.toHexString(magicNumber));
      }
      final int versionNumber = entireFileBuffer.getInt(4);
      if (versionNumber != 1) {
        throw new IOException("Bad version: " + versionNumber);
      }
      final int storedClusters = entireFileBuffer.getInt(8);
      if (storedClusters != NUM_CLUSTERS) {
        throw new IOException(
          "Expected K=" + NUM_CLUSTERS + " got " + storedClusters
        );
      }
      final int totalVectorCount = entireFileBuffer.getInt(12);
      final long centroidsSectionOffset = entireFileBuffer.getLong(16);
      final long quantizationParamsOffset = entireFileBuffer.getLong(24);
      clusterTableOffset = entireFileBuffer.getLong(32);
      vectorsSectionOffset = entireFileBuffer.getLong(40);
      labelsSectionOffset = entireFileBuffer.getLong(48);

      ivfCentroidsFlat = new float[NUM_CLUSTERS * NUM_DIMENSIONS];
      final int centroidsStart = (int) centroidsSectionOffset;
      for (int i = 0; i < NUM_CLUSTERS * NUM_DIMENSIONS; i++) {
        ivfCentroidsFlat[i] = entireFileBuffer.getFloat(centroidsStart + i * 4);
      }

      sq8MinValues = new float[NUM_DIMENSIONS];
      sq8MaxValues = new float[NUM_DIMENSIONS];
      sq8DimensionRanges = new float[NUM_DIMENSIONS];
      sq8InverseLevels = new float[NUM_DIMENSIONS];
      final int paramsStart = (int) quantizationParamsOffset;
      for (int d = 0; d < NUM_DIMENSIONS; d++) {
        sq8MinValues[d] = entireFileBuffer.getFloat(paramsStart + (d * 2) * 4);
        sq8MaxValues[d] = entireFileBuffer.getFloat(paramsStart + (d * 2 + 1) * 4);
        sq8DimensionRanges[d] = sq8MaxValues[d] - sq8MinValues[d];
        sq8InverseLevels[d] = sq8DimensionRanges[d] / (SQ8_LEVELS - 1);
      }

      reorderedFraudLabels = new byte[totalVectorCount];
      final int labelsStart = (int) labelsSectionOffset;
      for (int i = 0; i < totalVectorCount; i++) {
        reorderedFraudLabels[i] = entireFileBuffer.get(labelsStart + i);
      }
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
      scaledQuery[d] = (queryVector[d] - sq8MinValues[d]) / sq8InverseLevels[d];
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
      final int clusterBase = entireFileBuffer.getInt(
        (int) (clusterTableOffset + (long) clusterIndex * CLUSTER_TABLE_ENTRY_SIZE)
      );
      final int clusterSize = entireFileBuffer.getInt(
        (int) (clusterTableOffset + (long) clusterIndex * CLUSTER_TABLE_ENTRY_SIZE + 4L)
      );
      final int clusterVectorStart = (int) (vectorsSectionOffset
                                      + (long) clusterBase * NUM_DIMENSIONS);

      for (int vectorIndex = 0; vectorIndex < clusterSize; vectorIndex++) {
        final int vectorOffset = clusterVectorStart + vectorIndex * NUM_DIMENSIONS;
        final int globalIndex = clusterBase + vectorIndex;
        final float approxDistance = computeSq8Distance(scaledQuery, vectorOffset);
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
    final int vectorOffset
  ) {
    float sum = 0;
    for (int d = 0; d < NUM_DIMENSIONS; d++) {
      final float diff = scaledQuery[d] - (entireFileBuffer.get(vectorOffset + d) & 0xFF);
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
