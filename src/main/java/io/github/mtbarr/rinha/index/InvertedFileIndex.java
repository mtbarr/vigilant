package io.github.mtbarr.rinha.index;

import jakarta.annotation.PostConstruct;
import jakarta.enterprise.context.ApplicationScoped;
import java.io.File;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;

@ApplicationScoped
public class InvertedFileIndex {

  public static final int DIMENSIONS = 14;
  public static final int K_NEIGHBORS = 5;
  public static final float QUANTIZATION_SCALE = 10_000f;

  private static final int FAST_PROBE_COUNT = 8;
  private static final int FULL_PROBE_COUNT = 24;

  private int clusterCount;
  private int vectorCount;
  private short[][] dimensionsByVector;
  private byte[] vectorLabels;
  private int[] vectorOriginalIds;
  private int[] clusterStartOffset;
  private int[] clusterEndOffset;
  private short[][] quantizedCentroids;
  private short[][] bboxMinimum;
  private short[][] bboxMaximum;

  private volatile boolean isIndexReady = false;

  @PostConstruct
  void initialize() {
    final String indexPath = System.getenv().getOrDefault("INDEX_PATH", "/data/index.bin");
    try {
      loadIndex(indexPath);
      isIndexReady = true;
    } catch (final IOException exception) {
      System.err.println("Index not loaded yet: " + exception.getMessage());
      isIndexReady = false;
    }
  }

  private void loadIndex(final String indexPath) throws IOException {
    final File indexFile = new File(indexPath);
    try (final RandomAccessFile file = new RandomAccessFile(indexFile, "r");
         final FileChannel channel = file.getChannel()) {

      final ByteBuffer buffer = channel.map(
        FileChannel.MapMode.READ_ONLY, 0, indexFile.length());
      buffer.order(ByteOrder.LITTLE_ENDIAN);

      final int magic = buffer.getInt();
      if (magic != 0x49564636) {
        throw new IOException("Unknown index format: " + Integer.toHexString(magic));
      }

      vectorCount = buffer.getInt();
      clusterCount = buffer.getInt();

      final int dimensions = buffer.getInt();
      if (dimensions != DIMENSIONS) {
        throw new IOException("Expected " + DIMENSIONS + " dimensions, got " + dimensions);
      }

      buffer.getInt();
      final float scale = buffer.getFloat();
      if (Math.abs(scale - QUANTIZATION_SCALE) > 0.1f) {
        throw new IOException("Expected scale " + QUANTIZATION_SCALE + ", got " + scale);
      }

      final float[][] rawCentroids = new float[clusterCount][DIMENSIONS];
      for (int clusterIndex = 0; clusterIndex < clusterCount; clusterIndex++) {
        for (int dimension = 0; dimension < DIMENSIONS; dimension++) {
          rawCentroids[clusterIndex][dimension] = buffer.getFloat();
        }
      }

      bboxMinimum = new short[clusterCount][DIMENSIONS];
      bboxMaximum = new short[clusterCount][DIMENSIONS];
      for (int clusterIndex = 0; clusterIndex < clusterCount; clusterIndex++) {
        for (int dimension = 0; dimension < DIMENSIONS; dimension++) {
          bboxMinimum[clusterIndex][dimension] = buffer.getShort();
          bboxMaximum[clusterIndex][dimension] = buffer.getShort();
        }
      }

      final int[] clusterOffsets = new int[clusterCount + 1];
      for (int index = 0; index <= clusterCount; index++) {
        clusterOffsets[index] = buffer.getInt();
      }
      clusterStartOffset = new int[clusterCount];
      clusterEndOffset = new int[clusterCount];
      for (int clusterIndex = 0; clusterIndex < clusterCount; clusterIndex++) {
        clusterStartOffset[clusterIndex] = clusterOffsets[clusterIndex];
        clusterEndOffset[clusterIndex] = clusterOffsets[clusterIndex + 1];
      }

      dimensionsByVector = new short[DIMENSIONS][vectorCount];
      for (int vectorIndex = 0; vectorIndex < vectorCount; vectorIndex++) {
        for (int dimension = 0; dimension < DIMENSIONS; dimension++) {
          dimensionsByVector[dimension][vectorIndex] = buffer.getShort();
        }
      }

      vectorLabels = new byte[vectorCount];
      for (int position = 0; position < vectorCount; position++) {
        vectorLabels[position] = buffer.get();
      }

      vectorOriginalIds = new int[vectorCount];
      for (int position = 0; position < vectorCount; position++) {
        vectorOriginalIds[position] = buffer.getInt();
      }

      quantizedCentroids = new short[clusterCount][DIMENSIONS];
      for (int clusterIndex = 0; clusterIndex < clusterCount; clusterIndex++) {
        for (int dimension = 0; dimension < DIMENSIONS; dimension++) {
          quantizedCentroids[clusterIndex][dimension] = quantize(rawCentroids[clusterIndex][dimension]);
        }
      }
    }
  }

  public int search(final float[] queryVectorFloat) {
    final short[] queryQuantized = new short[DIMENSIONS];
    for (int dimension = 0; dimension < DIMENSIONS; dimension++) {
      queryQuantized[dimension] = quantize(queryVectorFloat[dimension]);
    }

    final int k = clusterCount;
    final long[] centroidDistances = new long[k];
    for (int clusterIndex = 0; clusterIndex < k; clusterIndex++) {
      centroidDistances[clusterIndex] = vectorSquaredDistance(
        queryQuantized, quantizedCentroids[clusterIndex]);
    }

    final int[] fastProbes = selectClosestCentroids(centroidDistances, FAST_PROBE_COUNT);
    final NearestNeighborsBuffer nearestNeighbors = new NearestNeighborsBuffer();
    final boolean[] visitedClusters = new boolean[k];

    for (final int clusterIndex : fastProbes) {
      visitedClusters[clusterIndex] = true;
      scanCluster(queryQuantized, clusterIndex, nearestNeighbors);
    }

    final int fastFraudVotes = nearestNeighbors.countFraudVotes();

    if (fastFraudVotes == 2 || fastFraudVotes == 3) {
      final int[] fullProbes = selectClosestCentroids(centroidDistances, FULL_PROBE_COUNT);
      for (final int clusterIndex : fullProbes) {
        if (!visitedClusters[clusterIndex]) {
          visitedClusters[clusterIndex] = true;
          scanCluster(queryQuantized, clusterIndex, nearestNeighbors);
        }
      }
    }

    for (int clusterIndex = 0; clusterIndex < k; clusterIndex++) {
      if (visitedClusters[clusterIndex]) {
        continue;
      }
      if (bboxLowerBound(queryQuantized, clusterIndex) <= nearestNeighbors.worstDistance()) {
        scanCluster(queryQuantized, clusterIndex, nearestNeighbors);
      }
    }

    return nearestNeighbors.countFraudVotes();
  }

  private void scanCluster(final short[] queryQuantized, final int clusterIndex,
                           final NearestNeighborsBuffer nearestNeighbors) {
    final int endOffset = clusterEndOffset[clusterIndex];
    for (int vectorPosition = clusterStartOffset[clusterIndex]; vectorPosition < endOffset; vectorPosition++) {
      final long squaredDistance = vectorSquaredDistance(queryQuantized, vectorPosition);
      nearestNeighbors.tryInsert(
        squaredDistance,
        vectorLabels[vectorPosition],
        vectorOriginalIds[vectorPosition]
      );
    }
  }

  private long vectorSquaredDistance(final short[] queryVector, final int vectorPosition) {
    long totalSum = 0L;

    int dimension = 0;
    for (; dimension + 3 < DIMENSIONS; dimension += 4) {
      final int diff0 = queryVector[dimension] - dimensionsByVector[dimension][vectorPosition];
      final int diff1 = queryVector[dimension + 1] - dimensionsByVector[dimension + 1][vectorPosition];
      final int diff2 = queryVector[dimension + 2] - dimensionsByVector[dimension + 2][vectorPosition];
      final int diff3 = queryVector[dimension + 3] - dimensionsByVector[dimension + 3][vectorPosition];
      totalSum += (long) diff0 * diff0;
      totalSum += (long) diff1 * diff1;
      totalSum += (long) diff2 * diff2;
      totalSum += (long) diff3 * diff3;
    }
    for (; dimension < DIMENSIONS; dimension++) {
      final int diff = queryVector[dimension] - dimensionsByVector[dimension][vectorPosition];
      totalSum += (long) diff * diff;
    }

    return totalSum;
  }

  private long vectorSquaredDistance(final short[] vectorA, final short[] vectorB) {
    long totalSum = 0L;

    int dimension = 0;
    for (; dimension + 3 < DIMENSIONS; dimension += 4) {
      final int diff0 = vectorA[dimension] - vectorB[dimension];
      final int diff1 = vectorA[dimension + 1] - vectorB[dimension + 1];
      final int diff2 = vectorA[dimension + 2] - vectorB[dimension + 2];
      final int diff3 = vectorA[dimension + 3] - vectorB[dimension + 3];
      totalSum += (long) diff0 * diff0;
      totalSum += (long) diff1 * diff1;
      totalSum += (long) diff2 * diff2;
      totalSum += (long) diff3 * diff3;
    }
    for (; dimension < DIMENSIONS; dimension++) {
      final int diff = vectorA[dimension] - vectorB[dimension];
      totalSum += (long) diff * diff;
    }

    return totalSum;
  }

  private long bboxLowerBound(final short[] queryVector, final int clusterIndex) {
    long squaredSum = 0;
    for (int dimension = 0; dimension < DIMENSIONS; dimension++) {
      final int queryComponent = queryVector[dimension];
      final int minimumBound = bboxMinimum[clusterIndex][dimension];
      final int maximumBound = bboxMaximum[clusterIndex][dimension];
      final int difference;
      if (queryComponent < minimumBound) {
        difference = queryComponent - minimumBound;
      } else if (queryComponent > maximumBound) {
        difference = queryComponent - maximumBound;
      } else {
        difference = 0;
      }
      squaredSum += (long) difference * difference;
    }
    return squaredSum;
  }

  private int[] selectClosestCentroids(final long[] distances, final int count) {
    final int k = clusterCount;
    if (count == 1) {
      int bestIndex = 0;
      for (int position = 1; position < k; position++) {
        if (distances[position] < distances[bestIndex]) {
          bestIndex = position;
        }
      }
      return new int[]{bestIndex};
    }

    final int[] selectedIndices = new int[count];
    final boolean[] isUsed = new boolean[k];

    for (int selectionRound = 0; selectionRound < count; selectionRound++) {
      int bestIndex = -1;
      for (int position = 0; position < k; position++) {
        if (isUsed[position]) {
          continue;
        }
        if (bestIndex == -1 || distances[position] < distances[bestIndex]) {
          bestIndex = position;
        }
      }
      isUsed[bestIndex] = true;
      selectedIndices[selectionRound] = bestIndex;
    }

    return selectedIndices;
  }

  private static short quantize(final float value) {
    if (value > Short.MAX_VALUE) {
      return Short.MAX_VALUE;
    } else if (value < Short.MIN_VALUE) {
      return Short.MIN_VALUE;
    } else {
      return (short) Math.round(value * QUANTIZATION_SCALE);
    }
  }

  public boolean isReady() {
    return isIndexReady;
  }
}
