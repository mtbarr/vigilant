package io.github.mtbarr.rinha.index;

import jakarta.annotation.PostConstruct;
import jakarta.enterprise.context.ApplicationScoped;
import java.io.File;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import jdk.incubator.vector.IntVector;
import jdk.incubator.vector.ShortVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

@ApplicationScoped
public class InvertedFileIndex {

  public static final int CLUSTER_COUNT = 256;
  public static final int DIMENSIONS = 14;
  public static final int K_NEIGHBORS = 5;
  public static final float QUANTIZATION_SCALE = 10_000f;

  private static final int FAST_PROBE_COUNT = 8;
  private static final int FULL_PROBE_COUNT = 24;

  private static final VectorSpecies<Short> SHORT_SPECIES = ShortVector.SPECIES_PREFERRED;
  private static final int VECTOR_LANE_COUNT = SHORT_SPECIES.length();

  private short[][] quantizedCentroids;
  private short[][] bboxMinimum;
  private short[][] bboxMaximum;
  private int[] clusterStartOffset;
  private int[] clusterEndOffset;
  private short[][] dimensionsByVector;
  private byte[] vectorLabels;
  private int[] vectorOriginalIds;

  private final int[] squaredComponentBuffer = new int[Math.max(VECTOR_LANE_COUNT, DIMENSIONS)];
  private final short[] gatherBuffer = new short[Math.max(VECTOR_LANE_COUNT, DIMENSIONS)];

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

      final int vectorCount = buffer.getInt();
      final int clusterCount = buffer.getInt();
      if (clusterCount != CLUSTER_COUNT) {
        throw new IOException("Expected " + CLUSTER_COUNT + " clusters, got " + clusterCount);
      }

      final int dimensions = buffer.getInt();
      if (dimensions != DIMENSIONS) {
        throw new IOException("Expected " + DIMENSIONS + " dimensions, got " + dimensions);
      }

      buffer.getInt(); // stride
      final float scale = buffer.getFloat();
      if (Math.abs(scale - QUANTIZATION_SCALE) > 0.1f) {
        throw new IOException("Expected scale " + QUANTIZATION_SCALE + ", got " + scale);
      }

      final float[][] rawCentroids = new float[CLUSTER_COUNT][DIMENSIONS];
      for (int clusterIndex = 0; clusterIndex < CLUSTER_COUNT; clusterIndex++) {
        for (int dimension = 0; dimension < DIMENSIONS; dimension++) {
          rawCentroids[clusterIndex][dimension] = buffer.getFloat();
        }
      }

      bboxMinimum = new short[CLUSTER_COUNT][DIMENSIONS];
      bboxMaximum = new short[CLUSTER_COUNT][DIMENSIONS];
      for (int clusterIndex = 0; clusterIndex < CLUSTER_COUNT; clusterIndex++) {
        for (int dimension = 0; dimension < DIMENSIONS; dimension++) {
          bboxMinimum[clusterIndex][dimension] = buffer.getShort();
          bboxMaximum[clusterIndex][dimension] = buffer.getShort();
        }
      }

      final int[] clusterOffsets = new int[CLUSTER_COUNT + 1];
      for (int index = 0; index <= CLUSTER_COUNT; index++) {
        clusterOffsets[index] = buffer.getInt();
      }
      clusterStartOffset = new int[CLUSTER_COUNT];
      clusterEndOffset = new int[CLUSTER_COUNT];
      for (int clusterIndex = 0; clusterIndex < CLUSTER_COUNT; clusterIndex++) {
        clusterStartOffset[clusterIndex] = clusterOffsets[clusterIndex];
        clusterEndOffset[clusterIndex] = clusterOffsets[clusterIndex + 1];
      }

      final short[] rawVectors = new short[vectorCount * DIMENSIONS];
      for (int position = 0; position < vectorCount * DIMENSIONS; position++) {
        rawVectors[position] = buffer.getShort();
      }

      dimensionsByVector = new short[DIMENSIONS][vectorCount];
      for (int vectorIndex = 0; vectorIndex < vectorCount; vectorIndex++) {
        final int basePosition = vectorIndex * DIMENSIONS;
        for (int dimension = 0; dimension < DIMENSIONS; dimension++) {
          dimensionsByVector[dimension][vectorIndex] = rawVectors[basePosition + dimension];
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

      quantizedCentroids = new short[CLUSTER_COUNT][DIMENSIONS];
      for (int clusterIndex = 0; clusterIndex < CLUSTER_COUNT; clusterIndex++) {
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

    final long[] centroidDistances = new long[CLUSTER_COUNT];
    for (int clusterIndex = 0; clusterIndex < CLUSTER_COUNT; clusterIndex++) {
      centroidDistances[clusterIndex] = vectorSquaredDistance(
        queryQuantized, quantizedCentroids[clusterIndex]);
    }

    final int[] fastProbes = selectClosestCentroids(centroidDistances, FAST_PROBE_COUNT);
    final NearestNeighborsBuffer nearestNeighbors = new NearestNeighborsBuffer();
    final boolean[] visitedClusters = new boolean[CLUSTER_COUNT];

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

    for (int clusterIndex = 0; clusterIndex < CLUSTER_COUNT; clusterIndex++) {
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

      for (int dimension = 0; dimension < DIMENSIONS; dimension++) {
        gatherBuffer[dimension] = dimensionsByVector[dimension][vectorPosition];
      }

      final long squaredDistance = vectorSquaredDistance(queryQuantized, gatherBuffer);
      nearestNeighbors.tryInsert(
        squaredDistance,
        vectorLabels[vectorPosition],
        vectorOriginalIds[vectorPosition]
      );
    }
  }

  private long vectorSquaredDistance(final short[] vectorA, final short[] vectorB) {
    long totalSum = 0L;

    for (int offset = 0; offset < DIMENSIONS; offset += VECTOR_LANE_COUNT) {
      final var mask = SHORT_SPECIES.indexInRange(offset, DIMENSIONS);
      final var shortA = ShortVector.fromArray(SHORT_SPECIES, vectorA, offset, mask);
      final var shortB = ShortVector.fromArray(SHORT_SPECIES, vectorB, offset, mask);

      final var intA0 = (IntVector) shortA.convert(VectorOperators.S2I, 0);
      final var intB0 = (IntVector) shortB.convert(VectorOperators.S2I, 0);
      final var intA1 = (IntVector) shortA.convert(VectorOperators.S2I, 1);
      final var intB1 = (IntVector) shortB.convert(VectorOperators.S2I, 1);

      final var difference0 = intA0.sub(intB0);
      final var difference1 = intA1.sub(intB1);
      final var squared0 = difference0.mul(difference0);
      final var squared1 = difference1.mul(difference1);

      squared0.intoArray(squaredComponentBuffer, 0);
      squared1.intoArray(squaredComponentBuffer, VECTOR_LANE_COUNT / 2);

      final int validComponents = Math.min(VECTOR_LANE_COUNT, DIMENSIONS - offset);
      for (int component = 0; component < validComponents; component++) {
        totalSum += squaredComponentBuffer[component];
      }
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
    if (count == 1) {
      int bestIndex = 0;
      for (int position = 1; position < CLUSTER_COUNT; position++) {
        if (distances[position] < distances[bestIndex]) {
          bestIndex = position;
        }
      }
      return new int[]{bestIndex};
    }

    final int[] selectedIndices = new int[count];
    final boolean[] isUsed = new boolean[CLUSTER_COUNT];

    for (int selectionRound = 0; selectionRound < count; selectionRound++) {
      int bestIndex = -1;
      for (int position = 0; position < CLUSTER_COUNT; position++) {
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
