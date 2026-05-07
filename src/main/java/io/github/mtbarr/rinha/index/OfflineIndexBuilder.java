package io.github.mtbarr.rinha.index;

import java.io.FileInputStream;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.IntStream;
import java.util.zip.GZIPInputStream;

public final class OfflineIndexBuilder {

  private static final int NUM_CLUSTERS = 512;
  private static final int NUM_DIMENSIONS = 14;
  private static final int SQ8_LEVELS = 256;
  private static final int TRAINING_SAMPLE_SIZE = 50_000;
  private static final int IVF_MAX_ITERATIONS = 25;
  private static final long RANDOM_SEED = 0xdeadbeefcafebabeL;

  private static final long HEADER_SIZE = 56L;
  private static final long CLUSTER_TABLE_ENTRY_SIZE = 8L;

  private OfflineIndexBuilder() {
  }

  public static void main(final String[] args) throws Exception {
    if (args.length < 2) {
      System.err.println("Usage: OfflineIndexBuilder <references.json.gz> <output.index.bin>");
      System.exit(1);
    }

    final String inputPath = args[0];
    final String outputPath = args[1];

    System.out.println("Loading dataset...");
    final float[][] featureVectors = loadFeatureVectors(inputPath);
    final byte[] fraudLabels = loadFraudLabels(inputPath);
    final int numVectors = featureVectors.length;
    System.out.println("  " + numVectors + " vectors, fraud=" + countFraud(fraudLabels));

    System.out.println("Computing SQ8 quantization parameters...");
    final float[] dimensionMinValues = computeDimensionMinValues(featureVectors);
    final float[] dimensionMaxValues = computeDimensionMaxValues(featureVectors);
    applyEpsilonToRanges(dimensionMinValues, dimensionMaxValues);

    System.out.println("K-Means++ init for IVF (K=" + NUM_CLUSTERS + ")...");
    final float[][] ivfCentroids = initializeKMeansPlusPlus(
      featureVectors,
      NUM_CLUSTERS,
      TRAINING_SAMPLE_SIZE,
      RANDOM_SEED
    );

    System.out.println("IVF Lloyd iterations...");
    final int[] vectorClusterAssignments = new int[numVectors];
    runLloydAlgorithm(
      featureVectors,
      ivfCentroids,
      vectorClusterAssignments,
      IVF_MAX_ITERATIONS
    );

    System.out.println("Reordering vectors and labels by cluster...");
    final int[] clusterCounts = computeClusterCounts(vectorClusterAssignments, numVectors);
    final int[] clusterCumulativeCounts = computeClusterCumulativeCounts(clusterCounts);
    final byte[] sq8QuantizedVectors = reorderAndQuantizeVectors(
      featureVectors,
      vectorClusterAssignments,
      clusterCumulativeCounts,
      dimensionMinValues,
      dimensionMaxValues
    );
    final byte[] reorderedLabels = reorderLabels(
      fraudLabels,
      vectorClusterAssignments,
      clusterCumulativeCounts
    );

    System.out.println("Writing index...");
    writeIndexToFile(
      outputPath,
      numVectors,
      ivfCentroids,
      dimensionMinValues,
      dimensionMaxValues,
      clusterCounts,
      clusterCumulativeCounts,
      sq8QuantizedVectors,
      reorderedLabels
    );

    System.out.println("Done. Index written to " + outputPath);
  }

  private static float[][] loadFeatureVectors(final String filePath) throws IOException {
    final byte[] rawData;
    try (final var gzipInputStream = new GZIPInputStream(new FileInputStream(filePath))) {
      rawData = gzipInputStream.readAllBytes();
    }

    final int estimatedCapacity = 3_200_000;
    final float[] flatVectors = new float[estimatedCapacity * NUM_DIMENSIONS];
    int vectorCount = 0;

    int searchPosition = 0;
    final byte[] vectorTag = "\"vector\":[".getBytes(StandardCharsets.US_ASCII);

    while (searchPosition < rawData.length) {
      final int vectorTagIndex = findByteArrayIndex(rawData, vectorTag, searchPosition);
      if (vectorTagIndex < 0) {
        break;
      }

      int cursor = vectorTagIndex + vectorTag.length;
      int dimension = 0;
      final int baseOffset = vectorCount * NUM_DIMENSIONS;
      while (dimension < NUM_DIMENSIONS) {
        while (cursor < rawData.length
               && (rawData[cursor] == ' ' || rawData[cursor] == '\n' || rawData[cursor] == '\r')) {
          cursor++;
        }
        final int valueEnd = findNextDelimiter(rawData, cursor);
        flatVectors[baseOffset + dimension] = parseFloatFromBytes(rawData, cursor, valueEnd);
        cursor = valueEnd + 1;
        dimension++;
      }

      vectorCount++;
      searchPosition = cursor;
    }

    final float[][] result = new float[vectorCount][NUM_DIMENSIONS];
    for (int i = 0; i < vectorCount; i++) {
      System.arraycopy(flatVectors, i * NUM_DIMENSIONS, result[i], 0, NUM_DIMENSIONS);
    }
    return result;
  }

  private static byte[] loadFraudLabels(final String filePath) throws IOException {
    final byte[] rawData;
    try (final var gzipInputStream = new GZIPInputStream(new FileInputStream(filePath))) {
      rawData = gzipInputStream.readAllBytes();
    }

    final int estimatedCapacity = 3_200_000;
    final byte[] labels = new byte[estimatedCapacity];
    int labelCount = 0;

    int searchPosition = 0;
    final byte[] labelTag = "\"label\":\"".getBytes(StandardCharsets.US_ASCII);

    while (searchPosition < rawData.length) {
      final int labelTagIndex = findByteArrayIndex(rawData, labelTag, searchPosition);
      if (labelTagIndex < 0) {
        break;
      }
      final int labelValueIndex = labelTagIndex + labelTag.length;
      labels[labelCount] = rawData[labelValueIndex] == 'f' ? (byte) 1 : (byte) 0;
      labelCount++;
      searchPosition = labelTagIndex + labelTag.length + 6;
    }

    return Arrays.copyOf(labels, labelCount);
  }

  private static int findByteArrayIndex(
    final byte[] data,
    final byte[] pattern,
    final int fromIndex
  ) {
    final int patternLength = pattern.length;
    final int searchLimit = data.length - patternLength;
    for (int i = fromIndex; i <= searchLimit; i++) {
      boolean match = true;
      for (int j = 0; j < patternLength; j++) {
        if (data[i + j] != pattern[j]) {
          match = false;
          break;
        }
      }
      if (match) {
        return i;
      }
    }
    return -1;
  }

  private static int findNextDelimiter(final byte[] raw, final int fromIndex) {
    int position = fromIndex;
    while (position < raw.length
           && raw[position] != ','
           && raw[position] != ']') {
      position++;
    }
    return position;
  }

  private static float parseFloatFromBytes(
    final byte[] raw,
    final int startIndex,
    final int endIndex
  ) {
    float result = 0;
    boolean isNegative = false;
    int position = startIndex;
    if (position < endIndex && raw[position] == '-') {
      isNegative = true;
      position++;
    }
    while (position < endIndex && raw[position] >= '0' && raw[position] <= '9') {
      result = result * 10 + (raw[position] - '0');
      position++;
    }
    if (position < endIndex && raw[position] == '.') {
      position++;
      float fractionalPart = 0;
      float divisor = 1;
      while (position < endIndex && raw[position] >= '0' && raw[position] <= '9') {
        fractionalPart = fractionalPart * 10 + (raw[position] - '0');
        divisor *= 10;
        position++;
      }
      result += fractionalPart / divisor;
    }
    return isNegative ? -result : result;
  }

  private static int countFraud(final byte[] fraudLabels) {
    int fraudCount = 0;
    for (final byte label : fraudLabels) {
      if (label == 1) {
        fraudCount++;
      }
    }
    return fraudCount;
  }

  private static float[] computeDimensionMinValues(final float[][] featureVectors) {
    final int numVectors = featureVectors.length;
    final float[] minValues = new float[NUM_DIMENSIONS];
    Arrays.fill(minValues, Float.MAX_VALUE);
    for (int i = 0; i < numVectors; i++) {
      for (int d = 0; d < NUM_DIMENSIONS; d++) {
        if (featureVectors[i][d] < minValues[d]) {
          minValues[d] = featureVectors[i][d];
        }
      }
    }
    return minValues;
  }

  private static float[] computeDimensionMaxValues(final float[][] featureVectors) {
    final int numVectors = featureVectors.length;
    final float[] maxValues = new float[NUM_DIMENSIONS];
    Arrays.fill(maxValues, Float.MIN_VALUE);
    for (int i = 0; i < numVectors; i++) {
      for (int d = 0; d < NUM_DIMENSIONS; d++) {
        if (featureVectors[i][d] > maxValues[d]) {
          maxValues[d] = featureVectors[i][d];
        }
      }
    }
    return maxValues;
  }

  private static void applyEpsilonToRanges(
    final float[] minValues,
    final float[] maxValues
  ) {
    for (int d = 0; d < NUM_DIMENSIONS; d++) {
      if (maxValues[d] - minValues[d] < 1e-6f) {
        maxValues[d] = minValues[d] + 1e-6f;
      }
    }
  }

  private static float[][] initializeKMeansPlusPlus(
    final float[][] featureVectors,
    final int numClusters,
    final int sampleSize,
    final long seed
  ) {
    final int numVectors = featureVectors.length;
    final int effectiveSampleSize = Math.min(numVectors, sampleSize);
    final int[] sampleIndices = new int[effectiveSampleSize];
    final long[] randomState = {seed};
    for (int i = 0; i < effectiveSampleSize; i++) {
      randomState[0] = randomState[0] * 6364136223846793005L + 1442695040888963407L;
      sampleIndices[i] = (int) ((randomState[0] >>> 33) % numVectors);
    }

    final float[][] centroids = new float[numClusters][NUM_DIMENSIONS];
    final double[] minDistances = new double[effectiveSampleSize];
    Arrays.fill(minDistances, Double.POSITIVE_INFINITY);

    centroids[0] = featureVectors[sampleIndices[0]].clone();

    for (int clusterIndex = 1; clusterIndex < numClusters; clusterIndex++) {
      final float[] lastCentroid = centroids[clusterIndex - 1];
      double totalDistance = 0;
      for (int i = 0; i < effectiveSampleSize; i++) {
        final double distance = computeSquaredEuclideanDistance(
          featureVectors[sampleIndices[i]],
          lastCentroid
        );
        if (distance < minDistances[i]) {
          minDistances[i] = distance;
        }
        totalDistance += minDistances[i];
      }
      randomState[0] = randomState[0] * 6364136223846793005L + 1442695040888963407L;
      final double randomValue = ((randomState[0] >>> 11) / (double) (1L << 53)) * totalDistance;
      double cumulativeDistance = 0;
      int chosenIndex = effectiveSampleSize - 1;
      for (int i = 0; i < effectiveSampleSize; i++) {
        cumulativeDistance += minDistances[i];
        if (cumulativeDistance >= randomValue) {
          chosenIndex = i;
          break;
        }
      }
      centroids[clusterIndex] = featureVectors[sampleIndices[chosenIndex]].clone();
    }
    return centroids;
  }

  private static void runLloydAlgorithm(
    final float[][] featureVectors,
    final float[][] centroids,
    final int[] vectorClusterAssignments,
    final int maxIterations
  ) {
    final int numVectors = featureVectors.length;
    final int numClusters = centroids.length;
    for (int iteration = 0; iteration < maxIterations; iteration++) {
      final AtomicInteger changeCount = new AtomicInteger(0);
      IntStream.range(0, numVectors).parallel().forEach(i -> {
        int bestCluster = 0;
        float bestDistance = Float.MAX_VALUE;
        for (int c = 0; c < numClusters; c++) {
          final float distance = computeSquaredEuclideanDistance(
            featureVectors[i],
            centroids[c]
          );
          if (distance < bestDistance) {
            bestDistance = distance;
            bestCluster = c;
          }
        }
        if (vectorClusterAssignments[i] != bestCluster) {
          vectorClusterAssignments[i] = bestCluster;
          changeCount.incrementAndGet();
        }
      });
      if (iteration > 0 && changeCount.get() == 0) {
        System.out.println("  converged at " + iteration);
        break;
      }
      final double[][] centroidSums = new double[numClusters][NUM_DIMENSIONS];
      final int[] clusterCounts = new int[numClusters];
      for (int i = 0; i < numVectors; i++) {
        final int cluster = vectorClusterAssignments[i];
        clusterCounts[cluster]++;
        for (int d = 0; d < NUM_DIMENSIONS; d++) {
          centroidSums[cluster][d] += featureVectors[i][d];
        }
      }
      for (int c = 0; c < numClusters; c++) {
        if (clusterCounts[c] > 0) {
          for (int d = 0; d < NUM_DIMENSIONS; d++) {
            centroids[c][d] = (float) (centroidSums[c][d] / clusterCounts[c]);
          }
        }
      }
      System.out.println("  iter " + (iteration + 1) + ": " + changeCount.get() + " changed");
    }
  }

  private static int[] computeClusterCounts(
    final int[] vectorClusterAssignments,
    final int numVectors
  ) {
    final int[] clusterCounts = new int[NUM_CLUSTERS];
    for (int i = 0; i < numVectors; i++) {
      clusterCounts[vectorClusterAssignments[i]]++;
    }
    return clusterCounts;
  }

  private static int[] computeClusterCumulativeCounts(final int[] clusterCounts) {
    final int[] cumulativeCounts = new int[NUM_CLUSTERS];
    int runningTotal = 0;
    for (int c = 0; c < NUM_CLUSTERS; c++) {
      cumulativeCounts[c] = runningTotal;
      runningTotal += clusterCounts[c];
    }
    return cumulativeCounts;
  }

  private static byte[] reorderAndQuantizeVectors(
    final float[][] featureVectors,
    final int[] vectorClusterAssignments,
    final int[] clusterCumulativeCounts,
    final float[] dimensionMinValues,
    final float[] dimensionMaxValues
  ) {
    final int numVectors = featureVectors.length;
    final byte[] sq8QuantizedVectors = new byte[numVectors * NUM_DIMENSIONS];
    final int[] positionTracker = new int[NUM_CLUSTERS];
    final float[] dimensionRanges = new float[NUM_DIMENSIONS];
    for (int d = 0; d < NUM_DIMENSIONS; d++) {
      dimensionRanges[d] = dimensionMaxValues[d] - dimensionMinValues[d];
    }

    for (int i = 0; i < numVectors; i++) {
      final int cluster = vectorClusterAssignments[i];
      final int position = positionTracker[cluster]++;
      final int globalIndex = clusterCumulativeCounts[cluster] + position;
      final int byteOffset = globalIndex * NUM_DIMENSIONS;
      for (int d = 0; d < NUM_DIMENSIONS; d++) {
        final float normalizedValue = (featureVectors[i][d] - dimensionMinValues[d])
                                      / dimensionRanges[d];
        sq8QuantizedVectors[byteOffset + d] = (byte) Math.round(
          normalizedValue * (SQ8_LEVELS - 1)
        );
      }
    }
    return sq8QuantizedVectors;
  }

  private static byte[] reorderLabels(
    final byte[] fraudLabels,
    final int[] vectorClusterAssignments,
    final int[] clusterCumulativeCounts
  ) {
    final int numVectors = fraudLabels.length;
    final byte[] reorderedLabels = new byte[numVectors];
    final int[] positionTracker = new int[NUM_CLUSTERS];

    for (int i = 0; i < numVectors; i++) {
      final int cluster = vectorClusterAssignments[i];
      final int position = positionTracker[cluster]++;
      final int globalIndex = clusterCumulativeCounts[cluster] + position;
      reorderedLabels[globalIndex] = fraudLabels[i];
    }
    return reorderedLabels;
  }

  private static void writeIndexToFile(
    final String filePath,
    final int numVectors,
    final float[][] ivfCentroids,
    final float[] dimensionMinValues,
    final float[] dimensionMaxValues,
    final int[] clusterCounts,
    final int[] clusterCumulativeCounts,
    final byte[] sq8QuantizedVectors,
    final byte[] reorderedLabels
  ) throws IOException {
    final long centroidsSectionOffset = HEADER_SIZE;
    final long quantizationParamsOffset = centroidsSectionOffset
                                          + (long) NUM_CLUSTERS * NUM_DIMENSIONS * 4L;
    final long clusterTableOffset = quantizationParamsOffset
                                    + (long) NUM_DIMENSIONS * 2L * 4L;
    final long vectorsSectionOffset = clusterTableOffset
                                      + (long) NUM_CLUSTERS * CLUSTER_TABLE_ENTRY_SIZE;
    final long labelsSectionOffset = vectorsSectionOffset
                                     + (long) numVectors * NUM_DIMENSIONS;
    final long totalFileSize = labelsSectionOffset + numVectors;

    try (final var randomAccessFile = new RandomAccessFile(filePath, "rw");
         final var fileChannel = randomAccessFile.getChannel()) {
      final ByteBuffer buffer = fileChannel.map(
        FileChannel.MapMode.READ_WRITE,
        0,
        totalFileSize
      );
      buffer.order(ByteOrder.LITTLE_ENDIAN);

      buffer.putInt(0, 0x52494E44);
      buffer.putInt(4, 1);
      buffer.putInt(8, NUM_CLUSTERS);
      buffer.putInt(12, numVectors);
      buffer.putLong(16, centroidsSectionOffset);
      buffer.putLong(24, quantizationParamsOffset);
      buffer.putLong(32, clusterTableOffset);
      buffer.putLong(40, vectorsSectionOffset);
      buffer.putLong(48, labelsSectionOffset);

      final int centroidsStart = (int) centroidsSectionOffset;
      for (int c = 0; c < NUM_CLUSTERS; c++) {
        for (int d = 0; d < NUM_DIMENSIONS; d++) {
          buffer.putFloat(centroidsStart + (c * NUM_DIMENSIONS + d) * 4, ivfCentroids[c][d]);
        }
      }

      final int paramsStart = (int) quantizationParamsOffset;
      for (int d = 0; d < NUM_DIMENSIONS; d++) {
        buffer.putFloat(paramsStart + (d * 2) * 4, dimensionMinValues[d]);
        buffer.putFloat(paramsStart + (d * 2 + 1) * 4, dimensionMaxValues[d]);
      }

      final int tableStart = (int) clusterTableOffset;
      for (int c = 0; c < NUM_CLUSTERS; c++) {
        buffer.putInt(tableStart + c * 8, clusterCumulativeCounts[c]);
        buffer.putInt(tableStart + c * 8 + 4, clusterCounts[c]);
      }

      final int vectorsStart = (int) vectorsSectionOffset;
      for (int i = 0; i < sq8QuantizedVectors.length; i++) {
        buffer.put(vectorsStart + i, sq8QuantizedVectors[i]);
      }

      final int labelsStart = (int) labelsSectionOffset;
      for (int i = 0; i < reorderedLabels.length; i++) {
        buffer.put(labelsStart + i, reorderedLabels[i]);
      }
    }
  }

  private static float computeSquaredEuclideanDistance(
    final float[] vectorA,
    final float[] vectorB
  ) {
    float sum = 0;
    for (int i = 0; i < NUM_DIMENSIONS; i++) {
      final float delta = vectorA[i] - vectorB[i];
      sum += delta * delta;
    }
    return sum;
  }
}
