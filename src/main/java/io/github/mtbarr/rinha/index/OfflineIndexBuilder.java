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
  private static final int IVF_MAX_ITERATIONS = 20;
  private static final long RANDOM_SEED = 42L;
  private static final float I16_SCALE = 32767.0f;

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
    final byte[] rawData;
    try (final var gzipInputStream = new GZIPInputStream(new FileInputStream(inputPath))) {
      rawData = gzipInputStream.readAllBytes();
    }

    final float[][] vectors = parseFeatureVectors(rawData);
    final byte[] labels = parseFraudLabels(rawData);
    final int numVectors = vectors.length;
    System.out.println("  " + numVectors + " vectors, fraud=" + countFraud(labels));

    System.out.println("K-Means++ init for IVF (K=" + NUM_CLUSTERS + ")...");
    final float[][] ivfCentroids = initializeKMeansPlusPlus(vectors, NUM_CLUSTERS, RANDOM_SEED);

    System.out.println("IVF Lloyd iterations...");
    final int[] clusterAssignments = new int[numVectors];
    runLloydAlgorithm(vectors, ivfCentroids, clusterAssignments, IVF_MAX_ITERATIONS);

    System.out.println("Converting to i16...");
    final short[] vectorsI16 = convertToI16(vectors);

    System.out.println("Building inverted lists...");
    final int[][] idsByCluster = buildInvertedLists(numVectors, clusterAssignments);

    System.out.println("Writing index...");
    writeIndexToFile(outputPath, numVectors, ivfCentroids, vectorsI16, idsByCluster, labels);

    System.out.println("Done.");
  }

  private static short[] convertToI16(final float[][] vectors) {
    final int n = vectors.length;
    final short[] result = new short[n * NUM_DIMENSIONS];
    for (int i = 0; i < n; i++) {
      for (int d = 0; d < NUM_DIMENSIONS; d++) {
        result[i * NUM_DIMENSIONS + d] = (short) Math.round(vectors[i][d] * I16_SCALE);
      }
    }
    return result;
  }

  private static int[][] buildInvertedLists(final int numVectors, final int[] assignments) {
    final int[] clusterCounts = new int[NUM_CLUSTERS];
    for (int i = 0; i < numVectors; i++) {
      clusterCounts[assignments[i]]++;
    }
    final int[][] idsByCluster = new int[NUM_CLUSTERS][];
    for (int c = 0; c < NUM_CLUSTERS; c++) {
      idsByCluster[c] = new int[clusterCounts[c]];
    }
    final int[] positionTracker = new int[NUM_CLUSTERS];
    for (int i = 0; i < numVectors; i++) {
      final int cluster = assignments[i];
      idsByCluster[cluster][positionTracker[cluster]++] = i;
    }
    return idsByCluster;
  }

  private static void writeIndexToFile(
    final String filePath,
    final int numVectors,
    final float[][] ivfCentroids,
    final short[] vectorsI16,
    final int[][] idsByCluster,
    final byte[] labels
  ) throws IOException {
    final long centroidsOffset = 40L;
    final long labelsOffset = centroidsOffset + (long) NUM_CLUSTERS * NUM_DIMENSIONS * 4L;
    long vectorsI16Offset = labelsOffset + numVectors;
    if (vectorsI16Offset % 4L != 0) {
      vectorsI16Offset = (vectorsI16Offset + 3L) & ~3L;
    }
    final long invertedListsOffset = vectorsI16Offset + (long) numVectors * NUM_DIMENSIONS * 2L;

    final long[] clusterListOffsets = new long[NUM_CLUSTERS];
    long offset = invertedListsOffset;
    for (int c = 0; c < NUM_CLUSTERS; c++) {
      clusterListOffsets[c] = offset;
      final long dataSize = 4L + (long) idsByCluster[c].length * 4L;
      offset += (dataSize + 3L) & ~3L;
    }
    final long totalFileSize = offset;

    try (final var randomAccessFile = new RandomAccessFile(filePath, "rw");
      final var fileChannel = randomAccessFile.getChannel()) {
      final ByteBuffer buffer = fileChannel.map(FileChannel.MapMode.READ_WRITE, 0, totalFileSize);
      buffer.order(ByteOrder.LITTLE_ENDIAN);

      // Header v2: magic, version, K, N, reserved, labelsOffset, vectorsI16Offset
      buffer.putInt(0x52494E44);
      buffer.putInt(2); // version 2 = i16
      buffer.putInt(NUM_CLUSTERS);
      buffer.putInt(numVectors);
      buffer.putLong(0L); // reserved
      buffer.putLong(labelsOffset);
      buffer.putLong(vectorsI16Offset);

      buffer.position((int) centroidsOffset);
      for (int c = 0; c < NUM_CLUSTERS; c++) {
        for (int d = 0; d < NUM_DIMENSIONS; d++) {
          buffer.putFloat(ivfCentroids[c][d]);
        }
      }

      buffer.position((int) labelsOffset);
      buffer.put(labels);

      buffer.position((int) vectorsI16Offset);
      for (int i = 0; i < vectorsI16.length; i++) {
        buffer.putShort(vectorsI16[i]);
      }

      for (int c = 0; c < NUM_CLUSTERS; c++) {
        buffer.position((int) clusterListOffsets[c]);
        buffer.putInt(idsByCluster[c].length);
        for (final int id : idsByCluster[c]) {
          buffer.putInt(id);
        }
        final int dataLen = 4 + idsByCluster[c].length * 4;
        final int padding = ((dataLen + 3) & ~3) - dataLen;
        for (int p = 0; p < padding; p++) {
          buffer.put((byte) 0);
        }
      }
    }

    System.out.println("  Index file size: " + totalFileSize + " bytes");
    System.out.println("  Vectors i16: " + vectorsI16Offset + " - " + invertedListsOffset
                       + " (" + ((invertedListsOffset - vectorsI16Offset) / 1024 / 1024) + " MB)");
    System.out.println("  Inverted lists: " + invertedListsOffset + " - " + totalFileSize
                       + " (" + ((totalFileSize - invertedListsOffset) / 1024 / 1024) + " MB)");
  }

  private static float[][] parseFeatureVectors(final byte[] rawData) {
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

  private static byte[] parseFraudLabels(final byte[] rawData) {
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

  private static int countFraud(final byte[] fraudLabels) {
    int fraudCount = 0;
    for (final byte label : fraudLabels) {
      if (label == 1) {
        fraudCount++;
      }
    }
    return fraudCount;
  }

  private static int findByteArrayIndex(final byte[] data, final byte[] pattern, final int fromIndex) {
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
    while (position < raw.length && raw[position] != ',' && raw[position] != ']') {
      position++;
    }
    return position;
  }

  private static float parseFloatFromBytes(final byte[] raw, final int startIndex, final int endIndex) {
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

  private static float[][] initializeKMeansPlusPlus(
    final float[][] vectors,
    final int numClusters,
    final long seed
  ) {
    final int numVectors = vectors.length;
    final java.util.Random rng = new java.util.Random(seed);
    final float[][] centroids = new float[numClusters][NUM_DIMENSIONS];
    final float[] distances = new float[numVectors];

    int first = rng.nextInt(numVectors);
    System.arraycopy(vectors[first], 0, centroids[0], 0, NUM_DIMENSIONS);
    for (int i = 0; i < numVectors; i++) {
      distances[i] = computeSquaredEuclideanDistance(vectors[i], centroids[0]);
    }

    for (int k = 1; k < numClusters; k++) {
      double total = 0;
      for (float d : distances) total += d;
      double threshold = rng.nextDouble() * total;
      double cumulative = 0;
      int chosen = numVectors - 1;
      for (int i = 0; i < numVectors; i++) {
        cumulative += distances[i];
        if (cumulative >= threshold) {
          chosen = i;
          break;
        }
      }
      System.arraycopy(vectors[chosen], 0, centroids[k], 0, NUM_DIMENSIONS);
      for (int i = 0; i < numVectors; i++) {
        float d = computeSquaredEuclideanDistance(vectors[i], centroids[k]);
        if (d < distances[i]) distances[i] = d;
      }
    }
    return centroids;
  }

  private static void runLloydAlgorithm(
    final float[][] vectors,
    final float[][] centroids,
    final int[] assignments,
    final int maxIterations
  ) {
    final int numVectors = vectors.length;
    final int numClusters = centroids.length;
    for (int iteration = 0; iteration < maxIterations; iteration++) {
      final AtomicInteger changeCount = new AtomicInteger(0);
      IntStream.range(0, numVectors).parallel().forEach(i -> {
        int bestCluster = 0;
        float bestDistance = Float.MAX_VALUE;
        for (int c = 0; c < numClusters; c++) {
          final float distance = computeSquaredEuclideanDistance(vectors[i], centroids[c]);
          if (distance < bestDistance) {
            bestDistance = distance;
            bestCluster = c;
          }
        }
        if (assignments[i] != bestCluster) {
          assignments[i] = bestCluster;
          changeCount.incrementAndGet();
        }
      });
      if (iteration > 0 && changeCount.get() == 0) {
        System.out.println("  converged at " + iteration);
        break;
      }
      final float[][] acc = new float[numClusters][NUM_DIMENSIONS];
      final int[] counts = new int[numClusters];
      for (int i = 0; i < numVectors; i++) {
        int c = assignments[i];
        counts[c]++;
        float[] v = vectors[i];
        float[] a = acc[c];
        for (int d = 0; d < NUM_DIMENSIONS; d++) a[d] += v[d];
      }
      for (int c = 0; c < numClusters; c++) {
        if (counts[c] > 0) {
          float inv = 1f / counts[c];
          float[] a = acc[c];
          float[] cen = centroids[c];
          for (int d = 0; d < NUM_DIMENSIONS; d++) cen[d] = a[d] * inv;
        }
      }
      System.out.println("  iter " + (iteration + 1) + ": " + changeCount.get() + " changed");
    }
  }

  private static float computeSquaredEuclideanDistance(final float[] vectorA, final float[] vectorB) {
    float sum = 0;
    for (int i = 0; i < NUM_DIMENSIONS; i++) {
      final float delta = vectorA[i] - vectorB[i];
      sum += delta * delta;
    }
    return sum;
  }
}
