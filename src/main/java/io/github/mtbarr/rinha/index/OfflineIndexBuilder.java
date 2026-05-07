package io.github.mtbarr.rinha.index;

import java.io.FileInputStream;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.IntStream;
import java.util.zip.GZIPInputStream;

public final class OfflineIndexBuilder {

  private static final int NUM_CLUSTERS = 512;
  private static final int NUM_DIMENSIONS = 14;
  private static final int PQ_M = 7;
  private static final int PQ_SUB_D = 2;
  private static final int PQ_CODEBOOK_SIZE = 256;
  private static final int TRAINING_SAMPLE_SIZE = Integer.MAX_VALUE;
  private static final int IVF_MAX_ITERATIONS = 20;
  private static final int PQ_MAX_ITERATIONS = 20;
  private static final long RANDOM_SEED = 42L;

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

    final ExecutorService parserExecutor = Executors.newFixedThreadPool(2);
    final Future<float[][]> vectorsFuture = parserExecutor.submit(() -> parseFeatureVectors(rawData));
    final Future<byte[]> labelsFuture = parserExecutor.submit(() -> parseFraudLabels(rawData));
    final float[][] vectors = vectorsFuture.get();
    final byte[] labels = labelsFuture.get();
    parserExecutor.shutdownNow();

    final int numVectors = vectors.length;
    System.out.println("  " + numVectors + " vectors, fraud=" + countFraud(labels));

    System.out.println("K-Means++ init for IVF (K=" + NUM_CLUSTERS + ")...");
    final float[][] ivfCentroids = initializeKMeansPlusPlus(
      vectors,
      NUM_CLUSTERS,
      RANDOM_SEED
    );

    System.out.println("IVF Lloyd iterations...");
    final int[] clusterAssignments = new int[numVectors];
    runLloydAlgorithm(
      vectors,
      ivfCentroids,
      clusterAssignments,
      IVF_MAX_ITERATIONS
    );

    System.out.println("Training PQ codebooks...");
    final float[][][] pqCodebooks = trainProductQuantization(
      vectors,
      PQ_MAX_ITERATIONS,
      RANDOM_SEED
    );

    System.out.println("Encoding vectors...");
    final byte[][] pqCodes = encodeAllVectors(vectors, pqCodebooks);

    System.out.println("Building inverted lists...");
    final int[][] idsByCluster = new int[NUM_CLUSTERS][];
    final byte[][] codesByCluster = new byte[NUM_CLUSTERS][];
    buildInvertedLists(
      numVectors,
      clusterAssignments,
      pqCodes,
      idsByCluster,
      codesByCluster
    );

    System.out.println("Writing index...");
    writeIndexToFile(
      outputPath,
      numVectors,
      ivfCentroids,
      pqCodebooks,
      idsByCluster,
      codesByCluster,
      vectors,
      labels
    );

    System.out.println("Done.");
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

  private static float[][][] trainProductQuantization(
    final float[][] vectors,
    final int maxIterations,
    final long seed
  ) {
    final float[][][] codebooks = new float[PQ_M][PQ_CODEBOOK_SIZE][PQ_SUB_D];

    IntStream.range(0, PQ_M).parallel().forEach(subspaceIndex -> {
      final int offset = subspaceIndex * PQ_SUB_D;
      final float[][] subVectors = new float[vectors.length][PQ_SUB_D];
      for (int i = 0; i < vectors.length; i++) {
        subVectors[i][0] = vectors[i][offset];
        subVectors[i][1] = vectors[i][offset + 1];
      }
      codebooks[subspaceIndex] = runKMeansOnSubspace(
        subVectors,
        PQ_CODEBOOK_SIZE,
        seed + subspaceIndex,
        maxIterations
      );
    });
    return codebooks;
  }

  private static float[][] runKMeansOnSubspace(
    final float[][] data,
    final int numClusters,
    final long seed,
    final int maxIterations
  ) {
    final int numPoints = data.length;
    final java.util.Random rng = new java.util.Random(seed);
    final float[][] centroids = initPlusPlus(data, numClusters, rng);
    final int[] assignments = new int[numPoints];

    for (int iteration = 0; iteration < maxIterations; iteration++) {
      final java.util.concurrent.atomic.AtomicInteger changeCount = new java.util.concurrent.atomic.AtomicInteger(0);
      java.util.stream.IntStream.range(0, numPoints).parallel().forEach(i -> {
        float bestDistance = Float.MAX_VALUE;
        int bestCluster = 0;
        for (int c = 0; c < numClusters; c++) {
          final float d0 = data[i][0] - centroids[c][0];
          final float d1 = data[i][1] - centroids[c][1];
          final float distance = d0 * d0 + d1 * d1;
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
        break;
      }
      final double[][] clusterSums = new double[numClusters][PQ_SUB_D];
      final int[] clusterCounts = new int[numClusters];
      for (int i = 0; i < numPoints; i++) {
        final int cluster = assignments[i];
        clusterCounts[cluster]++;
        clusterSums[cluster][0] += data[i][0];
        clusterSums[cluster][1] += data[i][1];
      }
      for (int c = 0; c < numClusters; c++) {
        if (clusterCounts[c] > 0) {
          final float inv = 1f / clusterCounts[c];
          centroids[c][0] = (float) (clusterSums[c][0] * inv);
          centroids[c][1] = (float) (clusterSums[c][1] * inv);
        }
      }
    }
    return centroids;
  }

  private static float[][] initPlusPlus(final float[][] data, final int K, final java.util.Random rng) {
    final int N = data.length;
    final float[][] centroids = new float[K][PQ_SUB_D];
    final float[] distances = new float[N];

    int first = rng.nextInt(N);
    System.arraycopy(data[first], 0, centroids[0], 0, PQ_SUB_D);
    for (int i = 0; i < N; i++) {
      distances[i] = squaredDist(data[i], centroids[0]);
    }

    for (int k = 1; k < K; k++) {
      double total = 0;
      for (float d : distances) total += d;
      double threshold = rng.nextDouble() * total;
      double cumulative = 0;
      int chosen = N - 1;
      for (int i = 0; i < N; i++) {
        cumulative += distances[i];
        if (cumulative >= threshold) {
          chosen = i;
          break;
        }
      }
      System.arraycopy(data[chosen], 0, centroids[k], 0, PQ_SUB_D);
      for (int i = 0; i < N; i++) {
        float d = squaredDist(data[i], centroids[k]);
        if (d < distances[i]) distances[i] = d;
      }
    }
    return centroids;
  }

  private static float squaredDist(final float[] a, final float[] b) {
    final float d0 = a[0] - b[0];
    final float d1 = a[1] - b[1];
    return d0 * d0 + d1 * d1;
  }

  private static byte[][] encodeAllVectors(final float[][] vectors, final float[][][] codebooks) {
    final int numVectors = vectors.length;
    final byte[][] codes = new byte[numVectors][PQ_M];
    IntStream.range(0, numVectors).parallel().forEach(i -> {
      for (int subspaceIndex = 0; subspaceIndex < PQ_M; subspaceIndex++) {
        final int offset = subspaceIndex * PQ_SUB_D;
        final float query0 = vectors[i][offset];
        final float query1 = vectors[i][offset + 1];
        float bestDistance = Float.MAX_VALUE;
        int bestCentroid = 0;
        for (int c = 0; c < PQ_CODEBOOK_SIZE; c++) {
          final float delta0 = query0 - codebooks[subspaceIndex][c][0];
          final float delta1 = query1 - codebooks[subspaceIndex][c][1];
          final float distance = delta0 * delta0 + delta1 * delta1;
          if (distance < bestDistance) {
            bestDistance = distance;
            bestCentroid = c;
          }
        }
        codes[i][subspaceIndex] = (byte) bestCentroid;
      }
    });
    return codes;
  }

  private static void buildInvertedLists(
    final int numVectors,
    final int[] assignments,
    final byte[][] codes,
    final int[][] idsByCluster,
    final byte[][] codesByCluster
  ) {
    final int[] clusterCounts = new int[NUM_CLUSTERS];
    for (int i = 0; i < numVectors; i++) {
      clusterCounts[assignments[i]]++;
    }
    for (int c = 0; c < NUM_CLUSTERS; c++) {
      idsByCluster[c] = new int[clusterCounts[c]];
      codesByCluster[c] = new byte[clusterCounts[c] * PQ_M];
    }
    final int[] positionTracker = new int[NUM_CLUSTERS];
    for (int i = 0; i < numVectors; i++) {
      final int cluster = assignments[i];
      final int position = positionTracker[cluster]++;
      idsByCluster[cluster][position] = i;
      System.arraycopy(codes[i], 0, codesByCluster[cluster], position * PQ_M, PQ_M);
    }
  }

  private static void writeIndexToFile(
    final String filePath,
    final int numVectors,
    final float[][] ivfCentroids,
    final float[][][] pqCodebooks,
    final int[][] idsByCluster,
    final byte[][] codesByCluster,
    final float[][] vectors,
    final byte[] labels
  ) throws IOException {
    final long centroidsOffset = 36L;
    final long codebooksOffset = centroidsOffset + (long) NUM_CLUSTERS * NUM_DIMENSIONS * 4L;
    final long vectorsOffset = codebooksOffset + (long) PQ_M * PQ_CODEBOOK_SIZE * PQ_SUB_D * 4L + 4L;
    final long labelsOffset = vectorsOffset + (long) numVectors * NUM_DIMENSIONS * 4L;

    long invertedListsOffset = labelsOffset + numVectors;
    if (invertedListsOffset % 4L != 0) {
      invertedListsOffset = (invertedListsOffset + 3L) & ~3L;
    }
    final long[] clusterListOffsets = new long[NUM_CLUSTERS];
    for (int c = 0; c < NUM_CLUSTERS; c++) {
      clusterListOffsets[c] = invertedListsOffset;
      final long dataSize = 4L + (long) idsByCluster[c].length * 4L
                            + (long) idsByCluster[c].length * PQ_M;
      final long paddedSize = (dataSize + 3L) & ~3L;
      invertedListsOffset += paddedSize;
    }
    final long totalFileSize = invertedListsOffset;

    try (final var randomAccessFile = new RandomAccessFile(filePath, "rw");
      final var fileChannel = randomAccessFile.getChannel()) {
      final ByteBuffer buffer = fileChannel.map(
        FileChannel.MapMode.READ_WRITE,
        0,
        totalFileSize
      );
      buffer.order(ByteOrder.LITTLE_ENDIAN);

      buffer.putInt(0x52494E44);
      buffer.putInt(1);
      buffer.putInt(NUM_CLUSTERS);
      buffer.putInt(numVectors);
      buffer.putLong(vectorsOffset);
      buffer.putLong(labelsOffset);

      buffer.position((int) centroidsOffset);
      for (int c = 0; c < NUM_CLUSTERS; c++) {
        for (int d = 0; d < NUM_DIMENSIONS; d++) {
          buffer.putFloat(ivfCentroids[c][d]);
        }
      }

      buffer.position((int) codebooksOffset);
      for (int m = 0; m < PQ_M; m++) {
        for (int c = 0; c < PQ_CODEBOOK_SIZE; c++) {
          for (int d = 0; d < PQ_SUB_D; d++) {
            buffer.putFloat(pqCodebooks[m][c][d]);
          }
        }
      }

      buffer.position((int) (vectorsOffset - 4L));
      buffer.putInt(numVectors);
      buffer.position((int) vectorsOffset);
      for (int i = 0; i < numVectors; i++) {
        for (int d = 0; d < NUM_DIMENSIONS; d++) {
          buffer.putFloat(vectors[i][d]);
        }
      }

      buffer.position((int) labelsOffset);
      buffer.put(labels);

      for (int c = 0; c < NUM_CLUSTERS; c++) {
        buffer.position((int) clusterListOffsets[c]);
        buffer.putInt(idsByCluster[c].length);
        for (final int id : idsByCluster[c]) {
          buffer.putInt(id);
        }
        buffer.put(codesByCluster[c]);
        final int dataLen = 4 + idsByCluster[c].length * 4 + idsByCluster[c].length * PQ_M;
        final int padding = ((dataLen + 3) & ~3) - dataLen;
        for (int p = 0; p < padding; p++) {
          buffer.put((byte) 0);
        }
      }
    }

    System.out.println("  Index file size: " + totalFileSize + " bytes");
    System.out.println("  Vectors section: " + vectorsOffset + " - " + labelsOffset
                       + " (" + ((labelsOffset - vectorsOffset) / 1024 / 1024) + " MB)");
    System.out.println("  Labels section: " + labelsOffset + " - " + clusterListOffsets[0]
                       + " (" + numVectors + " bytes)");
    System.out.println("  Inverted lists: " + clusterListOffsets[0] + " - " + totalFileSize
                       + " (" + ((totalFileSize - clusterListOffsets[0]) / 1024 / 1024) + " MB)");
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