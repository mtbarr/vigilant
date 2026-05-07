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

  private static final int NUM_CLUSTERS = 1024;
  private static final int NUM_DIMENSIONS = 14;
  private static final int PQ_M = 7;
  private static final int PQ_SUB_D = 2;
  private static final int PQ_CODEBOOK_SIZE = 256;
  private static final int TRAINING_SAMPLE_SIZE = 50_000;
  private static final int IVF_MAX_ITERATIONS = 25;
  private static final int PQ_MAX_ITERATIONS = 20;
  private static final long RANDOM_SEED = 0xdeadbeefcafebabeL;

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
      TRAINING_SAMPLE_SIZE,
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
    final int sampleSize,
    final long seed
  ) {
    final int numVectors = vectors.length;
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

    centroids[0] = vectors[sampleIndices[0]].clone();

    for (int clusterIndex = 1; clusterIndex < numClusters; clusterIndex++) {
      final float[] lastCentroid = centroids[clusterIndex - 1];
      double totalDistance = 0;
      for (int i = 0; i < effectiveSampleSize; i++) {
        final double distance = computeSquaredEuclideanDistance(vectors[sampleIndices[i]], lastCentroid);
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
      centroids[clusterIndex] = vectors[sampleIndices[chosenIndex]].clone();
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
      final double[][] centroidSums = new double[numClusters][NUM_DIMENSIONS];
      final int[] clusterCounts = new int[numClusters];

      final int processors = Runtime.getRuntime().availableProcessors();
      final int chunkSize = (numVectors + processors - 1) / processors;
      IntStream.range(0, processors).parallel().forEach(thread -> {
        final int start = thread * chunkSize;
        final int end = Math.min(start + chunkSize, numVectors);
        final double[][] localSums = new double[numClusters][NUM_DIMENSIONS];
        final int[] localCounts = new int[numClusters];
        for (int i = start; i < end; i++) {
          final int cluster = assignments[i];
          localCounts[cluster]++;
          for (int d = 0; d < NUM_DIMENSIONS; d++) {
            localSums[cluster][d] += vectors[i][d];
          }
        }
        synchronized (centroidSums) {
          for (int c = 0; c < numClusters; c++) {
            if (localCounts[c] > 0) {
              clusterCounts[c] += localCounts[c];
              for (int d = 0; d < NUM_DIMENSIONS; d++) {
                centroidSums[c][d] += localSums[c][d];
              }
            }
          }
        }
      });
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
    final long[] randomState = {seed};
    final float[][] centroids = new float[numClusters][PQ_SUB_D];
    final boolean[] taken = new boolean[numPoints];
    for (int i = 0; i < numClusters; i++) {
      int index;
      do {
        randomState[0] = randomState[0] * 6364136223846793005L + 1442695040888963407L;
        index = (int) ((randomState[0] >>> 33) % numPoints);
      } while (taken[index]);
      taken[index] = true;
      centroids[i][0] = data[index][0];
      centroids[i][1] = data[index][1];
    }

    final int[] assignments = new int[numPoints];
    for (int iteration = 0; iteration < maxIterations; iteration++) {
      int changeCount = 0;
      for (int i = 0; i < numPoints; i++) {
        float bestDistance = Float.MAX_VALUE;
        int bestCluster = 0;
        for (int c = 0; c < numClusters; c++) {
          final float delta0 = data[i][0] - centroids[c][0];
          final float delta1 = data[i][1] - centroids[c][1];
          final float distance = delta0 * delta0 + delta1 * delta1;
          if (distance < bestDistance) {
            bestDistance = distance;
            bestCluster = c;
          }
        }
        if (assignments[i] != bestCluster) {
          assignments[i] = bestCluster;
          changeCount++;
        }
      }
      if (iteration > 0 && changeCount == 0) {
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
          centroids[c][0] = (float) (clusterSums[c][0] / clusterCounts[c]);
          centroids[c][1] = (float) (clusterSums[c][1] / clusterCounts[c]);
        }
      }
    }
    return centroids;
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