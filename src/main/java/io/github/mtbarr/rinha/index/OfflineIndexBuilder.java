package io.github.mtbarr.rinha.index;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.RandomAccessFile;
import java.io.Reader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.IntStream;
import java.util.zip.GZIPInputStream;

public final class OfflineIndexBuilder {

  private record ParseResult(float[][] vectors, byte[] labels) {}


  private static final int CLUSTER_COUNT = 1024;
  private static final int DIMENSIONS = 14;
  private static final int TRAINING_SAMPLE_SIZE = 131_072;
  private static final int KMEANS_ITERATIONS = 10;
  private static final float QUANTIZATION_SCALE = 10_000f;
  private static final long RANDOM_SEED = 42;

  private OfflineIndexBuilder() {
    throw new AssertionError("Utility class should not be instantiated");
  }

  static void main(final String[] arguments) throws Exception {
    if (arguments.length < 2) {
      System.err.println("Usage: OfflineIndexBuilder <references.json.gz> <output.index.bin>");
      System.exit(1);
    }

    final String inputPath = arguments[0];
    final String outputPath = arguments[1];

    System.out.println("Reading references from: " + inputPath);
    final ParseResult data = parseReferences(inputPath);
    final int vectorCount = data.vectors.length;
    System.out.println("Loaded " + vectorCount + " vectors");

    System.out.println("Sampling " + TRAINING_SAMPLE_SIZE + " for K-Means training...");
    final float[][] trainingSample = reservoirSample(data.vectors, TRAINING_SAMPLE_SIZE);

    System.out.println("Running K-Means K=" + CLUSTER_COUNT + " ...");
    final float[][] centroids = kMeans(trainingSample);

    System.out.println("Assigning " + vectorCount + " vectors to nearest clusters (parallel)...");
    final int[] clusterAssignments = assignToClustersParallel(data.vectors, centroids);

    System.out.println("Building cluster ordering...");
    final int[] sortedOrder = buildClusterOrdering(clusterAssignments, vectorCount);

    final short[][] quantizedVectors = new short[vectorCount][DIMENSIONS];
    final byte[] reorderedLabels = new byte[vectorCount];
    final int[] reorderedOriginalIds = new int[vectorCount];
    for (int index = 0; index < vectorCount; index++) {
      final int sourceIndex = sortedOrder[index];
      for (int dimension = 0; dimension < DIMENSIONS; dimension++) {
        quantizedVectors[index][dimension] = quantize(data.vectors[sourceIndex][dimension]);
      }
      reorderedLabels[index] = data.labels[sourceIndex];
      reorderedOriginalIds[index] = sourceIndex;
    }

    System.out.println("Computing bounding boxes...");
    final short[][] bboxMinimum = new short[CLUSTER_COUNT][DIMENSIONS];
    final short[][] bboxMaximum = new short[CLUSTER_COUNT][DIMENSIONS];
    final int[] clusterOffsets = new int[CLUSTER_COUNT + 1];
    computeBoundingBoxes(
      quantizedVectors, clusterAssignments, sortedOrder,
      bboxMinimum, bboxMaximum, clusterOffsets, vectorCount
    );

    System.out.println("Writing index to: " + outputPath);
    writeIndex(
      outputPath, vectorCount, centroids,
      bboxMinimum, bboxMaximum, clusterOffsets,
      quantizedVectors, reorderedLabels, reorderedOriginalIds
    );

    System.out.println("Done! Index written successfully.");
  }

  private static ParseResult parseReferences(final String filePath) throws IOException {
    final List<float[]> vectorList = new ArrayList<>(3_000_000);
    final List<Byte> labelList = new ArrayList<>(3_000_000);

    try (final InputStream fileInput = new FileInputStream(filePath);
      final InputStream gzipInput = new GZIPInputStream(fileInput);
      final Reader reader = new InputStreamReader(gzipInput, StandardCharsets.UTF_8);
      final BufferedReader lineReader = new BufferedReader(reader)) {

      final StringBuilder text = new StringBuilder(300_000_000);
      String line;
      while ((line = lineReader.readLine()) != null) {
        text.append(line);
      }

      final String content = text.toString();
      int position = 0;

      while (true) {
        final int vectorTagStart = content.indexOf("\"vector\":[", position);
        if (vectorTagStart < 0) {
          break;
        }

        final float[] vector = new float[DIMENSIONS];
        int cursor = vectorTagStart + 10;

        for (int dimension = 0; dimension < DIMENSIONS; dimension++) {
          while (cursor < content.length() &&
                 (content.charAt(cursor) == ' ' || content.charAt(cursor) == '\n')) {
            cursor++;
          }
          final int valueEnd = dimension < DIMENSIONS - 1
                               ? content.indexOf(',', cursor)
                               : content.indexOf(']', cursor);
          vector[dimension] = Float.parseFloat(content.substring(cursor, valueEnd).trim());
          cursor = valueEnd + 1;
        }

        final int labelTag = content.indexOf("\"label\":\"", cursor);
        if (labelTag < 0) {
          break;
        }

        final int labelStart = labelTag + 9;
        final byte label = content.charAt(labelStart) == 'f' ? (byte) 1 : (byte) 0;

        vectorList.add(vector);
        labelList.add(label);

        final int quoteAfterLabel = content.indexOf('"', labelStart);
        position = quoteAfterLabel >= 0 ? quoteAfterLabel + 1 : cursor;
      }
    }

    final int total = vectorList.size();
    final float[][] vectors = vectorList.toArray(new float[total][]);
    final byte[] labels = new byte[total];
    for (int index = 0; index < total; index++) {
      labels[index] = labelList.get(index);
    }
    return new ParseResult(vectors, labels);
  }

  private static float[][] reservoirSample(final float[][] allVectors, final int sampleSize) {
    final int total = allVectors.length;
    if (total <= sampleSize) {
      return allVectors;
    }

    final Random random = new Random(RANDOM_SEED);
    final float[][] sample = new float[sampleSize][DIMENSIONS];

    for (int index = 0; index < sampleSize; index++) {
      System.arraycopy(allVectors[index], 0, sample[index], 0, DIMENSIONS);
    }

    for (int index = sampleSize; index < total; index++) {
      final int swapTarget = random.nextInt(index + 1);
      if (swapTarget < sampleSize) {
        System.arraycopy(allVectors[index], 0, sample[swapTarget], 0, DIMENSIONS);
      }
    }

    return sample;
  }

  private static float[][] kMeans(final float[][] trainingData) {
    final Random random = new Random(RANDOM_SEED);
    final int dataSize = trainingData.length;

    final float[][] centroids = new float[CLUSTER_COUNT][DIMENSIONS];
    final boolean[] taken = new boolean[dataSize];
    for (int cluster = 0; cluster < CLUSTER_COUNT; cluster++) {
      int chosen;
      do {
        chosen = random.nextInt(dataSize);
      } while (taken[chosen]);
      taken[chosen] = true;
      System.arraycopy(trainingData[chosen], 0, centroids[cluster], 0, DIMENSIONS);
    }

    final int[] assignments = new int[dataSize];

    for (int iteration = 0; iteration < KMEANS_ITERATIONS; iteration++) {
      final AtomicInteger changes = new AtomicInteger(0);

      IntStream.range(0, dataSize).parallel().forEach(dataIndex -> {
        final float[] vector = trainingData[dataIndex];
        int bestCluster = 0;
        float bestDistance = Float.MAX_VALUE;

        for (int cluster = 0; cluster < CLUSTER_COUNT; cluster++) {
          final float distance = squaredDistance(vector, centroids[cluster]);
          if (distance < bestDistance) {
            bestDistance = distance;
            bestCluster = cluster;
          }
        }

        if (assignments[dataIndex] != bestCluster) {
          assignments[dataIndex] = bestCluster;
          changes.incrementAndGet();
        }
      });

      if (iteration > 0 && changes.get() == 0) {
        System.out.println("  K-Means converged at iteration " + iteration);
        break;
      }

      final double[][] sums = new double[CLUSTER_COUNT][DIMENSIONS];
      final int[] clusterSizes = new int[CLUSTER_COUNT];

      for (int dataIndex = 0; dataIndex < dataSize; dataIndex++) {
        final float[] vector = trainingData[dataIndex];
        final int cluster = assignments[dataIndex];
        for (int d = 0; d < DIMENSIONS; d++) {
          sums[cluster][d] += vector[d];
        }
        clusterSizes[cluster]++;
      }

      for (int cluster = 0; cluster < CLUSTER_COUNT; cluster++) {
        if (clusterSizes[cluster] > 0) {
          for (int d = 0; d < DIMENSIONS; d++) {
            centroids[cluster][d] = (float) (sums[cluster][d] / clusterSizes[cluster]);
          }
        }
      }

      System.out.println("  Iteration " + (iteration + 1) + ": " + changes.get() + " changed");
    }

    return centroids;
  }

  private static int[] assignToClustersParallel(final float[][] vectors, final float[][] centroids) {
    final int[] assignments = new int[vectors.length];
    IntStream.range(0, vectors.length).parallel().forEach(vectorIndex -> {
      final float[] vector = vectors[vectorIndex];
      int bestCluster = 0;
      float bestDistance = Float.MAX_VALUE;
      for (int cluster = 0; cluster < CLUSTER_COUNT; cluster++) {
        final float distance = squaredDistance(vector, centroids[cluster]);
        if (distance < bestDistance) {
          bestDistance = distance;
          bestCluster = cluster;
        }
      }
      assignments[vectorIndex] = bestCluster;
    });
    return assignments;
  }

  private static float squaredDistance(final float[] vectorA, final float[] vectorB) {
    float sum = 0f;
    for (int d = 0; d < DIMENSIONS; d++) {
      final float diff = vectorA[d] - vectorB[d];
      sum += diff * diff;
    }
    return sum;
  }


  private static int[] buildClusterOrdering(final int[] assignments, final int vectorCount) {
    final int[] prefix = new int[CLUSTER_COUNT + 1];
    for (int index = 0; index < vectorCount; index++) {
      prefix[assignments[index] + 1]++;
    }
    for (int cluster = 0; cluster < CLUSTER_COUNT; cluster++) {
      prefix[cluster + 1] += prefix[cluster];
    }

    final int[] positions = Arrays.copyOf(prefix, CLUSTER_COUNT + 1);
    final int[] sortedIndices = new int[vectorCount];
    for (int index = 0; index < vectorCount; index++) {
      final int cluster = assignments[index];
      sortedIndices[positions[cluster]++] = index;
    }
    return sortedIndices;
  }


  private static void computeBoundingBoxes(
    final short[][] quantizedVectors,
    final int[] clusterAssignments,
    final int[] sortedOrder,
    final short[][] bboxMinimum,
    final short[][] bboxMaximum,
    final int[] clusterOffsets,
    final int vectorCount) {

    for (int cluster = 0; cluster < CLUSTER_COUNT; cluster++) {
      Arrays.fill(bboxMinimum[cluster], Short.MAX_VALUE);
      Arrays.fill(bboxMaximum[cluster], Short.MIN_VALUE);
    }

    int previousCluster = -1;
    for (int position = 0; position < vectorCount; position++) {
      final int sourceIndex = sortedOrder[position];
      final int vectorCluster = clusterAssignments[sourceIndex];

      if (vectorCluster != previousCluster) {
        if (previousCluster >= 0) {
          clusterOffsets[previousCluster + 1] = position;
        } else {
          clusterOffsets[0] = 0;
        }
        previousCluster = vectorCluster;
        clusterOffsets[previousCluster] = position;
      }

      final short[] vector = quantizedVectors[position];
      for (int d = 0; d < DIMENSIONS; d++) {
        if (vector[d] < bboxMinimum[previousCluster][d]) {
          bboxMinimum[previousCluster][d] = vector[d];
        }
        if (vector[d] > bboxMaximum[previousCluster][d]) {
          bboxMaximum[previousCluster][d] = vector[d];
        }
      }
    }

    if (previousCluster >= 0) {
      clusterOffsets[previousCluster + 1] = vectorCount;
    }
  }


  private static short quantize(final float value) {
    return (short) Math.clamp(
      Math.round(value * QUANTIZATION_SCALE),
      Short.MIN_VALUE,
      Short.MAX_VALUE
    );
  }


  private static void writeIndex(
    final String outputPath,
    final int vectorCount,
    final float[][] centroids,
    final short[][] bboxMinimum,
    final short[][] bboxMaximum,
    final int[] clusterOffsets,
    final short[][] quantizedVectors,
    final byte[] labels,
    final int[] originalIds
  ) throws IOException {

    final int centroidBytes = CLUSTER_COUNT * DIMENSIONS * 4;
    final int bboxBytes = 2 * CLUSTER_COUNT * DIMENSIONS * 2;
    final int offsetBytes = (CLUSTER_COUNT + 1) * 4;
    final int vectorBytes = vectorCount * DIMENSIONS * 2;
    final int labelBytes = vectorCount;
    final int originalIdBytes = vectorCount * 4;
    final long totalSize = 24L + centroidBytes + bboxBytes + offsetBytes
                           + vectorBytes + labelBytes + originalIdBytes;

    if (totalSize > Integer.MAX_VALUE) {
      throw new IOException("Index too large: " + totalSize);
    }

    try (final RandomAccessFile outputFile = new RandomAccessFile(outputPath, "rw");
      final FileChannel channel = outputFile.getChannel()) {

      final ByteBuffer buffer = channel.map(
        FileChannel.MapMode.READ_WRITE, 0, (int) totalSize);
      buffer.order(ByteOrder.LITTLE_ENDIAN);

      buffer.putInt(0x49564636);
      buffer.putInt(vectorCount);
      buffer.putInt(CLUSTER_COUNT);
      buffer.putInt(DIMENSIONS);
      buffer.putInt(DIMENSIONS);
      buffer.putFloat(QUANTIZATION_SCALE);

      for (int cluster = 0; cluster < CLUSTER_COUNT; cluster++) {
        for (int d = 0; d < DIMENSIONS; d++) {
          buffer.putFloat(centroids[cluster][d]);
        }
      }

      for (int cluster = 0; cluster < CLUSTER_COUNT; cluster++) {
        for (int d = 0; d < DIMENSIONS; d++) {
          buffer.putShort(bboxMinimum[cluster][d]);
        }
      }

      for (int cluster = 0; cluster < CLUSTER_COUNT; cluster++) {
        for (int d = 0; d < DIMENSIONS; d++) {
          buffer.putShort(bboxMaximum[cluster][d]);
        }
      }

      for (int index = 0; index <= CLUSTER_COUNT; index++) {
        buffer.putInt(clusterOffsets[index]);
      }

      for (int vector = 0; vector < vectorCount; vector++) {
        for (int d = 0; d < DIMENSIONS; d++) {
          buffer.putShort(quantizedVectors[vector][d]);
        }
      }

      buffer.put(labels);

      for (int vector = 0; vector < vectorCount; vector++) {
        buffer.putInt(originalIds[vector]);
      }
    }
  }

}
