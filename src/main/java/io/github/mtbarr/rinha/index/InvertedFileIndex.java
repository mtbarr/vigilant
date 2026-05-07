package io.github.mtbarr.rinha.index;

import static java.lang.ThreadLocal.withInitial;

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

  private final ThreadLocal<long[]> centroidDistanceBuffer = withInitial(() -> new long[0]);
  private final ThreadLocal<int[]> clusterVisitGeneration = withInitial(() -> new int[0]);
  private final ThreadLocal<int[]> requestGenerationCounter = withInitial(() -> new int[1]);
  private final ThreadLocal<short[]> quantizedQueryBuffer = withInitial(() -> new short[DIMENSIONS]);
  private final ThreadLocal<int[]> probeIndexBuffer = withInitial(() -> new int[FULL_PROBE_COUNT]);
  private final ThreadLocal<NearestNeighborsBuffer> nearestNeighborsBuffer = withInitial(NearestNeighborsBuffer::new);


  @PostConstruct
  void initialize() {
    final String indexPath = System.getenv().getOrDefault("INDEX_PATH", "/data/index.bin");
    try {
      loadIndex(indexPath);
      isIndexReady = true;
    } catch (final IOException exception) {
      System.err.println("Index not loaded: " + exception.getMessage());
    }
  }

  private void loadIndex(final String indexPath) throws IOException {
    final File indexFile = new File(indexPath);
    try (final RandomAccessFile file = new RandomAccessFile(indexFile, "r");
      final FileChannel channel = file.getChannel()) {

      final ByteBuffer buf = channel.map(FileChannel.MapMode.READ_ONLY, 0, indexFile.length());
      buf.order(ByteOrder.LITTLE_ENDIAN);

      final int magic = buf.getInt();
      if (magic != 0x49564636) {
        throw new IOException("Unknown index format: " + Integer.toHexString(magic));
      }

      vectorCount = buf.getInt();
      clusterCount = buf.getInt();

      final int dimensions = buf.getInt();
      if (dimensions != DIMENSIONS) {
        throw new IOException("Expected " + DIMENSIONS + " dimensions, got " + dimensions);
      }

      buf.getInt();
      final float scale = buf.getFloat();
      if (Math.abs(scale - QUANTIZATION_SCALE) > 0.1f) {
        throw new IOException(
          "Expected scale " + QUANTIZATION_SCALE + ", got " + scale);
      }

      final float[][] rawCentroids = new float[clusterCount][DIMENSIONS];
      for (int c = 0; c < clusterCount; c++) {
        for (int d = 0; d < DIMENSIONS; d++) {
          rawCentroids[c][d] = buf.getFloat();
        }
      }

      bboxMinimum = new short[clusterCount][DIMENSIONS];
      bboxMaximum = new short[clusterCount][DIMENSIONS];

      for (int c = 0; c < clusterCount; c++) {
        for (int d = 0; d < DIMENSIONS; d++) {
          bboxMinimum[c][d] = buf.getShort();
        }
      }
      for (int c = 0; c < clusterCount; c++) {
        for (int d = 0; d < DIMENSIONS; d++) {
          bboxMaximum[c][d] = buf.getShort();
        }
      }

      final int[] clusterOffsets = new int[clusterCount + 1];
      for (int i = 0; i <= clusterCount; i++) {
        clusterOffsets[i] = buf.getInt();
      }
      clusterStartOffset = new int[clusterCount];
      clusterEndOffset = new int[clusterCount];
      for (int c = 0; c < clusterCount; c++) {
        clusterStartOffset[c] = clusterOffsets[c];
        clusterEndOffset[c] = clusterOffsets[c + 1];
      }

      dimensionsByVector = new short[DIMENSIONS][vectorCount];
      for (int v = 0; v < vectorCount; v++) {
        for (int d = 0; d < DIMENSIONS; d++) {
          dimensionsByVector[d][v] = buf.getShort();
        }
      }

      vectorLabels = new byte[vectorCount];
      for (int v = 0; v < vectorCount; v++) {
        vectorLabels[v] = buf.get();
      }

      vectorOriginalIds = new int[vectorCount];
      for (int v = 0; v < vectorCount; v++) {
        vectorOriginalIds[v] = buf.getInt();
      }

      quantizedCentroids = new short[clusterCount][DIMENSIONS];
      for (int c = 0; c < clusterCount; c++) {
        for (int d = 0; d < DIMENSIONS; d++) {
          quantizedCentroids[c][d] = quantize(rawCentroids[c][d]);
        }
      }
    }
  }


  public int search(final float[] queryFloat) {

    long[] centroidDists = centroidDistanceBuffer.get();
    int[] visitedGen = clusterVisitGeneration.get();

    if (centroidDists.length < clusterCount) {
      centroidDists = new long[clusterCount];
      centroidDistanceBuffer.set(centroidDists);
      visitedGen = new int[clusterCount];
      clusterVisitGeneration.set(visitedGen);
    }

    final int[] genHolder = requestGenerationCounter.get();
    final int currentGen = ++genHolder[0];

    final short[] query = quantizedQueryBuffer.get();
    final int[] probes = probeIndexBuffer.get();
    final NearestNeighborsBuffer buf = nearestNeighborsBuffer.get();
    buf.reset();

    for (int d = 0; d < DIMENSIONS; d++) {
      query[d] = quantize(queryFloat[d]);
    }

    for (int c = 0; c < clusterCount; c++) {
      centroidDists[c] = vectorSquaredDistance(query, quantizedCentroids[c]);
    }

    selectTopN(centroidDists, FAST_PROBE_COUNT, probes);
    for (int i = 0; i < FAST_PROBE_COUNT; i++) {
      final int c = probes[i];
      visitedGen[c] = currentGen;
      scanCluster(query, c, buf);
    }

    final int fastVotes = buf.countFraudVotes();
    if (fastVotes == 2 || fastVotes == 3) {
      selectTopN(centroidDists, FULL_PROBE_COUNT, probes);
      for (int i = 0; i < FULL_PROBE_COUNT; i++) {
        final int c = probes[i];
        if (visitedGen[c] != currentGen) {
          visitedGen[c] = currentGen;
          scanCluster(query, c, buf);
        }
      }
    }

    for (int c = 0; c < clusterCount; c++) {
      if (visitedGen[c] == currentGen) {
        continue;
      }
      if (bboxLowerBound(query, c) <= buf.worstDistance()) {
        scanCluster(query, c, buf);
      }
    }

    return buf.countFraudVotes();
  }

  private void scanCluster(final short[] query, final int clusterIdx,
                           final NearestNeighborsBuffer buf) {
    final int end = clusterEndOffset[clusterIdx];
    for (int v = clusterStartOffset[clusterIdx]; v < end; v++) {
      buf.tryInsert(vectorSquaredDistance(query, v), vectorLabels[v], vectorOriginalIds[v]);
    }
  }


  private long vectorSquaredDistance(final short[] q, final int vecIdx) {
    long sum = 0L;

    int d = 0;
    for (; d + 3 < DIMENSIONS; d += 4) {
      final int d0 = q[d] - dimensionsByVector[d][vecIdx];
      final int d1 = q[d + 1] - dimensionsByVector[d + 1][vecIdx];
      final int d2 = q[d + 2] - dimensionsByVector[d + 2][vecIdx];
      final int d3 = q[d + 3] - dimensionsByVector[d + 3][vecIdx];
      sum += (long) d0 * d0 + (long) d1 * d1 + (long) d2 * d2 + (long) d3 * d3;
    }
    for (; d < DIMENSIONS; d++) {
      final int diff = q[d] - dimensionsByVector[d][vecIdx];
      sum += (long) diff * diff;
    }
    return sum;
  }


  private long vectorSquaredDistance(final short[] a, final short[] b) {
    long sum = 0L;
    int d = 0;
    for (; d + 3 < DIMENSIONS; d += 4) {
      final int d0 = a[d] - b[d];
      final int d1 = a[d + 1] - b[d + 1];
      final int d2 = a[d + 2] - b[d + 2];
      final int d3 = a[d + 3] - b[d + 3];
      sum += (long) d0 * d0 + (long) d1 * d1 + (long) d2 * d2 + (long) d3 * d3;
    }
    for (; d < DIMENSIONS; d++) {
      final int diff = a[d] - b[d];
      sum += (long) diff * diff;
    }
    return sum;
  }


  private long bboxLowerBound(final short[] query, final int clusterIdx) {
    long sum = 0L;
    for (int d = 0; d < DIMENSIONS; d++) {
      final int q = query[d];
      final int lo = bboxMinimum[clusterIdx][d];
      final int hi = bboxMaximum[clusterIdx][d];
      final int diff = (q < lo) ? (q - lo) : (q > hi) ? (q - hi) : 0;
      sum += (long) diff * diff;
    }
    return sum;
  }


  private void selectTopN(final long[] dists, final int n, final int[] out) {
    for (int i = 0; i < n; i++) {
      out[i] = i;
    }
    for (int i = n / 2 - 1; i >= 0; i--) {
      heapSiftDown(dists, out, i, n);
    }
    for (int i = n; i < clusterCount; i++) {
      if (dists[i] < dists[out[0]]) {
        out[0] = i;
        heapSiftDown(dists, out, 0, n);
      }
    }
  }

  private static void heapSiftDown(
    final long[] dists,
    final int[] heap,
    int pos,
    final int size
  ) {

    while (true) {
      int largest = pos;
      final int left = 2 * pos + 1;
      final int right = 2 * pos + 2;
      if (left < size && dists[heap[left]] > dists[heap[largest]]) {
        largest = left;
      }
      if (right < size && dists[heap[right]] > dists[heap[largest]]) {
        largest = right;
      }
      if (largest == pos) {
        break;
      }
      final int tmp = heap[pos];
      heap[pos] = heap[largest];
      heap[largest] = tmp;
      pos = largest;
    }
  }


  private static short quantize(final float value) {
    if (value >= Short.MAX_VALUE) {
      return Short.MAX_VALUE;
    }
    if (value <= Short.MIN_VALUE) {
      return Short.MIN_VALUE;
    }
    return (short) Math.round(value * QUANTIZATION_SCALE);
  }

  public boolean isReady() {
    return isIndexReady;
  }
}