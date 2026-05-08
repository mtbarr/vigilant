package io.github.mtbarr.rinha.index;

import jakarta.annotation.PostConstruct;
import jakarta.inject.Singleton;
import java.io.File;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel.MapMode;
import java.util.Arrays;

@Singleton
public class InvertedFileIndex {

  private static final int NUM_CLUSTERS = 512;
  private static final int NUM_DIMENSIONS = 14;
  private static final int NUM_PROBE_CLUSTERS = 24;
  private static final int NUM_PROBE_GRAY = 8;
  private static final int NUM_NEIGHBORS = 5;
  private static final float I16_SCALE = 32767.0f;

  private static final ValueLayout.OfInt INT_LE = ValueLayout.JAVA_INT.withOrder(ByteOrder.LITTLE_ENDIAN);
  private static final ValueLayout.OfLong LONG_LE = ValueLayout.JAVA_LONG.withOrder(ByteOrder.LITTLE_ENDIAN);

  private float[] ivfCentroidsFlat;
  private byte[] fraudLabels;
  private int[][] idsByCluster;
  private short[] vectorsI16;
  private volatile boolean isIndexReady = false;

  private final ThreadLocal<float[]> centroidDistanceBuffer = ThreadLocal.withInitial(
    () -> new float[NUM_CLUSTERS]
  );
  private final ThreadLocal<int[]> centroidOrderBuffer = ThreadLocal.withInitial(
    () -> new int[NUM_CLUSTERS]
  );
  private final ThreadLocal<float[]> queryScaledBuffer = ThreadLocal.withInitial(
    () -> new float[NUM_DIMENSIONS]
  );

  @PostConstruct
  void initialize() {
    final String indexPath = System.getenv().getOrDefault("INDEX_PATH", "/data/index.bin");
    try {
      loadIndexFromFile(indexPath);
      isIndexReady = true;
    } catch (final Exception exception) {
      System.err.println("Index not loaded: " + exception.getMessage());
      isIndexReady = false;
    }
  }

  private void loadIndexFromFile(final String filePath) throws IOException {
    final File indexFile = new File(filePath);
    final long fileLength = indexFile.length();
    try (final var randomAccessFile = new RandomAccessFile(indexFile, "r");
      final var fileChannel = randomAccessFile.getChannel()) {

      final MemorySegment indexSegment = fileChannel.map(MapMode.READ_ONLY, 0, fileLength, Arena.global());

      final int magicNumber = indexSegment.get(INT_LE, 0);
      if (magicNumber != 0x52494E44) {
        throw new IOException("Bad magic: " + Integer.toHexString(magicNumber));
      }
      final int versionNumber = indexSegment.get(INT_LE, 4);
      if (versionNumber != 2) {
        throw new IOException("Bad version: " + versionNumber + " (expected 2 for i16 index)");
      }
      final int storedClusters = indexSegment.get(INT_LE, 8);
      if (storedClusters != NUM_CLUSTERS) {
        throw new IOException("Expected K=" + NUM_CLUSTERS + " got " + storedClusters);
      }
      final int totalVectorCount = indexSegment.get(INT_LE, 12);
      final long labelsOffset = indexSegment.get(LONG_LE, 24);
      final long vectorsI16Offset = indexSegment.get(LONG_LE, 32);

      final long centroidsOffset = 40L;
      ivfCentroidsFlat = new float[NUM_CLUSTERS * NUM_DIMENSIONS];
      final MemorySegment centroidsSegment = indexSegment.asSlice(
        centroidsOffset, (long) NUM_CLUSTERS * NUM_DIMENSIONS * 4L);
      for (int i = 0; i < NUM_CLUSTERS * NUM_DIMENSIONS; i++) {
        ivfCentroidsFlat[i] = centroidsSegment.get(ValueLayout.JAVA_FLOAT.withOrder(ByteOrder.LITTLE_ENDIAN), (long) i * 4L);
      }

      fraudLabels = new byte[totalVectorCount];
      final MemorySegment labelsSegment = indexSegment.asSlice(labelsOffset, totalVectorCount);
      labelsSegment.asByteBuffer().get(fraudLabels);

      vectorsI16 = new short[totalVectorCount * NUM_DIMENSIONS];
      final MemorySegment vectorsSegment = indexSegment.asSlice(
        vectorsI16Offset, (long) totalVectorCount * NUM_DIMENSIONS * 2L);
      vectorsSegment.asByteBuffer().order(ByteOrder.LITTLE_ENDIAN).asShortBuffer().get(vectorsI16);

      final long invertedListsOffset = vectorsI16Offset + (long) totalVectorCount * NUM_DIMENSIONS * 2L;
      idsByCluster = new int[NUM_CLUSTERS][];
      long offset = invertedListsOffset;
      for (int c = 0; c < NUM_CLUSTERS; c++) {
        final int size = indexSegment.get(INT_LE, offset);
        final long idsOffset = offset + 4L;
        final int[] ids = new int[size];
        final MemorySegment idsSegment = indexSegment.asSlice(idsOffset, (long) size * 4L);
        idsSegment.asByteBuffer().order(ByteOrder.LITTLE_ENDIAN).asIntBuffer().get(ids);
        idsByCluster[c] = ids;
        final long dataSize = 4L + (long) size * 4L;
        offset += (dataSize + 3L) & ~3L;
      }

      int fraudCount = 0;
      for (byte label : fraudLabels) {
        if (label == 1) fraudCount++;
      }
      System.out.println("Index loaded: " + totalVectorCount + " vectors, " + fraudCount + " fraud (i16)");
    }
  }

  public int searchNearestNeighbors(
    final float[] queryVector,
    final int[] neighborIds,
    final float[] neighborDistances
  ) {
    final float[] centroidDistances = centroidDistanceBuffer.get();
    final int[] centroidOrder = centroidOrderBuffer.get();

    for (int ci = 0; ci < NUM_CLUSTERS; ci++) {
      final int base = ci * NUM_DIMENSIONS;
      float dist = 0.0f;
      for (int d = 0; d < NUM_DIMENSIONS; d++) {
        final float delta = queryVector[d] - ivfCentroidsFlat[base + d];
        dist += delta * delta;
      }
      centroidDistances[ci] = dist;
      centroidOrder[ci] = ci;
    }
    partialSortCentroids(centroidOrder, centroidDistances, NUM_PROBE_CLUSTERS);

    final float[] queryScaled = queryScaledBuffer.get();
    for (int d = 0; d < NUM_DIMENSIONS; d++) {
      queryScaled[d] = queryVector[d] * I16_SCALE;
    }

    Arrays.fill(neighborIds, -1);
    Arrays.fill(neighborDistances, Float.MAX_VALUE);

    for (int probe = 0; probe < NUM_PROBE_CLUSTERS; probe++) {
      final int ci = centroidOrder[probe];
      final int[] ids = idsByCluster[ci];

      for (int i = 0; i < ids.length; i++) {
        final float dist = i16SquaredDistance(queryScaled, ids[i]);
        if (dist < neighborDistances[NUM_NEIGHBORS - 1]) {
          insertIntoSortedArray(neighborIds, neighborDistances, NUM_NEIGHBORS, ids[i], dist);
        }
      }

      if (probe == NUM_PROBE_GRAY - 1) {
        int fc = 0;
        for (int k = 0; k < NUM_NEIGHBORS; k++) {
          if (neighborIds[k] >= 0 && fraudLabels[neighborIds[k]] == 1) {
            fc++;
          }
        }
        if (fc <= 1 || fc >= NUM_NEIGHBORS - 1) {
          return fc;
        }
      }
    }

    int fraudVoteCount = 0;
    for (int k = 0; k < NUM_NEIGHBORS; k++) {
      if (neighborIds[k] >= 0 && fraudLabels[neighborIds[k]] == 1) {
        fraudVoteCount++;
      }
    }
    return fraudVoteCount;
  }

  private float i16SquaredDistance(final float[] queryScaled, final int vectorId) {
    final int base = vectorId * NUM_DIMENSIONS;
    float sum = 0.0f;
    for (int d = 0; d < NUM_DIMENSIONS; d++) {
      final float diff = queryScaled[d] - vectorsI16[base + d];
      sum = Math.fma(diff, diff, sum);
    }
    return sum;
  }

  public boolean isReady() {
    return isIndexReady;
  }

  private static void insertIntoSortedArray(
    final int[] ids,
    final float[] dists,
    final int max,
    final int newId,
    final float newDist
  ) {
    int pos = max - 1;
    while (pos > 0 && dists[pos - 1] > newDist) {
      dists[pos] = dists[pos - 1];
      ids[pos] = ids[pos - 1];
      pos--;
    }
    dists[pos] = newDist;
    ids[pos] = newId;
  }

  private static void partialSortCentroids(
    final int[] order,
    final float[] dist,
    final int top
  ) {
    quickselect(order, dist, 0, order.length - 1, top);
    for (int i = 1; i < top; i++) {
      final int o = order[i];
      final float d = dist[o];
      int j = i - 1;
      while (j >= 0 && dist[order[j]] > d) {
        order[j + 1] = order[j];
        j--;
      }
      order[j + 1] = o;
    }
  }

  private static void quickselect(
    final int[] order,
    final float[] dist,
    final int l,
    final int r,
    final int k
  ) {
    if (l >= r) {
      return;
    }
    final int pi = partition(order, dist, l, r);
    final int rank = pi - l + 1;
    if (rank == k) {
      return;
    }
    if (k < rank) {
      quickselect(order, dist, l, pi - 1, k);
    } else {
      quickselect(order, dist, pi + 1, r, k - rank);
    }
  }

  private static int partition(
    final int[] order,
    final float[] dist,
    final int l,
    final int r
  ) {
    final float pivot = dist[order[r]];
    int si = l - 1;
    for (int i = l; i < r; i++) {
      if (dist[order[i]] <= pivot) {
        si++;
        final int tmp = order[si];
        order[si] = order[i];
        order[i] = tmp;
      }
    }
    final int tmp = order[si + 1];
    order[si + 1] = order[r];
    order[r] = tmp;
    return si + 1;
  }
}
