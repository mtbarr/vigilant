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

  private static final int NUM_CLUSTERS = 2048;
  private static final int NUM_DIMENSIONS = 14;
  private static final int PQ_M = 7;
  private static final int PQ_SUB_D = 2;
  private static final int PQ_CODEBOOK_SIZE = 256;
  private static final int NUM_PROBE_CLUSTERS = 24;
  private static final int NUM_PROBE_GRAY = 8;
  private static final int NUM_NEIGHBORS = 10;
  private static final int RERANK_TOP = 7;

  private static final ValueLayout.OfInt INT_LE = ValueLayout.JAVA_INT.withOrder(ByteOrder.LITTLE_ENDIAN);
  private static final ValueLayout.OfLong LONG_LE = ValueLayout.JAVA_LONG.withOrder(ByteOrder.LITTLE_ENDIAN);
  private static final ValueLayout.OfFloat FLOAT_LE = ValueLayout.JAVA_FLOAT.withOrder(ByteOrder.LITTLE_ENDIAN);

  private float[] ivfCentroidsFlat;
  private float[][] pqCodebooksFlat;
  private byte[] fraudLabels;
  private int[][] idsByCluster;
  private byte[][] codesByCluster;
  private volatile boolean isIndexReady = false;
  private MemorySegment indexSegment;
  private long vectorsOffset;

  private final ThreadLocal<float[]> centroidDistanceBuffer = ThreadLocal.withInitial(
    () -> new float[NUM_CLUSTERS]
  );
  private final ThreadLocal<int[]> centroidOrderBuffer = ThreadLocal.withInitial(
    () -> new int[NUM_CLUSTERS]
  );
  private final ThreadLocal<float[][]> adcLookupTable = ThreadLocal.withInitial(
    () -> new float[PQ_M][PQ_CODEBOOK_SIZE]
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
      if (versionNumber != 1) {
        throw new IOException("Bad version: " + versionNumber);
      }
      final int storedClusters = indexSegment.get(INT_LE, 8);
      if (storedClusters != NUM_CLUSTERS) {
        throw new IOException("Expected K=" + NUM_CLUSTERS + " got " + storedClusters);
      }
      final int totalVectorCount = indexSegment.get(INT_LE, 12);
      this.vectorsOffset = indexSegment.get(LONG_LE, 16);
      final long labelsOffset = indexSegment.get(LONG_LE, 24);

      final long centroidsOffset = 36L;
      ivfCentroidsFlat = new float[NUM_CLUSTERS * NUM_DIMENSIONS];
      final MemorySegment centroidsSegment = indexSegment.asSlice(
        centroidsOffset, (long) NUM_CLUSTERS * NUM_DIMENSIONS * 4L);
      for (int i = 0; i < NUM_CLUSTERS * NUM_DIMENSIONS; i++) {
        ivfCentroidsFlat[i] = centroidsSegment.get(FLOAT_LE, (long) i * 4L);
      }

      final long codebooksOffset = centroidsOffset + (long) NUM_CLUSTERS * NUM_DIMENSIONS * 4L;
      pqCodebooksFlat = new float[PQ_M][PQ_CODEBOOK_SIZE * PQ_SUB_D];
      final MemorySegment codebooksSegment = indexSegment.asSlice(
        codebooksOffset, (long) PQ_M * PQ_CODEBOOK_SIZE * PQ_SUB_D * 4L);
      for (int m = 0; m < PQ_M; m++) {
        for (int c = 0; c < PQ_CODEBOOK_SIZE; c++) {
          for (int d = 0; d < PQ_SUB_D; d++) {
            pqCodebooksFlat[m][c * PQ_SUB_D + d] = codebooksSegment.get(
              FLOAT_LE, (long) (m * PQ_CODEBOOK_SIZE * PQ_SUB_D + c * PQ_SUB_D + d) * 4L);
          }
        }
      }

      fraudLabels = new byte[totalVectorCount];
      final MemorySegment labelsSegment = indexSegment.asSlice(labelsOffset, totalVectorCount);
      labelsSegment.asByteBuffer().get(fraudLabels);

      // --- Copy inverted lists to heap: IDs (int[]) and PQ codes (byte[]) per cluster ---
      final long invertedListsOffset = labelsOffset + totalVectorCount;
      idsByCluster = new int[NUM_CLUSTERS][];
      codesByCluster = new byte[NUM_CLUSTERS][];
      long offset = invertedListsOffset;
      for (int c = 0; c < NUM_CLUSTERS; c++) {
        final int size = indexSegment.get(INT_LE, offset);
        final long idsOffset = offset + 4L;
        final long codesOffset = idsOffset + (long) size * 4L;
        final int[] ids = new int[size];
        final byte[] codes = new byte[size * PQ_M];

        final MemorySegment idsSegment = indexSegment.asSlice(idsOffset, (long) size * 4L);
        final MemorySegment codesSegment2 = indexSegment.asSlice(codesOffset, (long) size * PQ_M);
        idsSegment.asByteBuffer().order(ByteOrder.LITTLE_ENDIAN).asIntBuffer().get(ids);
        codesSegment2.asByteBuffer().get(codes);

        idsByCluster[c] = ids;
        codesByCluster[c] = codes;
        final long dataSize = 4L + (long) size * 4L + (long) size * PQ_M;
        offset += (dataSize + 3L) & ~3L;
      }
      this.indexSegment = indexSegment;

      int fraudCount = 0;
      for (byte label : fraudLabels) {
        if (label == 1) {
          fraudCount++;
        }
      }
      System.out.println("Index loaded: " + totalVectorCount + " vectors, " + fraudCount + " fraud");
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
      float dist = 0.0f;
      final int base = ci * NUM_DIMENSIONS;
      for (int d = 0; d < NUM_DIMENSIONS; d++) {
        final float delta = queryVector[d] - ivfCentroidsFlat[base + d];
        dist = Math.fma(delta, delta, dist);
      }
      centroidDistances[ci] = dist;
      centroidOrder[ci] = ci;
    }
    partialSortCentroids(centroidOrder, centroidDistances, NUM_PROBE_CLUSTERS);

    final float[][] adcTable = adcLookupTable.get();
    buildAdcLookupTable(queryVector, adcTable);

    Arrays.fill(neighborIds, -1);
    Arrays.fill(neighborDistances, Float.MAX_VALUE);

    for (int probe = 0; probe < NUM_PROBE_CLUSTERS; probe++) {
      final int ci = centroidOrder[probe];
      final int[] ids = idsByCluster[ci];
      final byte[] codes = codesByCluster[ci];

      for (int i = 0; i < ids.length; i++) {
        final float approx = adcDistance(adcTable, codes, i * PQ_M);
        if (approx < neighborDistances[NUM_NEIGHBORS - 1]) {
          insertIntoSortedArray(neighborIds, neighborDistances, NUM_NEIGHBORS, ids[i], approx);
        }
      }

      if (probe == NUM_PROBE_GRAY - 1) {
        int fc = 0;
        for (int k = 0; k < RERANK_TOP; k++) {
          if (neighborIds[k] >= 0 && fraudLabels[neighborIds[k]] == 1) {
            fc++;
          }
        }
        if (fc <= 1 || fc >= RERANK_TOP - 1) {
          return fc;
        }
      }
    }

    for (int k = 0; k < NUM_NEIGHBORS; k++) {
      if (neighborIds[k] >= 0) {
        final long baseOff = vectorsOffset + (long) neighborIds[k] * NUM_DIMENSIONS * 4L;
        float exactDist = 0f;
        for (int d = 0; d < NUM_DIMENSIONS; d++) {
          final float delta = queryVector[d] - indexSegment.get(FLOAT_LE, baseOff + d * 4L);
          exactDist = Math.fma(delta, delta, exactDist);
        }
        neighborDistances[k] = exactDist;
      } else {
        neighborDistances[k] = Float.MAX_VALUE;
      }
    }

    for (int i = 0; i < RERANK_TOP; i++) {
      int best = i;
      for (int j = i + 1; j < NUM_NEIGHBORS; j++) {
        if (neighborDistances[j] < neighborDistances[best]) {
          best = j;
        }
      }
      final int tmpId = neighborIds[i];
      neighborIds[i] = neighborIds[best];
      neighborIds[best] = tmpId;
      final float tmpDist = neighborDistances[i];
      neighborDistances[i] = neighborDistances[best];
      neighborDistances[best] = tmpDist;
    }

    int fraudVoteCount = 0;
    for (int k = 0; k < RERANK_TOP; k++) {
      if (fraudLabels[neighborIds[k]] == 1) {
        fraudVoteCount++;
      }
    }
    return fraudVoteCount;
  }

  private static float adcDistance(
    final float[][] table,
    final byte[] codes,
    final int off
  ) {
    return table[0][codes[off] & 0xFF]
           + table[1][codes[off + 1] & 0xFF]
           + table[2][codes[off + 2] & 0xFF]
           + table[3][codes[off + 3] & 0xFF]
           + table[4][codes[off + 4] & 0xFF]
           + table[5][codes[off + 5] & 0xFF]
           + table[6][codes[off + 6] & 0xFF];
  }

  private void buildAdcLookupTable(
    final float[] queryVector,
    final float[][] lookupTable
  ) {
    for (int m = 0; m < PQ_M; m++) {
      final float q0 = queryVector[m * PQ_SUB_D];
      final float q1 = queryVector[m * PQ_SUB_D + 1];
      final float[] cb = pqCodebooksFlat[m];
      final float[] row = lookupTable[m];
      for (int c = 0; c < PQ_CODEBOOK_SIZE; c++) {
        final float d0 = q0 - cb[c * 2];
        final float d1 = q1 - cb[c * 2 + 1];
        row[c] = d0 * d0 + d1 * d1;
      }
    }
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
