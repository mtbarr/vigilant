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
  private static final int PQ_M = 14;
  private static final int PQ_SUB_D = 1;
  private static final int PQ_CODEBOOK_SIZE = 256;
  private static final int NUM_PROBE_CLUSTERS = 12;
  private static final int NUM_PROBE_GRAY = 8;
  private static final int NUM_NEIGHBORS = 10;
  private static final int RERANK_TOP = 5;

  private static final ValueLayout.OfInt INT_LE = ValueLayout.JAVA_INT.withOrder(ByteOrder.LITTLE_ENDIAN);
  private static final ValueLayout.OfLong LONG_LE = ValueLayout.JAVA_LONG.withOrder(ByteOrder.LITTLE_ENDIAN);
  private static final ValueLayout.OfFloat FLOAT_LE = ValueLayout.JAVA_FLOAT.withOrder(ByteOrder.LITTLE_ENDIAN);

  private float[] ivfCentroidsFlat;
  private float[] pqCodebooksFlat;
  private byte[] fraudLabels;
  private int[] flatIds;
  private byte[] flatCodes;
  private int[] clusterOffsets;
  private volatile boolean isIndexReady = false;
  private float[] exactVectors;

  private final ThreadLocal<float[]> centroidDistanceBuffer = ThreadLocal.withInitial(
    () -> new float[NUM_CLUSTERS]
  );
  private final ThreadLocal<int[]> centroidOrderBuffer = ThreadLocal.withInitial(
    () -> new int[NUM_CLUSTERS]
  );
  private final ThreadLocal<float[]> adcLookupFlat = ThreadLocal.withInitial(
    () -> new float[PQ_M * PQ_CODEBOOK_SIZE]
  );

  @PostConstruct
  void initialize() {
    final String indexPath = System.getenv().getOrDefault("INDEX_PATH", "/data/index.bin");
    try {
      loadIndexFromFile(indexPath);
      warmup();
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
      long vectorsOffset = indexSegment.get(LONG_LE, 16);
      final long labelsOffset = indexSegment.get(LONG_LE, 24);

      final long centroidsOffset = 36L;
      ivfCentroidsFlat = new float[NUM_CLUSTERS * NUM_DIMENSIONS];
      final MemorySegment centroidsSegment = indexSegment.asSlice(
        centroidsOffset, (long) NUM_CLUSTERS * NUM_DIMENSIONS * 4L);
      for (int i = 0; i < NUM_CLUSTERS * NUM_DIMENSIONS; i++) {
        ivfCentroidsFlat[i] = centroidsSegment.get(FLOAT_LE, (long) i * 4L);
      }

      final long codebooksOffset = centroidsOffset + (long) NUM_CLUSTERS * NUM_DIMENSIONS * 4L;
      final long cbLen = (long) PQ_M * PQ_CODEBOOK_SIZE * PQ_SUB_D * 4L;
      pqCodebooksFlat = new float[PQ_M * PQ_CODEBOOK_SIZE * PQ_SUB_D];
      final MemorySegment codebooksSegment = indexSegment.asSlice(codebooksOffset, cbLen);
      for (int i = 0; i < pqCodebooksFlat.length; i++) {
        pqCodebooksFlat[i] = codebooksSegment.get(FLOAT_LE, (long) i * 4L);
      }

      fraudLabels = new byte[totalVectorCount];
      final MemorySegment labelsSegment = indexSegment.asSlice(labelsOffset, totalVectorCount);
      labelsSegment.asByteBuffer().get(fraudLabels);

      final long vecLen = (long) totalVectorCount * NUM_DIMENSIONS * 4L;
      exactVectors = new float[totalVectorCount * NUM_DIMENSIONS];
      indexSegment.asSlice(vectorsOffset, vecLen)
        .asByteBuffer().order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(exactVectors);

      // --- Copy inverted lists to flat arrays ---
      final long invertedListsOffset = labelsOffset + totalVectorCount;
      final int[] clusterSizes = new int[NUM_CLUSTERS];
      final long[] listPositions = new long[NUM_CLUSTERS];
      clusterOffsets = new int[NUM_CLUSTERS + 1];
      int totalSize = 0;
      long pos = invertedListsOffset;
      for (int c = 0; c < NUM_CLUSTERS; c++) {
        final int size = indexSegment.get(INT_LE, pos);
        clusterSizes[c] = size;
        clusterOffsets[c] = totalSize;
        totalSize += size;
        listPositions[c] = pos;
        final long dataSize = 4L + (long) size * 4L + (long) size * PQ_M;
        pos += (dataSize + 3L) & ~3L;
      }
      clusterOffsets[NUM_CLUSTERS] = totalSize;

      flatIds = new int[totalSize];
      flatCodes = new byte[totalSize * PQ_M];
      for (int c = 0; c < NUM_CLUSTERS; c++) {
        final int size = clusterSizes[c];
        final long idsOffset = listPositions[c] + 4L;
        final long codesOffset = idsOffset + (long) size * 4L;
        final int destOff = clusterOffsets[c];
        indexSegment.asSlice(idsOffset, (long) size * 4L)
          .asByteBuffer()
          .order(ByteOrder.LITTLE_ENDIAN)
          .asIntBuffer()
          .get(flatIds, destOff, size);
        indexSegment.asSlice(codesOffset, (long) size * PQ_M)
          .asByteBuffer()
          .get(flatCodes, destOff * PQ_M, size * PQ_M);
      }

      int fraudCount = 0;
      for (byte label : fraudLabels) {
        if (label == 1) {
          fraudCount++;
        }
      }
      System.out.println("Index loaded: " + totalVectorCount + " vectors, " + fraudCount + " fraud");
    }catch (Exception e){
      throw new IOException("Failed on loading index file" , e);
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
      final float d0 = queryVector[0] - ivfCentroidsFlat[base];
      float dist = d0 * d0;

      // Isso aqui vai ser mais rápido do que iterar....
      dist = Math.fma(queryVector[1] - ivfCentroidsFlat[base + 1], queryVector[1] - ivfCentroidsFlat[base + 1], dist);
      dist = Math.fma(queryVector[2] - ivfCentroidsFlat[base + 2], queryVector[2] - ivfCentroidsFlat[base + 2], dist);
      dist = Math.fma(queryVector[3] - ivfCentroidsFlat[base + 3], queryVector[3] - ivfCentroidsFlat[base + 3], dist);
      dist = Math.fma(queryVector[4] - ivfCentroidsFlat[base + 4], queryVector[4] - ivfCentroidsFlat[base + 4], dist);
      dist = Math.fma(queryVector[5] - ivfCentroidsFlat[base + 5], queryVector[5] - ivfCentroidsFlat[base + 5], dist);
      dist = Math.fma(queryVector[6] - ivfCentroidsFlat[base + 6], queryVector[6] - ivfCentroidsFlat[base + 6], dist);
      dist = Math.fma(queryVector[7] - ivfCentroidsFlat[base + 7], queryVector[7] - ivfCentroidsFlat[base + 7], dist);
      dist = Math.fma(queryVector[8] - ivfCentroidsFlat[base + 8], queryVector[8] - ivfCentroidsFlat[base + 8], dist);
      dist = Math.fma(queryVector[9] - ivfCentroidsFlat[base + 9], queryVector[9] - ivfCentroidsFlat[base + 9], dist);
      dist = Math.fma(
        queryVector[10] - ivfCentroidsFlat[base + 10], queryVector[10] - ivfCentroidsFlat[base + 10], dist);
      dist = Math.fma(
        queryVector[11] - ivfCentroidsFlat[base + 11], queryVector[11] - ivfCentroidsFlat[base + 11], dist);
      dist = Math.fma(
        queryVector[12] - ivfCentroidsFlat[base + 12], queryVector[12] - ivfCentroidsFlat[base + 12], dist);
      dist = Math.fma(
        queryVector[13] - ivfCentroidsFlat[base + 13], queryVector[13] - ivfCentroidsFlat[base + 13], dist);
      centroidDistances[ci] = dist;
      centroidOrder[ci] = ci;
    }
    final int[] probes = selectNearestCentroids(centroidDistances, centroidOrder);

    final float[] adcTable = adcLookupFlat.get();
    buildAdcLookupTable(queryVector, adcTable);

    Arrays.fill(neighborIds, -1);
    Arrays.fill(neighborDistances, Float.MAX_VALUE);

    for (int probe = 0; probe < NUM_PROBE_CLUSTERS; probe++) {
      final int ci = probes[probe];
      final int startIdx = clusterOffsets[ci];
      final int endIdx = clusterOffsets[ci + 1];

      int codeOff = startIdx * PQ_M;
      for (int i = startIdx; i < endIdx; i++, codeOff += PQ_M) {
        final float approx = adcTable[flatCodes[codeOff] & 0xFF]
                             + adcTable[256 + (flatCodes[codeOff + 1] & 0xFF)]
                             + adcTable[512 + (flatCodes[codeOff + 2] & 0xFF)]
                             + adcTable[768 + (flatCodes[codeOff + 3] & 0xFF)]
                             + adcTable[1024 + (flatCodes[codeOff + 4] & 0xFF)]
                             + adcTable[1280 + (flatCodes[codeOff + 5] & 0xFF)]
                             + adcTable[1536 + (flatCodes[codeOff + 6] & 0xFF)]
                             + adcTable[1792 + (flatCodes[codeOff + 7] & 0xFF)]
                             + adcTable[2048 + (flatCodes[codeOff + 8] & 0xFF)]
                             + adcTable[2304 + (flatCodes[codeOff + 9] & 0xFF)]
                             + adcTable[2560 + (flatCodes[codeOff + 10] & 0xFF)]
                             + adcTable[2816 + (flatCodes[codeOff + 11] & 0xFF)]
                             + adcTable[3072 + (flatCodes[codeOff + 12] & 0xFF)]
                             + adcTable[3328 + (flatCodes[codeOff + 13] & 0xFF)];
        if (approx < neighborDistances[NUM_NEIGHBORS - 1]) {
          insertIntoSortedArray(neighborIds, neighborDistances, flatIds[i], approx);
        }
      }
    }

    for (int k = 0; k < NUM_NEIGHBORS; k++) {
      if (neighborIds[k] >= 0) {
        final int baseOff = neighborIds[k] * NUM_DIMENSIONS;
        float exactDist = 0f;
        for (int d = 0; d < NUM_DIMENSIONS; d++) {
          final float delta = queryVector[d] - exactVectors[baseOff + d];
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

  private void buildAdcLookupTable(
    final float[] queryVector,
    final float[] lookupFlat
  ) {
    final float[] cb = this.pqCodebooksFlat;

    for (int m = 0; m < PQ_M; m++) {
      final float q = queryVector[m];
      final int offset = m << 8; // m * 256
      for (int c = 0; c < PQ_CODEBOOK_SIZE; c++) {
        int idx = offset + c;
        float diff = q - cb[idx];
        lookupFlat[idx] = diff * diff;
      }
    }
  }

  public boolean isReady() {
    return isIndexReady;
  }

  private void warmup() {
    final float[] q = new float[NUM_DIMENSIONS];
    final int[] ids = new int[NUM_NEIGHBORS];
    final float[] dists = new float[NUM_NEIGHBORS];
    for (int i = 0; i < 500; i++) {
      searchNearestNeighbors(q, ids, dists);
    }
  }

  private static void insertIntoSortedArray(
    final int[] ids,
    final float[] dists,
    final int newId,
    final float newDist
  ) {
    int pos = InvertedFileIndex.NUM_NEIGHBORS - 1;
    while (pos > 0 && dists[pos - 1] > newDist) {
      dists[pos] = dists[pos - 1];
      ids[pos] = ids[pos - 1];
      pos--;
    }
    dists[pos] = newDist;
    ids[pos] = newId;
  }

  private static int[] selectNearestCentroids(final float[] dist, final int[] scratch) {
    final int K = InvertedFileIndex.NUM_PROBE_CLUSTERS;
    final int[] indices = new int[K];
    final float[] top = new float[K];
    System.arraycopy(scratch, 0, indices, 0, K);
    System.arraycopy(dist, 0, top, 0, K);
    for (int i = K; i < dist.length; i++) {
      int worst = 0;
      float worstVal = top[0];
      for (int j = 1; j < K; j++) {
        if (top[j] > worstVal) {
          worst = j;
          worstVal = top[j];
        }
      }
      if (dist[i] < worstVal) {
        top[worst] = dist[i];
        indices[worst] = i;
      }
    }
    return indices;
  }
}
