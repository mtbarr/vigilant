package io.github.mtbarr.rinha.index;

import jakarta.annotation.PostConstruct;
import jakarta.enterprise.context.ApplicationScoped;
import java.io.File;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.channels.FileChannel.MapMode;

@ApplicationScoped
public class InvertedFileIndex {

  private static final int NUM_CLUSTERS = 512;
  private static final int NUM_DIMENSIONS = 14;
  private static final int PQ_M = 7;
  private static final int PQ_SUB_D = 2;
  private static final int PQ_CODEBOOK_SIZE = 256;
  private static final int NUM_PROBE_CLUSTERS = 24;
  private static final int NUM_PROBE_GRAY = 8;
  private static final int NUM_CANDIDATES = 50;
  private static final int NUM_NEIGHBORS = 5;

  private static final ValueLayout.OfInt INT_LE = ValueLayout.JAVA_INT.withOrder(ByteOrder.LITTLE_ENDIAN);
  private static final ValueLayout.OfLong LONG_LE = ValueLayout.JAVA_LONG.withOrder(ByteOrder.LITTLE_ENDIAN);
  private static final ValueLayout.OfFloat FLOAT_LE = ValueLayout.JAVA_FLOAT.withOrder(ByteOrder.LITTLE_ENDIAN);
  private static final ValueLayout.OfByte BYTE_LE = ValueLayout.JAVA_BYTE;

  private MemorySegment indexSegment;
  private float[] ivfCentroidsFlat;
  private float[][] pqCodebooksFlat;
  private byte[] fraudLabels;
  private long vectorsOffset;
  private long labelsOffset;
  private long invertedListsOffset;
  private int totalVectorCount;
  private volatile boolean isIndexReady = false;

  private final ThreadLocal<float[]> centroidDistanceBuffer = ThreadLocal.withInitial(
    () -> new float[NUM_CLUSTERS]
  );
  private final ThreadLocal<int[]> centroidOrderBuffer = ThreadLocal.withInitial(
    () -> new int[NUM_CLUSTERS]
  );
  private final ThreadLocal<float[][]> adcLookupTable = ThreadLocal.withInitial(
    () -> new float[PQ_M][PQ_CODEBOOK_SIZE]
  );
  private final ThreadLocal<int[]> coarseNeighborIds = ThreadLocal.withInitial(
    () -> new int[NUM_CANDIDATES]
  );
  private final ThreadLocal<float[]> coarseNeighborDistances = ThreadLocal.withInitial(
    () -> new float[NUM_CANDIDATES]
  );
  private final ThreadLocal<float[]> exactNeighborDistances = ThreadLocal.withInitial(
    () -> new float[NUM_CANDIDATES]
  );
  private final ThreadLocal<float[]> vectorReadBuffer = ThreadLocal.withInitial(
    () -> new float[NUM_DIMENSIONS]
  );

  @PostConstruct
  void initialize() {
    final String indexPath = System.getenv().getOrDefault(
      "INDEX_PATH",
      "/data/index.bin"
    );
    try {
      loadIndexFromFile(indexPath);
      runWarmupQueries();
      isIndexReady = true;
    } catch (final IOException exception) {
      System.err.println("Index not loaded: " + exception.getMessage());
      isIndexReady = false;
    }
  }

  private void loadIndexFromFile(final String filePath) throws IOException {
    final File indexFile = new File(filePath);
    try (final var randomAccessFile = new RandomAccessFile(indexFile, "r");
         final var fileChannel = randomAccessFile.getChannel()) {
      indexSegment = fileChannel.map(
        MapMode.READ_ONLY,
        0,
        indexFile.length(),
        Arena.global()
      );

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
      totalVectorCount = indexSegment.get(INT_LE, 12);
      vectorsOffset = indexSegment.get(LONG_LE, 16);
      labelsOffset = indexSegment.get(LONG_LE, 24);
      invertedListsOffset = 0;

      final long centroidsOffset = 36L;
      ivfCentroidsFlat = new float[NUM_CLUSTERS * NUM_DIMENSIONS];
      final MemorySegment centroidsSegment = indexSegment.asSlice(
        centroidsOffset,
        (long) NUM_CLUSTERS * NUM_DIMENSIONS * 4L
      );
      for (int i = 0; i < NUM_CLUSTERS * NUM_DIMENSIONS; i++) {
        ivfCentroidsFlat[i] = centroidsSegment.get(FLOAT_LE, (long) i * 4L);
      }

      final long codebooksOffset = centroidsOffset + (long) NUM_CLUSTERS * NUM_DIMENSIONS * 4L;
      pqCodebooksFlat = new float[PQ_M][PQ_CODEBOOK_SIZE * PQ_SUB_D];
      final MemorySegment codebooksSegment = indexSegment.asSlice(
        codebooksOffset,
        (long) PQ_M * PQ_CODEBOOK_SIZE * PQ_SUB_D * 4L
      );
      for (int m = 0; m < PQ_M; m++) {
        for (int c = 0; c < PQ_CODEBOOK_SIZE; c++) {
          for (int d = 0; d < PQ_SUB_D; d++) {
            pqCodebooksFlat[m][c * PQ_SUB_D + d] = codebooksSegment.get(
              FLOAT_LE,
              (long) (m * PQ_CODEBOOK_SIZE * PQ_SUB_D + c * PQ_SUB_D + d) * 4L
            );
          }
        }
      }

      final long invertedListsStart = labelsOffset + totalVectorCount;
      invertedListsOffset = invertedListsStart;

      fraudLabels = new byte[totalVectorCount];
      final MemorySegment labelsSegment = indexSegment.asSlice(
        labelsOffset,
        totalVectorCount
      );
      labelsSegment.asByteBuffer().get(fraudLabels);

      System.out.println("Index loaded: " + totalVectorCount + " vectors, "
                         + "vectorsOffset=" + vectorsOffset
                         + ", labelsOffset=" + labelsOffset
                         + ", invertedListsOffset=" + invertedListsStart);
      int fraudCount = 0;
      for (byte label : fraudLabels) {
        if (label == 1) fraudCount++;
      }
      System.out.println("Fraud labels: " + fraudCount + " / " + totalVectorCount);
    }
  }

  private void runWarmupQueries() {
    final float[] warmupQuery = new float[NUM_DIMENSIONS];
    final int[] warmupNeighbors = new int[NUM_NEIGHBORS];
    final float[] warmupDistances = new float[NUM_NEIGHBORS];
    for (int i = 0; i < 200; i++) {
      for (int d = 0; d < NUM_DIMENSIONS; d++) {
        warmupQuery[d] = (i * 7 + d) % 100 / 100f;
      }
      searchNearestNeighbors(warmupQuery, warmupNeighbors, warmupDistances);
    }
  }

  public int searchNearestNeighbors(
    final float[] queryVector,
    final int[] neighborIds,
    final float[] neighborDistances
  ) {
    final float[] centroidDistances = centroidDistanceBuffer.get();
    final int[] centroidOrder = centroidOrderBuffer.get();

    for (int clusterIndex = 0; clusterIndex < NUM_CLUSTERS; clusterIndex++) {
      float clusterDistance = 0;
      final int centroidBase = clusterIndex * NUM_DIMENSIONS;
      for (int dimIndex = 0; dimIndex < NUM_DIMENSIONS; dimIndex++) {
        final float delta = queryVector[dimIndex] - ivfCentroidsFlat[centroidBase + dimIndex];
        clusterDistance += delta * delta;
      }
      centroidDistances[clusterIndex] = clusterDistance;
      centroidOrder[clusterIndex] = clusterIndex;
    }
    partialSortCentroids(centroidOrder, centroidDistances, NUM_PROBE_CLUSTERS);

    final float[][] adcTable = adcLookupTable.get();
    buildAdcLookupTable(queryVector, adcTable);

    final int[] coarseIds = coarseNeighborIds.get();
    final float[] coarseDists = coarseNeighborDistances.get();
    java.util.Arrays.fill(coarseDists, Float.MAX_VALUE);
    java.util.Arrays.fill(coarseIds, -1);

    for (int probeIndex = 0; probeIndex < NUM_PROBE_CLUSTERS; probeIndex++) {
      final int clusterIndex = centroidOrder[probeIndex];
      final long clusterListOffset = readClusterListOffset(clusterIndex);
      final int clusterSize = readClusterSize(clusterListOffset);
      final int[] clusterIds = readClusterIds(clusterListOffset, clusterSize);
      final byte[] clusterPqCodes = readClusterPqCodes(clusterListOffset, clusterSize);

      for (int itemIndex = 0; itemIndex < clusterSize; itemIndex++) {
        final float approxDistance = computeAdcDistance(adcTable, clusterPqCodes, itemIndex * PQ_M);
        if (approxDistance < coarseDists[NUM_CANDIDATES - 1]) {
          insertIntoSortedArray(
            coarseIds,
            coarseDists,
            NUM_CANDIDATES,
            clusterIds[itemIndex],
            approxDistance
          );
        }
      }

      if (probeIndex == NUM_PROBE_GRAY - 1) {
        int fraudCount = 0;
        for (int i = 0; i < NUM_NEIGHBORS; i++) {
          if (coarseIds[i] >= 0 && fraudLabels[coarseIds[i]] == 1) {
            fraudCount++;
          }
        }
        if (fraudCount <= 1 || fraudCount >= NUM_NEIGHBORS - 1) {
          java.util.Arrays.fill(neighborIds, -1);
          java.util.Arrays.fill(neighborDistances, Float.MAX_VALUE);
          for (int i = 0; i < NUM_CANDIDATES; i++) {
            if (coarseIds[i] >= 0 && coarseDists[i] < neighborDistances[NUM_NEIGHBORS - 1]) {
              insertIntoSortedArray(neighborIds, neighborDistances, NUM_NEIGHBORS, coarseIds[i], coarseDists[i]);
            }
          }
          int fc = 0;
          for (int i = 0; i < NUM_NEIGHBORS; i++) {
            if (neighborIds[i] >= 0 && fraudLabels[neighborIds[i]] == 1) fc++;
          }
          return fc;
        }
      }
    }

    final int[] exactIds = coarseNeighborIds.get();
    final float[] exactDists = exactNeighborDistances.get();
    final float[] vec = vectorReadBuffer.get();
    System.arraycopy(coarseIds, 0, exactIds, 0, NUM_CANDIDATES);

    final long vectorBytesStart = (long) totalVectorCount * NUM_DIMENSIONS * 4L + vectorsOffset;

    for (int i = 0; i < NUM_CANDIDATES; i++) {
      final int id = coarseIds[i];
      if (id < 0) {
        exactDists[i] = Float.MAX_VALUE;
        continue;
      }
      final long vectorPosition = vectorsOffset + (long) id * NUM_DIMENSIONS * 4L;
      for (int d = 0; d < NUM_DIMENSIONS; d++) {
        vec[d] = indexSegment.get(FLOAT_LE, vectorPosition + (long) d * 4L);
      }
      exactDists[i] = computeSquaredDistance(queryVector, vec);
    }

    java.util.Arrays.fill(neighborIds, -1);
    java.util.Arrays.fill(neighborDistances, Float.MAX_VALUE);
    for (int i = 0; i < NUM_CANDIDATES; i++) {
      if (exactIds[i] >= 0 && exactDists[i] < neighborDistances[NUM_NEIGHBORS - 1]) {
        insertIntoSortedArray(
          neighborIds,
          neighborDistances,
          NUM_NEIGHBORS,
          exactIds[i],
          exactDists[i]
        );
      }
    }

    int fraudVoteCount = 0;
    for (int i = 0; i < NUM_NEIGHBORS; i++) {
      if (neighborIds[i] >= 0 && fraudLabels[neighborIds[i]] == 1) {
        fraudVoteCount++;
      }
    }
    return fraudVoteCount;
  }

  private long readClusterListOffset(final int clusterIndex) {
    long offset = invertedListsOffset;
    for (int c = 0; c < clusterIndex; c++) {
      final int size = indexSegment.get(INT_LE, offset);
      offset += 4L + (long) size * 4L + (long) size * PQ_M;
    }
    return offset;
  }

  private int readClusterSize(final long clusterListOffset) {
    return indexSegment.get(INT_LE, clusterListOffset);
  }

  private int[] readClusterIds(final long clusterListOffset, final int clusterSize) {
    final int[] ids = new int[clusterSize];
    final MemorySegment idsSegment = indexSegment.asSlice(
      clusterListOffset + 4L,
      (long) clusterSize * 4L
    );
    for (int i = 0; i < clusterSize; i++) {
      ids[i] = idsSegment.get(INT_LE, (long) i * 4L);
    }
    return ids;
  }

  private byte[] readClusterPqCodes(final long clusterListOffset, final int clusterSize) {
    final byte[] codes = new byte[clusterSize * PQ_M];
    final MemorySegment codesSegment = indexSegment.asSlice(
      clusterListOffset + 4L + (long) clusterSize * 4L,
      (long) clusterSize * PQ_M
    );
    codesSegment.asByteBuffer().get(codes);
    return codes;
  }

  private void buildAdcLookupTable(final float[] queryVector, final float[][] lookupTable) {
    for (int subspaceIndex = 0; subspaceIndex < PQ_M; subspaceIndex++) {
      final float queryComponent0 = queryVector[subspaceIndex * PQ_SUB_D];
      final float queryComponent1 = queryVector[subspaceIndex * PQ_SUB_D + 1];
      final float[] codebookFlat = pqCodebooksFlat[subspaceIndex];
      final float[] tableRow = lookupTable[subspaceIndex];
      for (int codebookIndex = 0; codebookIndex < PQ_CODEBOOK_SIZE; codebookIndex++) {
        final float delta0 = queryComponent0 - codebookFlat[codebookIndex * 2];
        final float delta1 = queryComponent1 - codebookFlat[codebookIndex * 2 + 1];
        tableRow[codebookIndex] = delta0 * delta0 + delta1 * delta1;
      }
    }
  }

  private float computeAdcDistance(final float[][] lookupTable, final byte[] pqCodes, final int codeOffset) {
    return lookupTable[0][pqCodes[codeOffset] & 0xFF]
      + lookupTable[1][pqCodes[codeOffset + 1] & 0xFF]
      + lookupTable[2][pqCodes[codeOffset + 2] & 0xFF]
      + lookupTable[3][pqCodes[codeOffset + 3] & 0xFF]
      + lookupTable[4][pqCodes[codeOffset + 4] & 0xFF]
      + lookupTable[5][pqCodes[codeOffset + 5] & 0xFF]
      + lookupTable[6][pqCodes[codeOffset + 6] & 0xFF];
  }

  private float computeSquaredDistance(final float[] vectorA, final float[] vectorB) {
    float sum = 0;
    for (int dimIndex = 0; dimIndex < NUM_DIMENSIONS; dimIndex++) {
      final float delta = vectorA[dimIndex] - vectorB[dimIndex];
      sum += delta * delta;
    }
    return sum;
  }

  private static void insertIntoSortedArray(
    final int[] neighborIds,
    final float[] neighborDistances,
    final int maxNeighbors,
    final int newId,
    final float newDistance
  ) {
    int insertPosition = maxNeighbors - 1;
    while (insertPosition > 0 && neighborDistances[insertPosition - 1] > newDistance) {
      neighborDistances[insertPosition] = neighborDistances[insertPosition - 1];
      neighborIds[insertPosition] = neighborIds[insertPosition - 1];
      insertPosition--;
    }
    neighborDistances[insertPosition] = newDistance;
    neighborIds[insertPosition] = newId;
  }

  private static void partialSortCentroids(
    final int[] centroidOrder,
    final float[] centroidDistances,
    final int topCount
  ) {
    quickselectCentroids(centroidOrder, centroidDistances, 0, centroidOrder.length - 1, topCount);
    for (int i = 1; i < topCount; i++) {
      final int currentOrder = centroidOrder[i];
      final float currentDistance = centroidDistances[currentOrder];
      int j = i - 1;
      while (j >= 0 && centroidDistances[centroidOrder[j]] > currentDistance) {
        centroidOrder[j + 1] = centroidOrder[j];
        j--;
      }
      centroidOrder[j + 1] = currentOrder;
    }
  }

  private static void quickselectCentroids(
    final int[] centroidOrder,
    final float[] centroidDistances,
    final int leftBoundary,
    final int rightBoundary,
    final int targetRank
  ) {
    if (leftBoundary >= rightBoundary) {
      return;
    }
    final int pivotIndex = partitionCentroids(centroidOrder, centroidDistances, leftBoundary, rightBoundary);
    final int pivotRank = pivotIndex - leftBoundary + 1;
    if (pivotRank == targetRank) {
      return;
    }
    if (targetRank < pivotRank) {
      quickselectCentroids(centroidOrder, centroidDistances, leftBoundary, pivotIndex - 1, targetRank);
    } else {
      quickselectCentroids(centroidOrder, centroidDistances, pivotIndex + 1, rightBoundary, targetRank - pivotRank);
    }
  }

  private static int partitionCentroids(
    final int[] centroidOrder,
    final float[] centroidDistances,
    final int leftBoundary,
    final int rightBoundary
  ) {
    final float pivotDistance = centroidDistances[centroidOrder[rightBoundary]];
    int swapIndex = leftBoundary - 1;
    for (int scanIndex = leftBoundary; scanIndex < rightBoundary; scanIndex++) {
      if (centroidDistances[centroidOrder[scanIndex]] <= pivotDistance) {
        swapIndex++;
        final int tempOrder = centroidOrder[swapIndex];
        centroidOrder[swapIndex] = centroidOrder[scanIndex];
        centroidOrder[scanIndex] = tempOrder;
      }
    }
    final int tempOrder = centroidOrder[swapIndex + 1];
    centroidOrder[swapIndex + 1] = centroidOrder[rightBoundary];
    centroidOrder[rightBoundary] = tempOrder;
    return swapIndex + 1;
  }

  public boolean isReady() {
    return isIndexReady;
  }
}