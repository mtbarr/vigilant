package io.github.mtbarr.rinha.index;

import jakarta.annotation.PostConstruct;
import jakarta.enterprise.context.ApplicationScoped;
import java.io.File;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.Arrays;

@ApplicationScoped
public class InvertedFileIndex {

  private static final int NUM_CLUSTERS = 512;
  private static final int NUM_DIMENSIONS = 14;
  private static final int PQ_NUM_SUBSPACES = 7;
  private static final int PQ_SUBSPACE_DIM = 2;
  private static final int PQ_CODEBOOK_SIZE = 256;
  private static final int NUM_PROBE_CLUSTERS = 64;
  private static final int NUM_CANDIDATES = 50;
  private static final int NUM_NEIGHBORS = 5;

  private float[] ivfCentroidsFlat;
  private int[][] vectorIdsByCluster;
  private byte[][] pqCodesByCluster;
  private byte[] fraudLabels;
  private float[][][] pqCodebooks;
  private float[][] pqCodebooksFlat;
  private MappedByteBuffer vectorsMappedBuffer;
  private long vectorsSectionOffset;
  private int totalVectorCount;
  private volatile boolean isIndexReady = false;

  private final ThreadLocal<float[]> centroidDistanceBuffer = ThreadLocal.withInitial(
    () -> new float[NUM_CLUSTERS]
  );
  private final ThreadLocal<int[]> centroidOrderBuffer = ThreadLocal.withInitial(
    () -> new int[NUM_CLUSTERS]
  );
  private final ThreadLocal<float[][]> adcLookupTable = ThreadLocal.withInitial(
    () -> new float[PQ_NUM_SUBSPACES][PQ_CODEBOOK_SIZE]
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
      final ByteBuffer buffer = fileChannel.map(
        FileChannel.MapMode.READ_ONLY,
        0,
        indexFile.length()
      );
      buffer.order(ByteOrder.LITTLE_ENDIAN);

      final int magicNumber = buffer.getInt();
      if (magicNumber != 0x52494E44) {
        throw new IOException("Bad magic: " + Integer.toHexString(magicNumber));
      }
      final int versionNumber = buffer.getInt();
      if (versionNumber != 1) {
        throw new IOException("Bad version: " + versionNumber);
      }
      final int storedClusters = buffer.getInt();
      if (storedClusters != NUM_CLUSTERS) {
        throw new IOException(
          "Expected K=" + NUM_CLUSTERS + " got " + storedClusters
        );
      }
      totalVectorCount = buffer.getInt();
      vectorsSectionOffset = buffer.getLong();

      ivfCentroidsFlat = new float[NUM_CLUSTERS * NUM_DIMENSIONS];
      for (int i = 0; i < NUM_CLUSTERS * NUM_DIMENSIONS; i++) {
        ivfCentroidsFlat[i] = buffer.getFloat();
      }

      pqCodebooks = new float[PQ_NUM_SUBSPACES][PQ_CODEBOOK_SIZE][PQ_SUBSPACE_DIM];
      pqCodebooksFlat = new float[PQ_NUM_SUBSPACES][PQ_CODEBOOK_SIZE * PQ_SUBSPACE_DIM];
      for (int subspaceIndex = 0; subspaceIndex < PQ_NUM_SUBSPACES; subspaceIndex++) {
        for (int codebookIndex = 0; codebookIndex < PQ_CODEBOOK_SIZE; codebookIndex++) {
          for (int dimIndex = 0; dimIndex < PQ_SUBSPACE_DIM; dimIndex++) {
            final float value = buffer.getFloat();
            pqCodebooks[subspaceIndex][codebookIndex][dimIndex] = value;
            pqCodebooksFlat[subspaceIndex][codebookIndex * PQ_SUBSPACE_DIM + dimIndex] = value;
          }
        }
      }

      final int storedVectorCount = buffer.getInt();
      if (storedVectorCount != totalVectorCount) {
        throw new IOException("N mismatch");
      }

      vectorsMappedBuffer = fileChannel.map(
        FileChannel.MapMode.READ_ONLY,
        vectorsSectionOffset,
        (long) totalVectorCount * NUM_DIMENSIONS * 4
      );
      vectorsMappedBuffer.order(ByteOrder.LITTLE_ENDIAN);

      buffer.position(
        (int) (vectorsSectionOffset + (long) totalVectorCount * NUM_DIMENSIONS * 4)
      );

      fraudLabels = new byte[totalVectorCount];
      buffer.get(fraudLabels);

      vectorIdsByCluster = new int[NUM_CLUSTERS][];
      pqCodesByCluster = new byte[NUM_CLUSTERS][];
      for (int clusterIndex = 0; clusterIndex < NUM_CLUSTERS; clusterIndex++) {
        final int clusterSize = buffer.getInt();
        vectorIdsByCluster[clusterIndex] = new int[clusterSize];
        pqCodesByCluster[clusterIndex] = new byte[clusterSize * PQ_NUM_SUBSPACES];
        for (int i = 0; i < clusterSize; i++) {
          vectorIdsByCluster[clusterIndex][i] = buffer.getInt();
        }
        buffer.get(pqCodesByCluster[clusterIndex]);
      }
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
    Arrays.fill(coarseDists, Float.MAX_VALUE);
    Arrays.fill(coarseIds, -1);

    for (int probeIndex = 0; probeIndex < NUM_PROBE_CLUSTERS; probeIndex++) {
      final int clusterIndex = centroidOrder[probeIndex];
      final int[] clusterIds = vectorIdsByCluster[clusterIndex];
      final byte[] clusterCodes = pqCodesByCluster[clusterIndex];
      final int clusterSize = clusterIds.length;

      for (int itemIndex = 0; itemIndex < clusterSize; itemIndex++) {
        final float approxDistance = computeAdcDistance(
          adcTable,
          clusterCodes,
          itemIndex * PQ_NUM_SUBSPACES
        );
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
    }

    final int[] exactIds = coarseNeighborIds.get();
    final float[] exactDists = exactNeighborDistances.get();
    final float[] vectorBuffer = vectorReadBuffer.get();
    System.arraycopy(coarseIds, 0, exactIds, 0, NUM_CANDIDATES);

    for (int candidateIndex = 0; candidateIndex < NUM_CANDIDATES; candidateIndex++) {
      final int vectorId = coarseIds[candidateIndex];
      if (vectorId < 0) {
        exactDists[candidateIndex] = Float.MAX_VALUE;
        continue;
      }
      readVectorById(vectorId, vectorBuffer);
      exactDists[candidateIndex] = computeSquaredDistance(queryVector, vectorBuffer);
    }

    Arrays.fill(neighborIds, -1);
    Arrays.fill(neighborDistances, Float.MAX_VALUE);
    for (int candidateIndex = 0; candidateIndex < NUM_CANDIDATES; candidateIndex++) {
      if (exactIds[candidateIndex] >= 0
          && exactDists[candidateIndex] < neighborDistances[NUM_NEIGHBORS - 1]) {
        insertIntoSortedArray(
          neighborIds,
          neighborDistances,
          NUM_NEIGHBORS,
          exactIds[candidateIndex],
          exactDists[candidateIndex]
        );
      }
    }

    int fraudVoteCount = 0;
    for (int neighborIndex = 0; neighborIndex < NUM_NEIGHBORS; neighborIndex++) {
      if (neighborIds[neighborIndex] >= 0
          && fraudLabels[neighborIds[neighborIndex]] == 1) {
        fraudVoteCount++;
      }
    }
    return fraudVoteCount;
  }

  private void buildAdcLookupTable(
    final float[] queryVector,
    final float[][] lookupTable
  ) {
    for (int subspaceIndex = 0; subspaceIndex < PQ_NUM_SUBSPACES; subspaceIndex++) {
      final float queryComponent0 = queryVector[subspaceIndex * PQ_SUBSPACE_DIM];
      final float queryComponent1 = queryVector[subspaceIndex * PQ_SUBSPACE_DIM + 1];
      final float[] codebookFlat = pqCodebooksFlat[subspaceIndex];
      final float[] tableRow = lookupTable[subspaceIndex];
      for (int codebookIndex = 0; codebookIndex < PQ_CODEBOOK_SIZE; codebookIndex++) {
        final float delta0 = queryComponent0 - codebookFlat[codebookIndex * 2];
        final float delta1 = queryComponent1 - codebookFlat[codebookIndex * 2 + 1];
        tableRow[codebookIndex] = delta0 * delta0 + delta1 * delta1;
      }
    }
  }

  private float computeAdcDistance(
    final float[][] lookupTable,
    final byte[] pqCodes,
    final int codeOffset
  ) {
    return lookupTable[0][pqCodes[codeOffset] & 0xFF]
      + lookupTable[1][pqCodes[codeOffset + 1] & 0xFF]
      + lookupTable[2][pqCodes[codeOffset + 2] & 0xFF]
      + lookupTable[3][pqCodes[codeOffset + 3] & 0xFF]
      + lookupTable[4][pqCodes[codeOffset + 4] & 0xFF]
      + lookupTable[5][pqCodes[codeOffset + 5] & 0xFF]
      + lookupTable[6][pqCodes[codeOffset + 6] & 0xFF];
  }

  private void readVectorById(final int vectorId, final float[] outputBuffer) {
    final int bytePosition = (int) ((long) vectorId * NUM_DIMENSIONS * 4);
    vectorsMappedBuffer.position(bytePosition);
    for (int dimIndex = 0; dimIndex < NUM_DIMENSIONS; dimIndex++) {
      outputBuffer[dimIndex] = vectorsMappedBuffer.getFloat();
    }
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
    quickselectCentroids(
      centroidOrder,
      centroidDistances,
      0,
      centroidOrder.length - 1,
      topCount
    );
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
    final int pivotIndex = partitionCentroids(
      centroidOrder,
      centroidDistances,
      leftBoundary,
      rightBoundary
    );
    final int pivotRank = pivotIndex - leftBoundary + 1;
    if (pivotRank == targetRank) {
      return;
    }
    if (targetRank < pivotRank) {
      quickselectCentroids(
        centroidOrder,
        centroidDistances,
        leftBoundary,
        pivotIndex - 1,
        targetRank
      );
    } else {
      quickselectCentroids(
        centroidOrder,
        centroidDistances,
        pivotIndex + 1,
        rightBoundary,
        targetRank - pivotRank
      );
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
