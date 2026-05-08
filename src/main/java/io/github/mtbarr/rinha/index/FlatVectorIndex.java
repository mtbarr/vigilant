package io.github.mtbarr.rinha.index;

import jakarta.annotation.PostConstruct;
import jakarta.inject.Singleton;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel.MapMode;
import java.util.Arrays;
import jdk.incubator.vector.IntVector;
import jdk.incubator.vector.ShortVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

@Singleton
public class FlatVectorIndex {

  private static final int PADDED = 16;
  private static final int K = 5;
  private static final float SCALE = 8192f;

  private static final VectorSpecies<Short> S16 = ShortVector.SPECIES_256;
  private static final VectorSpecies<Integer> S32 = IntVector.SPECIES_256;

  private static final ValueLayout.OfInt INT_LE = ValueLayout.JAVA_INT.withOrder(ByteOrder.LITTLE_ENDIAN);

  private short[] vectors;
  private byte[] labels;
  private int count;
  private volatile boolean ready;

  private final ThreadLocal<short[]> queryBuf = ThreadLocal.withInitial(
    () -> new short[PADDED]
  );

  @PostConstruct
  void initialize() {
    final String path = System.getenv().getOrDefault("INDEX_PATH", "/app/data/flat_index.bin");
    try (final var raf = new RandomAccessFile(path, "r");
      final var ch = raf.getChannel()) {
      final MemorySegment seg = ch.map(MapMode.READ_ONLY, 0, raf.length(), Arena.global());
      if (seg.get(INT_LE, 0) != 0x464C4154) {
        throw new IOException("Bad magic");
      }
      if (seg.get(INT_LE, 4) != 1) {
        throw new IOException("Bad version");
      }
      count = seg.get(INT_LE, 8);
      if (seg.get(INT_LE, 12) != PADDED) {
        throw new IOException("Bad padded_d");
      }
      final long dataOff = 20L;
      final long dataLen = (long) count * PADDED * 2L;
      vectors = new short[count * PADDED];
      seg.asSlice(dataOff, dataLen).asByteBuffer().order(ByteOrder.LITTLE_ENDIAN).asShortBuffer().get(vectors);
      labels = new byte[count];
      seg.asSlice(dataOff + dataLen, count).asByteBuffer().get(labels);
      ready = true;
      System.out.println("Flat index loaded: " + count + " vectors");
    } catch (final Exception e) {
      System.err.println("Flat index load failed: " + e.getMessage());
    }
  }

  public int searchNearestNeighbors(
    final float[] queryFloat,
    final int[] outIds,
    final float[] outDists
  ) {
    final short[] q = queryBuf.get();
    for (int d = 0; d < 14; d++) {
      q[d] = (short) Math.clamp(Math.round(queryFloat[d] * SCALE), Short.MIN_VALUE, Short.MAX_VALUE);
    }
    for (int d = 14; d < PADDED; d++) {
      q[d] = 0;
    }

    Arrays.fill(outIds, -1);
    Arrays.fill(outDists, Float.MAX_VALUE);

    for (int i = 0; i < count; i++) {
      final int off = i * PADDED;
      final ShortVector vq = ShortVector.fromArray(S16, q, 0);
      final ShortVector vr = ShortVector.fromArray(S16, vectors, off);
      final ShortVector diff = vq.sub(vr);
      final IntVector lo = (IntVector) diff.convertShape(VectorOperators.S2I, S32, 0);
      final IntVector hi = (IntVector) diff.convertShape(VectorOperators.S2I, S32, 1);
      final float dist = lo.mul(lo).add(hi.mul(hi)).reduceLanes(VectorOperators.ADD) / (SCALE * SCALE);
      if (dist < outDists[K - 1]) {
        insert(outIds, outDists, K, i, dist);
      }
    }

    int votes = 0;
    for (int k = 0; k < K; k++) {
      if (outIds[k] >= 0 && labels[outIds[k]] == 1) {
        votes++;
      }
    }
    return votes;
  }

  private static void insert(final int[] ids, final float[] dists, final int max, final int newId, final float newDist) {
    int pos = max - 1;
    while (pos > 0 && dists[pos - 1] > newDist) {
      dists[pos] = dists[pos - 1];
      ids[pos] = ids[pos - 1];
      pos--;
    }
    dists[pos] = newDist;
    ids[pos] = newId;
  }

  public boolean isReady() {
    return ready;
  }
}
