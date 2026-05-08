package io.github.mtbarr.rinha.index;

import jakarta.annotation.PostConstruct;
import jakarta.inject.Singleton;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
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

  private short[] vectors;
  private byte[] labels;
  private int count;
  private volatile boolean ready;

  private final ThreadLocal<short[]> queryBuf = ThreadLocal.withInitial(
    () -> new short[PADDED]
  );

  @PostConstruct
  void initialize() {
    try {
      load();
      ready = true;
      System.out.println("Flat index loaded: " + count + " vectors");
    } catch (final Exception e) {
      System.err.println("Flat index load failed: " + e.getMessage());
    }
  }

  private void load() throws IOException {
    try (final InputStream is = getClass().getResourceAsStream("/flat_index.bin")) {
      if (is == null) throw new IOException("Resource /flat_index.bin not found");
      final byte[] data = is.readAllBytes();
      final ByteBuffer buf = ByteBuffer.wrap(data).order(ByteOrder.LITTLE_ENDIAN);
      if (buf.getInt() != 0x464C4154) throw new IOException("Bad magic");
      if (buf.getInt() != 1) throw new IOException("Bad version");
      count = buf.getInt();
      if (buf.getInt() != PADDED) throw new IOException("Bad padded_d");
      buf.getInt(); // scale
      final int total = count * PADDED;
      vectors = new short[total];
      buf.asShortBuffer().get(vectors);
      buf.position(buf.position() + total * 2);
      labels = new byte[count];
      buf.get(labels);
    }
  }

  public int searchNearestNeighbors(
    final float[] queryFloat,
    final int[] outIds,
    final float[] outDists
  ) {
    final short[] q = queryBuf.get();
    for (int d = 0; d < PADDED; d++) {
      if (d < 14) {
        final int v = Math.round(queryFloat[d] * SCALE);
        q[d] = (short) Math.max(Short.MIN_VALUE, Math.min(Short.MAX_VALUE, v));
      } else {
        q[d] = 0;
      }
    }

    Arrays.fill(outIds, -1);
    Arrays.fill(outDists, Float.MAX_VALUE);

    for (int i = 0; i < count; i++) {
      final float dist = simdDist(q, i);
      if (dist < outDists[K - 1]) {
        insert(outIds, outDists, K, i, dist);
      }
    }

    int votes = 0;
    for (int k = 0; k < K; k++) {
      if (outIds[k] >= 0 && labels[outIds[k]] == 1) votes++;
    }
    return votes;
  }

  private float simdDist(final short[] q, final int refIdx) {
    final int off = refIdx * PADDED;
    final ShortVector vq = ShortVector.fromArray(S16, q, 0);
    final ShortVector vr = ShortVector.fromArray(S16, vectors, off);
    final ShortVector diff = vq.sub(vr);
    final IntVector lo = (IntVector) diff.convertShape(VectorOperators.S2I, S32, 0);
    final IntVector hi = (IntVector) diff.convertShape(VectorOperators.S2I, S32, 1);
    final int sum = lo.mul(lo).add(hi.mul(hi)).reduceLanes(VectorOperators.ADD);
    return (float) sum / (SCALE * SCALE);
  }

  private static void insert(
    final int[] ids, final float[] dists, final int max,
    final int newId, final float newDist
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

  public boolean isReady() {
    return ready;
  }
}
