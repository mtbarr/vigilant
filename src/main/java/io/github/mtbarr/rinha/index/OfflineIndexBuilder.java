package io.github.mtbarr.rinha.index;

import java.io.FileInputStream;
import java.io.RandomAccessFile;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.zip.GZIPInputStream;

public final class OfflineIndexBuilder {

  private static final int DIMENSIONS = 14;
  private static final int PADDED = 16;
  private static final float SCALE = 8192f;

  private OfflineIndexBuilder() {
  }

  public static void main(final String[] args) throws Exception {
    if (args.length < 2) {
      System.err.println("Usage: OfflineIndexBuilder <references.json.gz> <flat_index.bin>");
      System.exit(1);
    }

    final String inputPath = args[0];
    final String outputPath = args[1];

    System.out.println("Loading dataset...");
    final byte[] rawData;
    try (final var gz = new GZIPInputStream(new FileInputStream(inputPath))) {
      rawData = gz.readAllBytes();
    }

    final float[][] vectors = parseFeatureVectors(rawData);
    final byte[] labels = parseFraudLabels(rawData);
    final int count = vectors.length;
    System.out.println("  " + count + " vectors, fraud=" + countFraud(labels));

    System.out.println("Quantizing to i16...");
    final short[] quantized = new short[count * PADDED];
    for (int i = 0; i < count; i++) {
      int off = i * PADDED;
      for (int d = 0; d < DIMENSIONS; d++) {
        final int v = Math.round(vectors[i][d] * SCALE);
        quantized[off + d] = (short) Math.max(Short.MIN_VALUE, Math.min(Short.MAX_VALUE, v));
      }
    }

    System.out.println("Writing flat_index.bin...");
    try (final var raf = new RandomAccessFile(outputPath, "rw");
      final var ch = raf.getChannel()) {
      final long totalSize = 20L + (long) count * PADDED * 2L + count;
      final ByteBuffer buf = ch.map(FileChannel.MapMode.READ_WRITE, 0, totalSize);
      buf.order(ByteOrder.LITTLE_ENDIAN);
      buf.putInt(0x464C4154);
      buf.putInt(1);
      buf.putInt(count);
      buf.putInt(PADDED);
      buf.putInt((int) SCALE);
      for (int i = 0; i < count * PADDED; i++) {
        buf.putShort(quantized[i]);
      }
      buf.put(labels);
    }

    System.out.println("Done. " + count + " vectors, "
                       + (count * PADDED * 2 + count) / 1024 + " KB");
  }

  private static float[][] parseFeatureVectors(final byte[] rawData) {
    final int estimated = 3_200_000;
    final float[] flat = new float[estimated * DIMENSIONS];
    int count = 0;
    int pos = 0;
    final byte[] tag = "\"vector\":[".getBytes(StandardCharsets.US_ASCII);
    while (pos < rawData.length) {
      final int tagIdx = find(rawData, tag, pos);
      if (tagIdx < 0) {
        break;
      }
      int cur = tagIdx + tag.length;
      final int base = count * DIMENSIONS;
      for (int d = 0; d < DIMENSIONS; d++) {
        while (cur < rawData.length && (rawData[cur] == ' ' || rawData[cur] == '\n' || rawData[cur] == '\r')) {
          cur++;
        }
        final int end = findDelim(rawData, cur);
        flat[base + d] = parseFloat(rawData, cur, end);
        cur = end + 1;
      }
      count++;
      pos = cur;
    }
    final float[][] result = new float[count][DIMENSIONS];
    for (int i = 0; i < count; i++) {
      System.arraycopy(flat, i * DIMENSIONS, result[i], 0, DIMENSIONS);
    }
    return result;
  }

  private static byte[] parseFraudLabels(final byte[] rawData) {
    final int estimated = 3_200_000;
    final byte[] labels = new byte[estimated];
    int count = 0;
    int pos = 0;
    final byte[] tag = "\"label\":\"".getBytes(StandardCharsets.US_ASCII);
    while (pos < rawData.length) {
      final int t = find(rawData, tag, pos);
      if (t < 0) {
        break;
      }
      labels[count++] = rawData[t + tag.length] == 'f' ? (byte) 1 : (byte) 0;
      pos = t + tag.length + 6;
    }
    return Arrays.copyOf(labels, count);
  }

  private static int countFraud(final byte[] labels) {
    int n = 0;
    for (final byte b : labels) {
      if (b == 1) {
        n++;
      }
    }
    return n;
  }

  private static int find(final byte[] data, final byte[] pat, final int from) {
    final int limit = data.length - pat.length;
    for (int i = from; i <= limit; i++) {
      boolean match = true;
      for (int j = 0; j < pat.length; j++) {
        if (data[i + j] != pat[j]) {
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

  private static int findDelim(final byte[] d, final int from) {
    int i = from;
    while (i < d.length && d[i] != ',' && d[i] != ']') {
      i++;
    }
    return i;
  }

  private static float parseFloat(final byte[] d, int start, final int end) {
    float r = 0;
    boolean neg = false;
    if (start < end && d[start] == '-') {
      neg = true;
      start++;
    }
    while (start < end && d[start] >= '0' && d[start] <= '9') {
      r = r * 10 + (d[start] - '0');
      start++;
    }
    if (start < end && d[start] == '.') {
      start++;
      float f = 0, div = 1;
      while (start < end && d[start] >= '0' && d[start] <= '9') {
        f = f * 10 + (d[start] - '0');
        div *= 10;
        start++;
      }
      r += f / div;
    }
    return neg ? -r : r;
  }
}
