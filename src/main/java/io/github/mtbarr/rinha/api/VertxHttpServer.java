package io.github.mtbarr.rinha.api;

import io.github.mtbarr.rinha.index.InvertedFileIndex;
import io.github.mtbarr.rinha.service.FraudRequestParser;
import io.quarkus.runtime.StartupEvent;
import io.vertx.core.Vertx;
import io.vertx.core.buffer.Buffer;
import io.vertx.core.http.HttpServer;
import io.vertx.core.http.HttpServerOptions;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.enterprise.event.Observes;
import jakarta.inject.Inject;
import java.nio.charset.StandardCharsets;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

@ApplicationScoped
public class VertxHttpServer {

  private static final Buffer EMPTY_RESPONSE = Buffer.buffer("{\"approved\":true,\"fraud_score\":0.0}".getBytes(StandardCharsets.UTF_8));
  private static final Buffer NOT_FOUND = Buffer.buffer("Not found".getBytes(StandardCharsets.UTF_8));
  private static final Buffer READY_RESPONSE = Buffer.buffer("OK".getBytes(StandardCharsets.UTF_8));

  @Inject
  Vertx vertx;

  @Inject
  InvertedFileIndex vectorIndex;

  @Inject
  FraudRequestParser fraudRequestParser;

  void onStart(@Observes StartupEvent event) {
    final HttpServerOptions options = new HttpServerOptions()
      .setPort(8080)
      .setHost("0.0.0.0")
      .setMaxWebSocketFrameSize(0)
      .setCompressionSupported(false)
      .setTcpFastOpen(true)
      .setTcpNoDelay(true)
      .setAcceptBacklog(4096);

    final ExecutorService workerPool = Executors.newFixedThreadPool(1);

    vertx.createHttpServer(options)
      .requestHandler(request -> {
        final String path = request.path();
        if ("GET".equals(request.method().name()) && "/ready".equals(path)) {
          request.response()
            .putHeader("Content-Type", "text/plain")
            .end(READY_RESPONSE);
          return;
        }
        if ("POST".equals(request.method().name()) && "/fraud-score".equals(path)) {
          workerPool.execute(() -> handleFraudScore(request));
          return;
        }
        request.response()
          .setStatusCode(404)
          .end(NOT_FOUND);
      })
      .listen();
  }

  private void handleFraudScore(final io.vertx.core.http.HttpServerRequest request) {
    request.bodyHandler(body -> {
      final byte[] rawBytes = body.getBytes();
      Buffer responsePayload;
      if (!vectorIndex.isReady()) {
        responsePayload = EMPTY_RESPONSE;
      } else {
        responsePayload = computeResponse(rawBytes);
      }
      request.response()
        .putHeader("Content-Type", "application/json")
        .end(responsePayload);
    });
    request.exceptionHandler(throwable -> {
      try {
        request.response()
          .setStatusCode(500)
          .end(EMPTY_RESPONSE);
      } catch (final Exception ignored) {
      }
    });
  }

  private Buffer computeResponse(final byte[] rawBytes) {
    try {
      final float[] featureVector = fraudRequestParser.extractVector(rawBytes);
      final int fraudVotes = vectorIndex.search(featureVector);
      final double fraudScore = fraudVotes * 0.2;
      final boolean isApproved = fraudVotes < 3;
      return Buffer.buffer(("{\"approved\":" + isApproved + ",\"fraud_score\":" + fraudScore + "}").getBytes(StandardCharsets.UTF_8));
    } catch (final Exception unexpectedError) {
      return EMPTY_RESPONSE;
    }
  }
}
