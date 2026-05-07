package io.github.mtbarr.rinha.api;

import io.github.mtbarr.rinha.index.InvertedFileIndex;
import io.github.mtbarr.rinha.service.FraudRequestParser;
import io.quarkus.runtime.StartupEvent;
import io.vertx.core.Vertx;
import io.vertx.core.http.HttpServerOptions;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.enterprise.event.Observes;
import jakarta.inject.Inject;
import java.nio.charset.StandardCharsets;

@ApplicationScoped
public class VertxHttpServer {

  private static final String EMPTY_RESPONSE = "{\"approved\":true,\"fraud_score\":0.0}";
  private static final String NOT_FOUND = "Not found";
  private static final String READY_RESPONSE = "OK";

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
      .setCompressionSupported(false)
      .setTcpFastOpen(true)
      .setTcpNoDelay(true)
      .setAcceptBacklog(4096);

    vertx.createHttpServer(options)
      .requestHandler(request -> {
        final String method = request.method().name();
        final String path = request.path();
        if ("GET".equals(method) && "/ready".equals(path)) {
          request.response()
            .putHeader("Content-Type", "text/plain")
            .end(READY_RESPONSE);
          return;
        }
        if ("POST".equals(method) && "/fraud-score".equals(path)) {
          request.bodyHandler(body -> {
            final String responsePayload;
            if (!vectorIndex.isReady()) {
              responsePayload = EMPTY_RESPONSE;
            } else {
              responsePayload = computeResponse(body.getBytes());
            }
            request.response()
              .putHeader("Content-Type", "application/json")
              .end(responsePayload);
          });
          return;
        }
        request.response()
          .setStatusCode(404)
          .end(NOT_FOUND);
      })
      .listen();
  }

  private String computeResponse(final byte[] rawBytes) {
    try {
      final float[] featureVector = fraudRequestParser.extractVector(rawBytes);
      final int fraudVotes = vectorIndex.search(featureVector);
      final double fraudScore = fraudVotes * 0.2;
      final boolean isApproved = fraudVotes < 3;
      return "{\"approved\":" + isApproved + ",\"fraud_score\":" + fraudScore + "}";
    } catch (final Exception unexpectedError) {
      return EMPTY_RESPONSE;
    }
  }
}
