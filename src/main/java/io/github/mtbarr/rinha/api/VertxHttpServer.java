package io.github.mtbarr.rinha.api;

import io.github.mtbarr.rinha.index.InvertedFileIndex;
import io.github.mtbarr.rinha.service.FraudRequestParser;
import io.quarkus.runtime.StartupEvent;
import io.vertx.core.Vertx;
import io.vertx.core.http.HttpServerOptions;
import jakarta.enterprise.event.Observes;
import jakarta.inject.Inject;
import jakarta.inject.Singleton;

@Singleton
public final class VertxHttpServer {

  private static final int NEIGHBOR_COUNT = 10;

  private static final String[] FRAUD_SCORE_RESPONSES = {
    "{\"approved\":true,\"fraud_score\":0.0}",
    "{\"approved\":true,\"fraud_score\":0.2}",
    "{\"approved\":true,\"fraud_score\":0.4}",
    "{\"approved\":false,\"fraud_score\":0.6}",
    "{\"approved\":false,\"fraud_score\":0.8}",
    "{\"approved\":false,\"fraud_score\":1.0}"
  };

  private static final String HEALTH_CHECK_RESPONSE = "OK";
  private static final String NOT_FOUND_RESPONSE = "Not found";

  private static final String HTTP_METHOD_GET = "GET";
  private static final String HTTP_METHOD_POST = "POST";
  private static final String PATH_READY = "/ready";
  private static final String PATH_FRAUD_SCORE = "/fraud-score";
  private static final String CONTENT_TYPE_JSON = "application/json";
  private static final String CONTENT_TYPE_TEXT = "text/plain";

  private final ThreadLocal<int[]> neighborIdBuffer = ThreadLocal.withInitial(
    () -> new int[NEIGHBOR_COUNT]
  );
  private final ThreadLocal<float[]> neighborDistanceBuffer = ThreadLocal.withInitial(
    () -> new float[NEIGHBOR_COUNT]
  );

  @Inject
  Vertx vertxEngine;

  @Inject
  InvertedFileIndex fraudVectorIndex;

  @Inject
  FraudRequestParser requestFeatureExtractor;

  void onStart(final @Observes StartupEvent startupEvent) {
    fraudVectorIndex.isReady();

    final HttpServerOptions serverOptions = new HttpServerOptions()
      .setPort(8080)
      .setHost("0.0.0.0")
      .setCompressionSupported(false)
      .setTcpFastOpen(true)
      .setTcpNoDelay(true)
      .setAcceptBacklog(4096);

    vertxEngine.createHttpServer(serverOptions)
      .requestHandler(httpRequest -> {
        final String requestMethod = httpRequest.method().name();
        final String requestPath = httpRequest.path();
        if (HTTP_METHOD_GET.equals(requestMethod) && PATH_READY.equals(requestPath)) {
          httpRequest.response()
            .putHeader("Content-Type", CONTENT_TYPE_TEXT)
            .setStatusCode(fraudVectorIndex.isReady() ? 200 : 503)
            .end(fraudVectorIndex.isReady() ? HEALTH_CHECK_RESPONSE : "Not ready");
          return;
        }
        if (HTTP_METHOD_POST.equals(requestMethod) && PATH_FRAUD_SCORE.equals(requestPath)) {
          httpRequest.bodyHandler(requestBody -> {
            final String payload = requestBody.toString();
            vertxEngine.executeBlocking(
              () -> buildFraudScoreResponse(payload),
              false
            ).onSuccess(responsePayload -> {
              httpRequest.response()
                .putHeader("Content-Type", CONTENT_TYPE_JSON)
                .end(responsePayload);
            }).onFailure(err -> {
              httpRequest.response()
                .putHeader("Content-Type", CONTENT_TYPE_JSON)
                .end(FRAUD_SCORE_RESPONSES[0]);
            });
          });
          return;
        }
        httpRequest.response()
          .setStatusCode(404)
          .end(NOT_FOUND_RESPONSE);
      })
      .listen();
  }

  private String buildFraudScoreResponse(final String requestPayload) {
    try {
      final float[] featureVector = requestFeatureExtractor.extractFeatureVector(requestPayload);
      final int[] neighborIds = neighborIdBuffer.get();
      final float[] neighborDistances = neighborDistanceBuffer.get();
      final int fraudVoteCount = fraudVectorIndex.searchNearestNeighbors(
        featureVector,
        neighborIds,
        neighborDistances
      );
      return FRAUD_SCORE_RESPONSES[fraudVoteCount];
    } catch (final Exception processingError) {
      return FRAUD_SCORE_RESPONSES[0];
    }
  }
}
