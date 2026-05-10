package io.github.mtbarr.rinha.api;

import io.github.mtbarr.rinha.index.InvertedFileIndex;
import io.github.mtbarr.rinha.service.FraudRequestParser;
import io.quarkus.runtime.StartupEvent;
import io.vertx.core.Vertx;
import io.vertx.core.http.HttpMethod;
import io.vertx.core.http.HttpServerOptions;
import io.vertx.core.http.HttpServerResponse;
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

  private static final String PATH_READY = "/ready";
  private static final String PATH_FRAUD_SCORE = "/fraud-score";
  private static final String CONTENT_TYPE_JSON = "application/json";

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

  void onStart(@Observes final StartupEvent startupEvent) {
    final HttpServerOptions serverOptions = new HttpServerOptions()
      .setPort(8080)
      .setHost("0.0.0.0")
      .setCompressionSupported(false)
      .setTcpFastOpen(true)
      .setTcpNoDelay(true)
      .setAcceptBacklog(4_096);

    vertxEngine.createHttpServer(serverOptions)
      .requestHandler(httpRequest -> {
        final String path = httpRequest.path();

        if (PATH_FRAUD_SCORE.equals(path)) {
          if (httpRequest.method() != HttpMethod.POST) {
            httpRequest.response().setStatusCode(405).end();
            return;
          }
          httpRequest.bodyHandler(requestBody -> {
            final HttpServerResponse response = httpRequest.response();
            response.putHeader("Content-Type", CONTENT_TYPE_JSON);
            try {
              response.end(buildFraudScoreResponse(requestBody.getBytes()));
            } catch (final Exception unexpectedError) {
              response.end(FRAUD_SCORE_RESPONSES[0]);
            }
          });
          return;
        }

        if (PATH_READY.equals(path)) {
          httpRequest.response()
            .setStatusCode(fraudVectorIndex.isReady() ? 200 : 503)
            .end();
          return;
        }

        httpRequest.response().setStatusCode(404).end();
      })
      .listen();
  }

  private String buildFraudScoreResponse(final byte[] requestPayload) {
    final float[] featureVector = requestFeatureExtractor.extractFeatureVector(requestPayload);
    final int fraudVoteCount = fraudVectorIndex.searchNearestNeighbors(
      featureVector,
      neighborIdBuffer.get(),
      neighborDistanceBuffer.get()
    );
    return FRAUD_SCORE_RESPONSES[fraudVoteCount];
  }
}