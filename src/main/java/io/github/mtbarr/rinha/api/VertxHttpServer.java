package io.github.mtbarr.rinha.api;

import io.github.mtbarr.rinha.index.InvertedFileIndex;
import io.github.mtbarr.rinha.service.FraudRequestParser;
import io.quarkus.runtime.StartupEvent;
import io.vertx.core.Vertx;
import io.vertx.core.http.HttpMethod;
import io.vertx.core.http.HttpServerOptions;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.enterprise.event.Observes;
import jakarta.inject.Inject;

@ApplicationScoped
public class VertxHttpServer {

  private static final String PATH_FRAUD_SCORE = "/fraud-score";
  private static final String PATH_READY = "/ready";
  private static final String CONTENT_TYPE_JSON = "application/json";

  /**
   * As únicas 6 respostas possíveis, pré-montadas no startup. Indexado por fraudVotes (0..5) — elimina String
   * concatenation e o problema de representação float (ex: 3 * 0.2 = 0.6000000000000001).
   */
  private static final String[] SCORE_RESPONSES = {
    "{\"approved\":true,\"fraud_score\":0.0000}",
    "{\"approved\":true,\"fraud_score\":0.2000}",
    "{\"approved\":true,\"fraud_score\":0.4000}",
    "{\"approved\":false,\"fraud_score\":0.6000}",
    "{\"approved\":false,\"fraud_score\":0.8000}",
    "{\"approved\":false,\"fraud_score\":1.0000}",
    };

  @Inject
  Vertx vertx;

  @Inject
  InvertedFileIndex vectorIndex;

  @Inject
  FraudRequestParser fraudRequestParser;

  void onStart(@Observes final StartupEvent event) {
    final HttpServerOptions options = new HttpServerOptions()
      .setPort(8080)
      .setHost("0.0.0.0")
      .setCompressionSupported(false)
      .setTcpFastOpen(true)
      .setTcpNoDelay(true)
      .setAcceptBacklog(4_096);

    vertx.createHttpServer(options)
      .requestHandler(req -> {
        final String path = req.path();

        // Hot path primeiro — /fraud-score é chamado ordens de grandeza mais do que /ready
        if (PATH_FRAUD_SCORE.equals(path)) {
          if (req.method() != HttpMethod.POST) {
            req.response().setStatusCode(405).end();
            return;
          }
          req.bodyHandler(body -> {
            final String response;
            if (!vectorIndex.isReady()) {
              response = SCORE_RESPONSES[0];
            } else {
              response = computeResponse(body.getBytes());
            }
            req.response()
              .putHeader("Content-Type", CONTENT_TYPE_JSON)
              .end(response);
          });
          return;
        }

        if (PATH_READY.equals(path)) {
          req.response()
            .setStatusCode(vectorIndex.isReady() ? 200 : 503)
            .end();
          return;
        }

        req.response().setStatusCode(404).end();
      })
      .listen();
  }

  private String computeResponse(final byte[] rawBytes) {
    try {
      final float[] featureVector = fraudRequestParser.extractVector(rawBytes);
      return SCORE_RESPONSES[vectorIndex.search(featureVector)];
    } catch (final Exception unexpectedError) {
      return SCORE_RESPONSES[0];
    }
  }
}