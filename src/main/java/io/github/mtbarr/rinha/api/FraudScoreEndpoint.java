package io.github.mtbarr.rinha.api;

import io.github.mtbarr.rinha.index.InvertedFileIndex;
import io.github.mtbarr.rinha.service.FraudRequestParser;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import jakarta.ws.rs.Consumes;
import jakarta.ws.rs.POST;
import jakarta.ws.rs.Path;
import jakarta.ws.rs.Produces;
import jakarta.ws.rs.core.MediaType;
import java.nio.charset.StandardCharsets;

@Path("/fraud-score")
@ApplicationScoped
public class FraudScoreEndpoint {

  private static final byte[] EMPTY_RESPONSE = "{\"approved\":true,\"fraud_score\":0.0}"
    .getBytes(StandardCharsets.UTF_8);

  @Inject
  InvertedFileIndex vectorIndex;

  @Inject
  FraudRequestParser fraudRequestParser;

  @POST
  @Consumes(MediaType.APPLICATION_JSON)
  @Produces(MediaType.APPLICATION_OCTET_STREAM)
  public byte[] computeFraudScore(final byte[] requestBody) {
    if (!vectorIndex.isReady()) {
      return EMPTY_RESPONSE;
    }
    try {
      final float[] featureVector = fraudRequestParser.extractVector(requestBody);
      final int fraudVotes = vectorIndex.search(featureVector);
      final double fraudScore = fraudVotes * 0.2;
      final boolean isApproved = fraudVotes < 3;
      return ("{\"approved\":" + isApproved + ",\"fraud_score\":" + fraudScore + "}").getBytes(StandardCharsets.UTF_8);
    } catch (final Exception unexpectedError) {
      return EMPTY_RESPONSE;
    }
  }
}
