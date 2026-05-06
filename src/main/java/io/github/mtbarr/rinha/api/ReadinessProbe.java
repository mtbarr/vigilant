package io.github.mtbarr.rinha.api;

import io.github.mtbarr.rinha.index.InvertedFileIndex;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import jakarta.ws.rs.GET;
import jakarta.ws.rs.Path;
import jakarta.ws.rs.core.Response;

@Path("/ready")
@ApplicationScoped
public class ReadinessProbe {

  @Inject
  InvertedFileIndex vectorIndex;

  @GET
  public Response checkReadiness() {
    if (vectorIndex.isReady()) {
      return Response.ok().build();
    }
    return Response.status(Response.Status.SERVICE_UNAVAILABLE).build();
  }
}
