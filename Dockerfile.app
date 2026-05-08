# Dockerfile.app — build rapido p/ testes (flat_index.bin gerado offline em src/main/resources/)
# NOTA: o flat_index.bin é embeddado no binario nativo via quarkus.native.resources.includes
FROM ghcr.io/graalvm/graalvm-community:25 AS native-builder
WORKDIR /build
COPY . .
RUN chmod +x gradlew
RUN ./gradlew quarkusBuild \
    -Dquarkus.package.type=native \
    --no-daemon

FROM gcr.io/distroless/base-debian12
WORKDIR /app

COPY --from=native-builder /build/build/*-runner /app/application

EXPOSE 8080
ENTRYPOINT ["/app/application"]
