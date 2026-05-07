# Dockerfile.app — build rapido p/ testes (index.bin ja deve existir em data/)
FROM ghcr.io/graalvm/graalvm-community:25 AS native-builder
WORKDIR /build
COPY . .
RUN chmod +x gradlew
RUN ./gradlew quarkusBuild \
    -Dquarkus.package.type=native \
    -Dquarkus.native.additional-build-args="-march=haswell,--add-modules=jdk.incubator.vector" \
    --no-daemon

FROM gcr.io/distroless/base-debian12
WORKDIR /app

COPY --from=native-builder /build/build/*-runner /app/application
COPY data/index.bin /app/data/index.bin
COPY --from=native-builder /usr/lib64/libz.so.1 /lib/x86_64-linux-gnu/libz.so.1

EXPOSE 8080
ENTRYPOINT ["/app/application"]
