# Stage 1: Build index offline (quantizacao i16 + escrita flat_index.bin)
FROM eclipse-temurin:25-jdk AS index-builder
WORKDIR /build

COPY gradlew gradlew.bat ./
RUN chmod +x gradlew
COPY gradle ./gradle
COPY build.gradle.kts settings.gradle.kts gradle.properties ./
RUN ./gradlew :compileJava --no-daemon || true

COPY src ./src
RUN ./gradlew :compileJava --no-daemon

RUN mkdir -p /data && java \
    -Xmx4g \
    -cp "$(./gradlew -q :printClasspath --no-daemon):build/classes/java/main" \
    io.github.mtbarr.rinha.index.OfflineIndexBuilder \
    src/main/resources/references.json.gz \
    /data/flat_index.bin

# Stage 2: Native image com GraalVM CE 25
FROM ghcr.io/graalvm/graalvm-community:25 AS native-builder
WORKDIR /build
COPY . .
COPY --from=index-builder /data/flat_index.bin src/main/resources/flat_index.bin
RUN chmod +x gradlew
RUN ./gradlew quarkusBuild \
    -Dquarkus.package.type=native \
    --no-daemon

# Stage 3: Runtime (flat_index.bin embeddado no binario — zero I/O de disco)
FROM gcr.io/distroless/base-debian12
WORKDIR /app

COPY --from=native-builder /build/build/*-runner /app/application
COPY --from=native-builder /usr/lib64/libz.so.1 /lib/x86_64-linux-gnu/libz.so.1

EXPOSE 8080

ENTRYPOINT ["/app/application"]
