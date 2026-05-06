# Stage 1: Build index offline (K-Means + quantizacao)
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
    /data/index.bin

# Stage 2: Native image com GraalVM CE 25
FROM ghcr.io/graalvm/graalvm-community:25 AS native-builder
WORKDIR /build
COPY . .
RUN chmod +x gradlew
RUN ./gradlew quarkusBuild \
    -Dquarkus.package.type=native \
    -Dquarkus.native.additional-build-args="-J--add-modules=jdk.incubator.vector,-march=haswell" \
    --no-daemon

# Stage 3: Runtime minimo (so o binario nativo)
FROM quay.io/quarkus/quarkus-micro-image:2.0
WORKDIR /app

COPY --from=native-builder /build/build/*-runner /app/application
COPY --from=index-builder /data/index.bin /app/data/index.bin
COPY src/main/resources/mcc_risk.json /app/data/mcc_risk.json

RUN chmod +x /app/application

EXPOSE 8080

ENTRYPOINT ["/app/application"]
