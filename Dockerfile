# Stage 1: Build index offline (K-Means + quantizacao)
FROM eclipse-temurin:25-jdk AS index-builder
WORKDIR /build

COPY gradlew gradlew.bat ./
COPY gradle ./gradle
COPY build.gradle.kts settings.gradle.kts gradle.properties ./
RUN ./gradlew :compileJava --no-daemon || true

COPY src ./src
COPY data /data

RUN ./gradlew :compileJava --no-daemon

RUN java \
    -Xmx4g \
    -cp "$(./gradlew -q :printClasspath --no-daemon):build/classes/java/main" \
    io.github.mtbarr.rinha.index.OfflineIndexBuilder \
    /data/references.json.gz \
    /data/index.bin


# Stage 2: Native image com GraalVM CE 25
FROM ghcr.io/graalvm/graalvm-community:25 AS native-builder
WORKDIR /build
COPY . .
RUN ./gradlew build \
    -Dquarkus.native.enabled=true \
    -Dquarkus.native.additional-build-args="-J--add-modules=jdk.incubator.vector,-march=haswell" \
    --no-daemon


# Stage 3: Runtime minimo (so o binario nativo)
FROM quay.io/quarkus/quarkus-micro-image:2.0
WORKDIR /app

COPY --from=native-builder /build/build/*-runner /app/application
COPY --from=index-builder /data/index.bin /app/data/index.bin
COPY data/mcc_risk.json /app/data/

RUN chmod +x /app/application

EXPOSE 8080

ENTRYPOINT ["/app/application"]
