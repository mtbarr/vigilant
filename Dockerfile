# Stage 1: Build index offline (compila so o builder, zero deps)
FROM eclipse-temurin:25-jdk AS index-builder
WORKDIR /build
COPY src/main/java/io/github/mtbarr/rinha/index/OfflineIndexBuilder.java \
     src/main/java/io/github/mtbarr/rinha/index/OfflineIndexBuilder.java
RUN mkdir -p classes && javac -d classes \
    src/main/java/io/github/mtbarr/rinha/index/OfflineIndexBuilder.java
COPY src/main/resources/references.json.gz src/main/resources/references.json.gz
RUN java -Xmx4g -cp classes \
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
    -Dquarkus.native.additional-build-args="-march=haswell,--gc=serial" \
    --no-daemon

# Stage 3: Runtime com Debian 12 (GLIBC 2.36, suporta o binario nativo)
FROM gcr.io/distroless/base-debian12
WORKDIR /app

COPY --from=native-builder /build/build/*-runner /app/application
COPY --from=index-builder /data/index.bin /app/data/index.bin

COPY --from=native-builder /usr/lib64/libz.so.1 /lib/x86_64-linux-gnu/libz.so.1

EXPOSE 8080

ENTRYPOINT ["/app/application"]
