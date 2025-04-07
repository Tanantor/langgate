FROM envoyproxy/envoy:v1.33-latest

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3=3.10* \
        python3-pip=22.* && \
    pip3 install --no-cache-dir jinja2==3.1.* && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    mkdir -p /etc/envoy/config.d

COPY services/envoy/scripts/ /etc/envoy/scripts/

EXPOSE 10000

ENTRYPOINT ["/bin/sh", "/etc/envoy/scripts/start.sh"]
