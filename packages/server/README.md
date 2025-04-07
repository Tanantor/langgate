# LangGate Server

Server implementation for the LangGate AI Gateway. This package provides the FastAPI server implementation that hosts the LangGate registry API.

## Features

- FastAPI-based API server
- Registry endpoints for model information
- Configurable settings through environment variables or config files
- Structured logging with request context

## Usage

```python
# Run the server with uvicorn
import uvicorn
from langgate.server import app

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=4000)
```

## Docker Deployment

The LangGate repository includes a Dockerfile specifically for running this server:

```bash
# Build the Docker image
docker build -t langgate-server .

# Run the container
docker run -p 4000:4000 langgate-server
```

You can also use the provided Docker Compose setup for a complete deployment (which runs the server behind an Envoy proxy on port 10000):

```bash
# Start the full LangGate stack
make compose-up

# For development with hot reloading
make compose-dev
```

## Configuration

The server can be configured through:
- Environment variables
- A `.env` file in your working directory
- A langgate_config.yaml file in your working directory
- Command-line arguments to the uvicorn server

See the main LangGate documentation for more details on configuration options.
