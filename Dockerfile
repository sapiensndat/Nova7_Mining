# Stage 1: Serve the application with Caddy
# This uses the caddy:2-alpine image as the base.
FROM caddy:2-alpine

# Set the working directory inside the container.
WORKDIR /app

# Install Python, pip, and a production WSGI server (gunicorn).
# The --no-cache flag reduces the size of the final image.
RUN apk add --no-cache python3 py3-pip py3-gunicorn

# Copy the requirements file first to take advantage of Docker's layer caching.
COPY requirements.txt .

# Copy all application files from the root directory into the container's /app directory.
# This step is where your JSON key file is copied (unless it's ignored by .dockerignore).
COPY . .

# Install dependencies from requirements.txt.
# The --break-system-packages flag is sometimes needed in Alpine to install packages without conflicts.
RUN python3 -m pip install --no-cache-dir -r requirements.txt --break-system-packages

# Expose port 8080 so that Caddy can be reached from outside the container.
EXPOSE 8080

# The command to run when the container starts.
# We've added "--forwarded-allow-ips '*'" to tell gunicorn to trust headers from the proxy.
CMD ["sh", "-c", "gunicorn --bind localhost:9000 --forwarded-allow-ips '*' gold:app & caddy run --config Caddyfile --adapter caddyfile"]
