FROM python:3.11-slim

# Copy the built distribution files from the Poetry build
COPY dist/*.whl /app/
# Copy the data folder recursively from the source directory
#COPY src/omni_aiq_usd_code/data/ /app/data/
# Copy the examples directory with the config file
COPY examples/ /app/examples/

WORKDIR /app

# Install the package from the wheel file
RUN pip install --no-cache-dir *.whl

# Set environment variables
# Create Docker environment marker
RUN touch /.dockerenv

# Create non-root user for security
RUN groupadd --gid 1000 appgroup \
    && useradd --uid 1000 --gid appgroup --shell /bin/bash --create-home appuser \
    && chown -R appuser:appgroup /app

# Switch to non-root user
USER appuser

# Set the command to run the application
# Using the console script defined in pyproject.toml
CMD ["usd-code-aiq"]
