FROM python:3.12-slim-trixie
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set the current working directory to usr/vector-dbs-crud.
# This is where we'll put the requirements-dev.txt file and the app directory.
WORKDIR /usr/repos/vector-dbs-crud

# Copy all application files to the container
COPY . /usr/repos/vector-dbs-crud

# install system dependencies
RUN apt-get update \
  && apt-get -y install \
  curl \
  make \
  git \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

# install python dependencies
RUN make install-dev

# Keep the container running for interactive use
CMD ["tail", "-f", "/dev/null"]
