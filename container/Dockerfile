FROM ubuntu18:py38

USER root
ARG WORKPATH=/home/uniview
WORKDIR $WORKPATH

####### pip cpu requirements start ######
RUN ln -sf /usr/bin/python3 /usr/bin/python
RUN ln -sf /usr/bin/pip3 /usr/bin/pip

### production or development
### set ARG values using –build-arg: $ docker build --build-arg ENV_MODE=dev
ARG ENV_MODE

ENV ENV_MODE=${ENV_MODE} \
  PYTHONFAULTHANDLER=1 \
  PYTHONUNBUFFERED=1 \
  PYTHONHASHSEED=random \
  PIP_NO_CACHE_DIR=off \
  PIP_DISABLE_PIP_VERSION_CHECK=on \
  PIP_DEFAULT_TIMEOUT=100 \
  POETRY_VERSION=1.0.10

# System deps:
RUN pip install "poetry==$POETRY_VERSION"

# Creating folders, and files for a project:
COPY . $WORKPATH/

# Project initialization:
RUN poetry config virtualenvs.create false \
  && poetry install $(test "$ENV_MODE" == prod && echo "--no-dev") --no-interaction --no-ansi
