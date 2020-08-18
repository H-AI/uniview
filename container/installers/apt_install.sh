apt-get -qq -y update && \
DEBIAN_FRONTEND=noninteractive apt-get -qq -y install \
    gcc \
    g++ \
    zlibc \
    zlib1g-dev \
    libssl-dev \
    libbz2-dev \
    libsqlite3-dev \
    libncurses5-dev \
    libgdbm-dev \
    libgdbm-compat-dev \
    liblzma-dev \
    libreadline-dev \
    uuid-dev \
    libffi-dev \
    tk-dev \
    wget \
    curl \
    git \
    make \
    sudo \
    bash-completion \
    tree \
    vim \
    zip \
    unzip \
    software-properties-common && \
mv /usr/bin/lsb_release /usr/bin/lsb_release.bak && \
apt-get -y autoclean && \
apt-get -y autoremove && \
rm -rf /var/lib/apt-get/lists/*

