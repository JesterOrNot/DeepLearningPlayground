FROM gitpod/workspace-full

RUN sudo apt-get update -qq \
    && sudo apt-get install -yq \
       graphviz \
       xdg-utils \
       texlive-full \
       inotify-tools \
    && sudo rm -rf /var/lib/apt/lists/*

RUN brew install pandoc
