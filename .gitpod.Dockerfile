FROM gitpod/workspace-full:latest

RUN sudo apt-get update -qq \
    && sudo apt-get install -yq \
       graphviz \
       xdg-utils \
    && sudo rm -rf /var/lib/apt/lists/*
