FROM gitpod/workspace-full

RUN sudo apt-get update -qq \
    && sudo apt-get install -yq \
       graphviz
