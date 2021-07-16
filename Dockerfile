# downloads the neccessary tensorflow packages instead of using pip
FROM tensorflow/tensorflow:latest-gpu

# install neccessary dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        wget \
        curl \
        software-properties-common

RUN pip install --upgrade pip && pip install setuptools

# for hyper parameter tuning
RUN pip3 install cloudml-hypertune

# Installs google cloud sdk, this is mostly for using gsutil to export model.
RUN wget -nv \
    https://dl.google.com/dl/cloudsdk/release/google-cloud-sdk.tar.gz && \
    mkdir /root/tools && \
    tar xvzf google-cloud-sdk.tar.gz -C /root/tools && \
    rm google-cloud-sdk.tar.gz && \
    /root/tools/google-cloud-sdk/install.sh --usage-reporting=false \
        --path-update=false --bash-completion=false \
        --disable-installation-options && \
    rm -rf /root/.config/* && \
    ln -s /root/.config /config && \
    # Remove the backup directory that gcloud creates
    rm -rf /root/tools/google-cloud-sdk/.install/.backup

# Path configuration
ENV PATH $PATH:/root/tools/google-cloud-sdk/bin
# Make sure gsutil will use the default service account
RUN echo '[GoogleCompute]\nservice_account = default' > /etc/boto.cfg



WORKDIR /usr/src/app

# copies all ther files in the above working directory to the docker image
COPY . .

RUN apt-get update

RUN pip install -r /usr/src/app/requirements.txt
# allows to run python file
ENTRYPOINT ["python3", "/usr/src/app/trainer.py"]

# docker build -f Dockerfile -t food_trainer:test-gpu-2 .
# since theres no support, dont need to run, we can just run on the cloud
# docker run --gpus all -it food_trainer:test-gpu-1 --batch_size 1