# Compute Engine Instance parameters
export IMAGE_FAMILY="tf-latest-cu100"
export ZONE="us-west1-b"
export INSTANCE_NAME="notebook-executor"
export INSTANCE_TYPE="n1-standard-8"
export ACCELERATOR="type=nvidia-tesla-v100,count=1"

# Notebook parameters
export INPUT_NOTEBOOK_PATH="master-thesis/notebooks/neural_architecture_search.ipynb"
export OUTPUT_NOTEBOOK_PATH="master-thesis/notebooks/neural_architecture_search-output.ipynb"
export PARAMETERS_FILE="params.yaml" # Optional
export PARAMETERS="-p batch_size 128 -p epochs 40"  # Optional
export STARTUP_SCRIPT="pip install wandb && papermill ${INPUT_NOTEBOOK_PATH} ${OUTPUT_NOTEBOOK_PATH}"


gcloud compute instances create $INSTANCE_NAME \
        --zone=$ZONE \
        --image-family=$IMAGE_FAMILY \
        --image-project=deeplearning-platform-release \
        --maintenance-policy=TERMINATE \
        --accelerator=$ACCELERATOR \
        --machine-type=$INSTANCE_TYPE \
        --boot-disk-size=100GB \
        --scopes=https://www.googleapis.com/auth/cloud-platform \
        --metadata="install-nvidia-driver=True,startup-script=${STARTUP_SCRIPT}"

gcloud --quiet compute instances delete $INSTANCE_NAME --zone $ZONE