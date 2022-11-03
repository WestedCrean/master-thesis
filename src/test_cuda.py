import torch
import tensorflow as tf
from loguru import logger

logger.info("CHECKING FOR PYTORCH")
if torch.cuda.is_available():
    logger.info("There are %d GPU(s) available." % torch.cuda.device_count())
    logger.info("We will use the GPU:", torch.cuda.get_device_name(0))

else:
    logger.info("No GPU available, using the CPU instead.")

logger.info("\n\nCHECKING FOR TENSORFLOW")
logger.info(tf.config.list_physical_devices("GPU"))
