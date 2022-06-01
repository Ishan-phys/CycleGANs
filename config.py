import torch 
from torchvision.transforms import Compose, ToTensor, Normalize, Resize

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_DIR = "data/train"
TEST_DIR = "data/test"
BATCH_SIZE = 1
LEARNING_RATE = 0.0002
LAMBDA_IDENTITY = 5.0
LAMBDA_CYCLE = 10.0
NUM_EPOCHS = 200
RES_BLOCKS = 19
INPUT_SHAPE = (3, 256, 256)

transforms = Compose(
    [Resize((256, 256)),
     ToTensor(),
     Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)