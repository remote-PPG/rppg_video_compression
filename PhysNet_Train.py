
# %%
from rppg_toolbox.src.dataset_reader.UBFC_rPPG import UBFCrPPGsDatasetReader
from rppg_toolbox.src.data_generator.PhysNet import PhysNetDataGenerator,PhysNetDataConfig
from rppg_toolbox.src.common.cache import CacheType
from rppg_toolbox.src.stepbystep.v4 import StepByStep
from torch import optim
from rppg_toolbox.src.model import PhysNet
from rppg_toolbox.src.loss import Neg_PearsonLoss

MODEL_SAVE_PATH = r"./out/model/PhysNet_Train_in_UBFC-rPPG.mdl"
TRAIN_CACHE = CacheType.NEW_CACHE
VAL_CACHE = CacheType.NEW_CACHE

dataset_path = r"/public/share/weiyuanwang/dataset/UBFC-rPPG"
train_cache_path = r"~/cache/PhysNet/UBFC-rPPG/train"
val_cache_path = r"~/cache/PhysNet/UBFC-rPPG/val"

WIDTH = 128
HEIGHT = 128
STEP = 70
T = 120
BATCH = 4
dataset = 2



model = PhysNet(T)
loss = Neg_PearsonLoss()
optimizer = optim.Adam(model.parameters(),lr=0.0001)
sbs_physnet = StepByStep(model,loss,optimizer)


if __name__ == '__main__':
    train_dataset_config = PhysNetDataConfig(
        cache_root= train_cache_path,
        cache_type= TRAIN_CACHE,
        generate_num_workers=12,
        step=STEP,
        width=WIDTH,
        height=HEIGHT,
        slice_interval=T,
        num_workers = 12,
        batch_size=BATCH,
        shuffle=True,
        load_to_memory=False
    )
    val_dataset_config = PhysNetDataConfig(
        cache_root=val_cache_path,
        cache_type=VAL_CACHE,
        generate_num_workers=12,
        step=STEP,
        width=WIDTH,
        height=HEIGHT,
        slice_interval=T,
        num_workers=12,
        batch_size=BATCH,
        load_to_memory=False
    )
    train_dataset_reader = UBFCrPPGsDatasetReader(dataset_path,dataset=dataset,dataset_list=[
        'subject10', 'subject11', 'subject12', 'subject13', 'subject14', 'subject15',
        'subject16', 'subject17', 'subject18', 'subject20', 'subject22', 'subject23', 'subject24',
        'subject25', 'subject26', 'subject27', 'subject30', 'subject31', 'subject32',
        'subject33', 'subject34', 'subject35', 'subject36', 'subject37', 'subject38', 'subject39',
        'subject40', 'subject41', 'subject42', 'subject43', 'subject44', 'subject45',
    ])
    val_dataset_reader = UBFCrPPGsDatasetReader(dataset_path,dataset=dataset,dataset_list=[
        'subject1','subject3','subject4','subject5','subject49', 
        'subject8', 'subject9', 'subject48','subject46', 'subject47', 
    ])
    train_dataset_generator = PhysNetDataGenerator(config=train_dataset_config)
    train_raw_data = train_dataset_reader.read() if TRAIN_CACHE == CacheType.NEW_CACHE else None
    train_dataloader = train_dataset_generator.generate(train_raw_data)

    val_dataset_generator = PhysNetDataGenerator(config=val_dataset_config)
    val_raw_data = val_dataset_reader.read() if VAL_CACHE == CacheType.NEW_CACHE else None
    val_dataloader = val_dataset_generator.generate(val_raw_data)

    sbs_physnet.set_loaders(train_dataloader,val_dataloader)
    sbs_physnet.set_tensorboard("physnet_train_in_UBFC-rPPG")
    try:
        sbs_physnet.train(10)
    except:
        pass
    finally:
        sbs_physnet.save_best_checkpoint(MODEL_SAVE_PATH)
else:
    try:
        sbs_physnet.load_checkpoint(MODEL_SAVE_PATH)
    except:
        raise Exception('No any checkpoint, run train first!')