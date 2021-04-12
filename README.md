# file_transport
import torch
from utils.util import *
from settings.configs import *
from datas.data_utils import *
from models.resnet import *
from tests.validatoion import *
from trains import train_Baseline,train_L2RW,train_MW_Net,\
    train_MCSWN_01a,train_MCSWN_02a,train_MCSWN_03a,train_MCSWN_04a
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# get shell arg
args=parse_arguments()
# fixed seed
set_seed(args.seed)
# device detect
device=set_device(args)
# load config
config=load_config()

# get imbalance ratio
sample_probability = get_imbalance_ratios(imb_factor=args.imb_factor,cls_num=args.num_classes)
# create  inverse_imbalance_sampler
inverse_imbalance_sampler = torch.utils.data.WeightedRandomSampler(
    weights=sample_probability.sort(reverse = True),num_samples=args.,repalcement=True)

# Dataloader set
kwargs = {'num_workers': config['Dataloader_set']['num_workers'],
          'pin_memory': config['Dataloader_set']['pin_memory']}
# get dataset
train_data_meta,train_data,test_dataset = build_dataset(args.seed,args.dataset,args.num_meta)
# init weights list
weights=torch.zeros(len(train_data_meta),dtype=torch.long)

# get imbalance ratio
sample_probability = get_imbalance_ratios(imb_factor=args.imb_factor,cls_num=args.num_classes)
# get reverse
sample_probability.sort(reverse=False)

# make imbalanced dataset
# give sample weight
for index,(data,target) in enumerate(train_data_meta):
    weights[index]=sample_probability[target.item()]

# create  inverse_imbalance_sampler
inverse_imbalance_sampler = torch.utils.data.WeightedRandomSampler(
    weights=weights,num_samples=len(train_data_meta),replacement=True)

# ake imbalanced dataset
imbalanced_train_dataset=get_imbalance_dataset(train_data,args)
