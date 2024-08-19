import time
import torch
import os
import json
from torch.utils.data import Dataset

num_gpus = 8
is_xla = True

if is_xla:
  import torch_xla.core.xla_model as xm
  import torch_xla.distributed.parallel_loader as pl
  if os.environ.get('XRT_WORKERS') is None:
    # Add ENVARS required for XLA
    host=json.loads(os.environ["SM_HOSTS"])[0]
    os.environ["XRT_WORKERS"] = f'localservice:0;{host}:43857'
    os.environ['XRT_SHARD_WORLD_SIZE'] = '1'
    os.environ['XRT_HOST_ORDINAL'] = '0'
    os.environ['FI_EFA_USE_DEVICE_RDMA'] = '1'
    os.environ['NCCL_PROTO'] =  'simple'
    os.environ['XLA_FIX_DIV_FP64'] = '1'
    os.environ['OFI_NCCL_NIC_DUP_CONNS'] = str(num_gpus)
    os.environ["GPU_NUM_DEVICES"] = str(num_gpus)
else:
  # DDP setup
  import torch.distributed as dist
  def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR',
                                               'localhost')
    os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT',
                                               str(2222))
    dist.init_process_group('nccl', rank=rank,
                            world_size=world_size)
  # wrap the model with DDP
  def wrap_model(model,local_rank):
    from torch.nn.parallel import DistributedDataParallel as DDP
    model.to(torch.cuda.current_device())
    model = DDP(model,device_ids=[local_rank])
    return model
  
  # A fake dataset
class FakeDataset(Dataset):
  def __len__(self):
    return 10000000
  def __getitem__(self, index):
    rand_image = torch.randn([3, 224, 224], dtype=torch.float32)
    label = torch.tensor(data=[index % 1000], dtype=torch.int64)
    return rand_image, label
  
  def build_model():
    from transformers import ViTForImageClassification, ViTConfig
    return ViTForImageClassification(ViTConfig(num_labels=1000))
 
  def main(rank, world_size=num_gpus):
  dataset = FakeDataset()
  model = build_model()

  if is_xla:
    device = xm.xla_device()
    rank = xm.get_local_ordinal()
    model = model.to(device)
  else:
    setup(rank, world_size)
    torch.cuda.set_device(rank)
    model = wrap_model(model,rank)
  batch_size = 128
  optimizer = torch.optim.Adam(model.parameters())
  data_loader = torch.utils.data.DataLoader(
                                  dataset,
                                  batch_size=batch_size,
   
                                  num_workers=12)
  if is_xla:
    data_loader = pl.MpDeviceLoader(data_loader, device)  loss_function = torch.nn.CrossEntropyLoss()
  t0 = time.perf_counter()
  for idx, (inputs, targets) in enumerate(data_loader, start=1):
    if not is_xla:
      inputs = inputs.to(torch.cuda.current_device())
      targets = targets.to(torch.cuda.current_device())
    targets = torch.squeeze(targets,-1)
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = loss_function(outputs['logits'], targets)
    loss.backward()
    if is_xla:
      xm.optimizer_step(optimizer)
    else:
      optimizer.step()
    if rank == 0 and idx%1000 == 0:
      batch_time = time.perf_counter() - t0
      print(f'step: {idx}: mean step time is {batch_time/1000}')
      t0 = time.perf_counter()  if not is_xla:
    dist.destroy_process_group()

def _mp_fn(index):
  main(index)

if __name__ == "__main__":
  if is_xla:
    import torch_xla.distributed.xla_multiprocessing as mp
  else:
    import torch.multiprocessing as mp
  mp.spawn(_mp_fn, nprocs=num_gpus, join=True)

  image_uri = "763104351884.dkr.ecr.us-east-1.amazonaws.com/" \
            "huggingface-pytorch-trcomp-training:1.11.0-" \
            "transformers4.21.1-gpu-py38-cu113-ubuntu20.04"

from sagemaker.pytorch import PyTorch
estimator = PyTorch(entry_point='train.py',
                    role=<role>,
                    instance_type='ml.p4d.24xlarge',
                    instance_count=1,
                    image_uri=image_uri)
estimator.fit()