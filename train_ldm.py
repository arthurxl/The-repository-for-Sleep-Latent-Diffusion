import torch.nn as nn
from monai.utils import first
from tqdm import tqdm
from utils import *
from models.AutoEncoder import autoencoder
from models.unet import build_network
from models.DDPM import DDPM
from torch.cuda.amp import autocast


class Stage1Wrapper(nn.Module):

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z_mu, z_sigma = self.model.encode(x)
        z = self.model.sampling(z_mu, z_sigma)
        return z


def train_LDM(args, all_real_loader):
    print("**************************LDM*****************************")

    state_dict = torch.load(args.ae_model_out_dir + "/final_model.pth",
                            map_location=torch.device('cuda'))

    stage1 = autoencoder
    stage1.load_state_dict(state_dict)
    stage1.to(args.device)

    with torch.no_grad():
        with autocast(enabled=True):
            check_data = first(all_real_loader)[0].to(args.device)
            z = stage1.encode_stage_2_inputs(check_data)
    print(f"Scaling factor set to {1 / torch.std(z)}")
    scale_factor = 1 / torch.std(z)

    autoencoderkl = Stage1Wrapper(model=stage1)

    autoencoderkl.eval()
    autoencoderkl = autoencoderkl.to(args.device)
    ddpm = init_DDPM(args)
    net = init_Net(args)
    n_steps = ddpm.n_steps
    loss_fn = torch.nn.MSELoss()

    optimizer = torch.optim.Adam(net.parameters(), lr=args.ldm_lr, betas=(args.beta1, 0.999))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[0.8 * args.eval_freq],
                                                     gamma=args.gamma)

    start_epoch = 0
    print(f"Starting Training")

    for epoch in range(start_epoch, args.ldm_num_epoch):
        trained_ddpm, net, train_loss = train_single_item(args, epoch, autoencoderkl, ddpm, n_steps, net,
                                                          all_real_loader, loss_fn,
                                                          optimizer, scale_factor)
        scheduler.step()


def train_single_item(args, epoch, stage1, ddpm, n_steps, net, train_loader, loss_fn, optimizer, scale_factor):
    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), ncols=100)
    progress_bar.set_description(f"Epoch {epoch}")
    net.train()
    total_loss = 0
    num = 0
    for idx, batch_real in progress_bar:
        batch_real = batch_real[0]
        current_batch_size = batch_real.shape[0]
        num += 1
        batch_real = batch_real.to(args.device)
        with torch.no_grad():
            e = stage1(batch_real) * scale_factor
        t = torch.randint(0, n_steps, (current_batch_size,)).to(args.device)
        eps = torch.randn_like(e).to(args.device)
        x_t = ddpm.sample_forward(e, t, eps)
        eps_theta = net(x_t, t.reshape(current_batch_size, 1))
        snr = ddpm.compute_snr(t)
        mse_loss_weights = (
                torch.stack([snr, args.snr_gamma * torch.ones_like(t)], dim=1).min(dim=1)[0] / snr
        )
        loss = loss_fn(eps_theta.float(), eps.float(), reduction="none")
        loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
        loss = loss.mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        progress_bar.set_postfix({"epoch": epoch, "loss": f"{loss.item():.5f}"})

    return ddpm, net, total_loss / num


def init_DDPM(opt):
    ddpm = DDPM(opt.device, opt.n_steps)
    return ddpm


def init_Net(opt):
    net = build_network(opt.n_steps)
    net = net.to(opt.device)
    net.apply(weights_init)
    return net
