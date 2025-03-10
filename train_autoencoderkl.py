from generative.losses import PatchAdversarialLoss
from torch.nn import L1Loss
from tqdm import tqdm
from utils import *
from models.AutoEncoder import autoencoder, patchdiscriminator



def train_aekl(args, all_real_loader):
    print("**************************Autoencoder*****************************")

    aekl_model = autoencoder
    discriminator = patchdiscriminator

    aekl_model = aekl_model.to(args.device)
    discriminator = discriminator.to(args.device)

    optimizer_g = torch.optim.Adam(params=aekl_model.parameters(),
                                   lr=args.optimizer_g_lr)
    optimizer_d = torch.optim.Adam(params=discriminator.parameters(),
                                   lr=args.optimizer_d_lr)

    l1_loss = L1Loss()
    adv_loss = PatchAdversarialLoss(criterion="least_squares")
    adv_weight = args.adv_weight
    kl_weight = args.kl_weight

    start_epoch = 0

    for epoch in range(start_epoch, args.aekl_num_epoch):
        aekl_model.train()
        discriminator.train()
        total_g_train_loss = 0
        epoch_loss = 0
        gen_epoch_loss = 0
        disc_epoch_loss = 0
        kl_epoch_loss = 0
        progress_bar = tqdm(enumerate(all_real_loader), total=len(all_real_loader), ncols=150)
        progress_bar.set_description(f"Epoch {epoch}")
        num = 0
        for step, batch_real in progress_bar:
            num += 1
            eeg_data = batch_real[0].to(args.device)

            optimizer_g.zero_grad(set_to_none=True)
            reconstruction, z_mu, z_sigma = aekl_model(eeg_data)

            recons_loss = l1_loss(reconstruction.float(), eeg_data.float())

            kl_loss = KL_loss(z_mu, z_sigma)

            logits_fake = discriminator(reconstruction.contiguous().float())[-1]
            generator_loss = adv_loss(logits_fake, target_is_real=True, for_discriminator=False)

            loss_g = recons_loss + kl_weight * kl_loss + adv_weight * generator_loss
            loss_g.backward()
            optimizer_g.step()
            total_g_train_loss += loss_g.item()

            optimizer_d.zero_grad(set_to_none=True)

            logits_fake = discriminator(reconstruction.contiguous().detach())[-1]
            loss_d_fake = adv_loss(logits_fake, target_is_real=False, for_discriminator=True)
            logits_real = discriminator(eeg_data.contiguous().detach())[-1]
            loss_d_real = adv_loss(logits_real, target_is_real=True, for_discriminator=True)
            discriminator_loss = (loss_d_fake + loss_d_real) * 0.5

            loss_d = adv_weight * discriminator_loss

            loss_d.backward()
            optimizer_d.step()

            epoch_loss += recons_loss.item()
            gen_epoch_loss += generator_loss.item()
            disc_epoch_loss += discriminator_loss.item()
            kl_epoch_loss += kl_loss.item()

            progress_bar.set_postfix(
                {
                    "l1_loss": epoch_loss / (step + 1),
                    "gen_loss": gen_epoch_loss / (step + 1),
                    "disc_loss": disc_epoch_loss / (step + 1),
                    "kl_loss": kl_epoch_loss / (step + 1)

                }
            )
