import os
import time
import tqdm

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.tensorboard import SummaryWriter

from replic.modules.discriminator import Discriminator
from replic.modules.translator import Translator
from replic.utils.train_utils import vsp_loss, cycle_consistency_loss, reconstruction_loss
from replic.utils.data_utils import create_dataloaders
from replic.utils.config import TrainingConfig

def train(config: "TrainingConfig"):
    """
    Main training function for Vec2Vec.
    
    - Correct separate GAN training steps for generator and discriminators.
    - Automatic Mixed Precision (AMP) for performance.
    - Model checkpointing to save progress.
    - TensorBoard logging for experiment tracking.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(config.seed)

    # Setup for logging and saving models
    run_name = f"vec2vec_run_{int(time.time())}"
    os.makedirs(run_name, exist_ok=True)
    writer = SummaryWriter(log_dir=run_name)

    sup_loader, unsup_loader = create_dataloaders(
        config,
        num_workers=4,
        pin_memory=True
    )

    translator = Translator(
        sup_dim=config.sup_emb_dim,
        unsup_dim=config.unsup_emb_dim,
        latent_dim=max(config.sup_emb_dim, config.unsup_emb_dim),
        hidden_dim=config.translator_hidden_dim,
        depth=config.translator_depth
    ).to(device)

    sup_discriminator = Discriminator(
        latent_dim=config.sup_emb_dim,
        disc_dim=config.disc_dim,
        depth=config.discriminator_depth
    ).to(device)

    unsup_discriminator = Discriminator(
        latent_dim=config.unsup_emb_dim,
        disc_dim=config.disc_dim,
        depth=config.discriminator_depth
    ).to(device)
    
    translator = torch.compile(translator)
    sup_discriminator = torch.compile(sup_discriminator)
    unsup_discriminator = torch.compile(unsup_discriminator)

    # Optimizers with fused=True for better performance on CUDA
    use_fused = device
    translator_opt = optim.Adam(translator.parameters(), lr=config.lr, fused=use_fused)
    sup_disc_opt = optim.Adam(sup_discriminator.parameters(), lr=config.disc_lr, fused=use_fused)
    unsup_disc_opt = optim.Adam(unsup_discriminator.parameters(), lr=config.disc_lr, fused=use_fused)

    scaler = torch.cuda.amp.GradScaler(enabled=(device))
    
    print(f"🚀 Starting training for {config.epochs} epochs on {device.upper()}...")
    global_step = 0
    num_batches = min(len(sup_loader), len(unsup_loader))

    for epoch in range(config.epochs):
        epoch_losses = {k: [] for k in ['rec', 'vsp', 'cc', 'disc', 'gen']}
        
        pbar = tqdm(zip(sup_loader, unsup_loader), total=num_batches, desc=f"Epoch {epoch+1}/{config.epochs}")
        for sup_batch, unsup_batch in pbar:
            sup_emb = sup_batch[0].to(device)
            unsup_emb = unsup_batch[0].to(device)

            for param in sup_discriminator.parameters(): param.requires_grad = True
            for param in unsup_discriminator.parameters(): param.requires_grad = True
            
            sup_disc_opt.zero_grad()
            unsup_disc_opt.zero_grad()

            with torch.cuda.amp.autocast(enabled=(device)):
                fake_sup = translator.translate_unsup_to_sup(unsup_emb).detach()
                fake_unsup = translator.translate_sup_to_unsup(sup_emb).detach()

                d_loss_sup = (F.binary_cross_entropy_with_logits(sup_discriminator(sup_emb), torch.ones_like(sup_discriminator(sup_emb))) +
                              F.binary_cross_entropy_with_logits(sup_discriminator(fake_sup), torch.zeros_like(sup_discriminator(fake_sup))))

                d_loss_unsup = (F.binary_cross_entropy_with_logits(unsup_discriminator(unsup_emb), torch.ones_like(unsup_discriminator(unsup_emb))) +
                                F.binary_cross_entropy_with_logits(unsup_discriminator(fake_unsup), torch.zeros_like(unsup_discriminator(fake_unsup))))
            
            scaler.scale(d_loss_sup).backward()
            scaler.step(sup_disc_opt)

            scaler.scale(d_loss_unsup).backward()
            scaler.step(unsup_disc_opt)
            
            disc_loss = (d_loss_sup + d_loss_unsup) / 2
            epoch_losses['disc'].append(disc_loss.item())


            for param in sup_discriminator.parameters(): param.requires_grad = False
            for param in unsup_discriminator.parameters(): param.requires_grad = False

            translator_opt.zero_grad()
            
            with torch.cuda.amp.autocast(enabled=(device)):
                sup_to_unsup = translator.translate_sup_to_unsup(sup_emb)
                unsup_to_sup = translator.translate_unsup_to_sup(unsup_emb)

                sup_recon = translator.reconstruct_sup(sup_emb)
                unsup_recon = translator.reconstruct_unsup(unsup_emb)
                sup_cycle = translator.translate_unsup_to_sup(sup_to_unsup)
                unsup_cycle = translator.translate_sup_to_unsup(unsup_to_sup)
                
                rec_loss = (reconstruction_loss(sup_emb, sup_recon) + reconstruction_loss(unsup_emb, unsup_recon)) / 2
                vsp_loss_val = (vsp_loss(sup_emb, sup_to_unsup) + vsp_loss(unsup_emb, unsup_to_sup)) / 2
                cc_loss = (cycle_consistency_loss(sup_emb, unsup_cycle) + cycle_consistency_loss(unsup_emb, sup_cycle)) / 2

                gen_loss_sup = F.binary_cross_entropy_with_logits(sup_discriminator(unsup_to_sup), torch.ones_like(sup_discriminator(unsup_to_sup)))
                gen_loss_unsup = F.binary_cross_entropy_with_logits(unsup_discriminator(sup_to_unsup), torch.ones_like(unsup_discriminator(sup_to_unsup)))
                gen_loss = (gen_loss_sup + gen_loss_unsup) / 2
                
                total_gen_loss = (config.loss_coefficient_rec * rec_loss +
                                  config.loss_coefficient_vsp * vsp_loss_val +
                                  config.loss_coefficient_cc * cc_loss +
                                  config.loss_coefficient_gen * gen_loss)

            scaler.scale(total_gen_loss).backward()
            scaler.step(translator_opt)
            scaler.update()

            epoch_losses['rec'].append(rec_loss.item())
            epoch_losses['vsp'].append(vsp_loss_val.item())
            epoch_losses['cc'].append(cc_loss.item())
            epoch_losses['gen'].append(gen_loss.item())

            pbar.set_postfix({
                'D_loss': disc_loss.item(),
                'G_loss': total_gen_loss.item()
            })
            writer.add_scalar('Loss/Discriminator_Step', disc_loss.item(), global_step)
            writer.add_scalar('Loss/Generator_Step', total_gen_loss.item(), global_step)
            global_step += 1

        for name, values in epoch_losses.items():
            writer.add_scalar(f'Loss_Epoch/{name}', np.mean(values), epoch)
            
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(run_name, f"checkpoint_epoch_{epoch+1}.pt")
            torch.save({
                'epoch': epoch,
                'translator_state_dict': translator.state_dict(),
                'sup_disc_state_dict': sup_discriminator.state_dict(),
                'unsup_disc_state_dict': unsup_discriminator.state_dict(),
                'translator_opt_state_dict': translator_opt.state_dict(),
                'sup_disc_opt_state_dict': sup_disc_opt.state_dict(),
                'unsup_disc_opt_state_dict': unsup_disc_opt.state_dict(),
            }, checkpoint_path)
            print(f"\n✅ Checkpoint saved to {checkpoint_path}")

    writer.close()
    return translator