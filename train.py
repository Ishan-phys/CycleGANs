import torch
from tqdm import tqdm
from torchvision.utils import save_image
import config

class train_func:
    
    def __init__(self, disc_A, disc_B, gen_AB, gen_BA, opt_disc, opt_gen, l1, mse):
        self.disc_A = disc_A 
        self.disc_B = disc_B
        self.gen_AB = gen_AB
        self.gen_BA = gen_BA
        self.opt_disc = opt_disc
        self.opt_gen = opt_gen
        self.l1 = l1
        self.mse = mse
    
    def train(self, loader):
        A_reals = 0
        A_fakes = 0
        loop = tqdm(loader, leave=True)

        for idx, (A, B) in enumerate(loop):
            A = A.to(config.DEVICE)
            B = B.to(config.DEVICE)
            
            # Train Discriminators A and B
            with torch.cuda.amp.autocast():
                
                fake_A = self.gen_BA(B)
                D_A_real = self.disc_A(A)
                D_A_fake = self.disc_A(fake_A.detach())
                A_reals += D_A_real.mean().item()
                A_fakes += D_A_fake.mean().item()
                D_A_real_loss = self.mse(D_A_real, torch.ones_like(D_A_real))
                D_A_fake_loss = self.mse(D_A_fake, torch.zeros_like(D_A_fake))
                D_A_loss = D_A_real_loss + D_A_fake_loss

                fake_B = self.gen_AB(A)
                D_B_real = self.disc_B(B)
                D_B_fake = self.disc_B(fake_B.detach())
                D_B_real_loss = self.mse(D_B_real, torch.ones_like(D_B_real))
                D_B_fake_loss = self.mse(D_B_fake, torch.zeros_like(D_B_fake))
                D_B_loss = D_B_real_loss + D_B_fake_loss

                D_loss = (D_A_loss + D_B_loss)/2

            self.opt_disc.zero_grad()
            D_loss.backward()
            self.opt_disc.step()

            # Train Generators A and B
            with torch.cuda.amp.autocast():

                # adversarial loss for both generators
                D_A_fake = self.disc_A(fake_A)
                D_B_fake = self.disc_B(fake_B)
                loss_G_A = self.mse(D_A_fake, torch.ones_like(D_A_fake))
                loss_G_B = self.mse(D_B_fake, torch.ones_like(D_B_fake))

                loss_gan = (loss_G_A + loss_G_B) / 2

                # cycle loss
                cycle_B = self.gen_AB(fake_A)
                cycle_A = self.gen_BA(fake_B)
                cycle_B_loss = self.l1(cycle_B, B)
                cycle_A_loss = self.l1(cycle_A, A)

                loss_cycle = (cycle_A_loss + cycle_B_loss) / 2

                # identity loss 
                identity_B = self.gen_AB(B)
                identity_A = self.gen_BA(A)
                identity_B_loss = self.l1(identity_B, B)
                identity_A_loss = self.l1(identity_A, A)
                
                loss_identity = (identity_A_loss + identity_B_loss) / 2

                # add all together
                G_loss = loss_gan + loss_cycle * config.LAMBDA_CYCLE + loss_identity * config.LAMBDA_IDENTITY
                
            self.opt_gen.zero_grad()
            G_loss.backward()
            self.opt_gen.step()

            save_image(fake_A*0.5+0.5, f"./saved_images/A_{idx}.png")
            save_image(fake_B*0.5+0.5, f"./saved_images/B_{idx}.png")
