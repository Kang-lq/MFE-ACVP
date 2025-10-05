import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from Bio import SeqIO
import matplotlib.pyplot as plt
from collections import defaultdict
import math
import argparse


class Config:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Paths
    train_data_path = "./data/train.fasta"
    output_dir = "./output"

    # Model parameters
    noise_dim = 256
    max_len = 60
    batch_size = 64
    epochs = 1000
    lr_G = 8e-6
    lr_D = 8e-6

    # Network architecture
    gen_hidden_dims = [512, 1024]
    disc_hidden_dims = [512, 256]

    # Property weights
    property_weights = {
        'charge_mean': -0.917, 'charge_std': 3.842,
        'gravy_mean': -0.326, 'gravy_std': 0.566,
        'aromatic_mean': 0.096, 'aromatic_std': 0.071,
    }


class AminoAcid:
    aa_list = 'ACDEFGHIKLMNPQRSTVWY'
    aa_to_idx = {aa: i for i, aa in enumerate(aa_list)}
    idx_to_aa = {i: aa for i, aa in enumerate(aa_list)}

    charge_table = {'D': -1.2, 'E': -1.1, 'K': 0.9, 'R': 0.8, 'H': 0.1}
    gravy_table = {
        'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
        'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
        'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
        'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2
    }
    aromatic_aas = {'F', 'Y', 'W'}

    @staticmethod
    def calc_physchem(seq):
        charge = sum(AminoAcid.charge_table.get(aa, 0) for aa in seq)
        gravy = sum(AminoAcid.gravy_table.get(aa, 0) for aa in seq) / len(seq)
        aromatic = sum(aa in AminoAcid.aromatic_aas for aa in seq) / len(seq)
        return charge, gravy, aromatic

    @staticmethod
    def seq_from_logits(logits, length, temp=0.7):
        aa_probs = F.softmax(logits[:length] / temp, dim=-1)
        seq = []
        for p in aa_probs:
            idx = torch.multinomial(p, 1).item()
            seq.append(AminoAcid.idx_to_aa.get(idx, 'A'))
        return ''.join(seq)


class ACVPDataset(Dataset):
    def __init__(self, fasta_path):
        self.sequences = []
        self.properties = defaultdict(list)

        for record in SeqIO.parse(fasta_path, "fasta"):
            seq = str(record.seq).upper()
            if all(aa in AminoAcid.aa_to_idx for aa in seq):
                charge, gravy, aromatic = AminoAcid.calc_physchem(seq)
                if 15 <= len(seq) <= 60:
                    self.sequences.append(seq)
                    self.properties['charge'].append(charge)
                    self.properties['gravy'].append(gravy)
                    self.properties['aromatic'].append(aromatic)

        self.stats = {
            'charge_mean': np.mean(self.properties['charge']),
            'charge_std': np.std(self.properties['charge']),
            'gravy_mean': np.mean(self.properties['gravy']),
            'gravy_std': np.std(self.properties['gravy']),
            'aromatic_mean': np.mean(self.properties['aromatic']),
            'aromatic_std': np.std(self.properties['aromatic']),
        }
        np.random.shuffle(self.sequences)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        encoded = np.zeros((Config.max_len, len(AminoAcid.aa_list)))
        for i, aa in enumerate(seq[:Config.max_len]):
            encoded[i, AminoAcid.aa_to_idx[aa]] = 1
        return torch.FloatTensor(encoded)


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_len = nn.Embedding(Config.max_len - 14, 32)

        self.fc1 = nn.Linear(Config.noise_dim + 32, Config.gen_hidden_dims[0])
        self.bn1 = nn.BatchNorm1d(Config.gen_hidden_dims[0])

        self.fc2 = nn.Linear(Config.gen_hidden_dims[0], Config.gen_hidden_dims[1])
        self.bn2 = nn.BatchNorm1d(Config.gen_hidden_dims[1])

        self.seq_head = nn.Sequential(
            nn.Linear(Config.gen_hidden_dims[1], Config.max_len * len(AminoAcid.aa_list)),
            nn.Tanh()
        )

        self.charge_head = nn.Sequential(
            nn.Linear(Config.gen_hidden_dims[1], 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1)
        )

        self.hydro_head = nn.Sequential(
            nn.Linear(Config.gen_hidden_dims[1], 128),
            nn.Tanh()
        )

    def forward(self, z, lengths):
        len_emb = self.embed_len(lengths - 15)
        x = torch.cat([z, len_emb], dim=1)

        h1 = self.fc1(x)
        if h1.size(0) > 1:
            h1 = self.bn1(h1)
        h1 = F.leaky_relu(h1, 0.2)

        h2 = self.fc2(h1)
        if h2.size(0) > 1:
            h2 = self.bn2(h2)
        h2 = F.leaky_relu(h2, 0.2)

        seq_output = self.seq_head(h2).view(-1, Config.max_len, len(AminoAcid.aa_list))
        charge = self.charge_head(h2)
        hydro = self.hydro_head(h2)

        return seq_output, torch.cat([charge, hydro], dim=1)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(Config.max_len * len(AminoAcid.aa_list), Config.disc_hidden_dims[0]),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(Config.disc_hidden_dims[0], Config.disc_hidden_dims[1]),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(Config.disc_hidden_dims[1], 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x.view(x.size(0), -1))


class Trainer:
    def __init__(self, data_path):
        self.device = Config.device
        self.dataset = ACVPDataset(data_path)
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=Config.batch_size,
            shuffle=True,
            drop_last=True
        )

        self.G = Generator().to(self.device)
        self.D = Discriminator().to(self.device)

        self.optim_G = optim.Adam(self.G.parameters(), lr=Config.lr_G, betas=(0.5, 0.999))
        self.optim_D = optim.Adam(self.D.parameters(), lr=Config.lr_D, betas=(0.5, 0.999))

        self.history = defaultdict(list)
        self.current_epoch = 0

    def _calc_gradient_penalty(self, real_data, fake_data):
        alpha = torch.rand(real_data.size(0), 1, 1, device=self.device)
        interpolates = alpha * real_data + (1 - alpha) * fake_data
        interpolates.requires_grad_(True)

        d_interpolates = self.D(interpolates)
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(d_interpolates),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        return ((gradients.norm(2, dim=1) - 1) ** 2).mean()

    def _property_loss(self, pred_properties, gen_lengths, seq_logits):
        charge_target = torch.randn_like(pred_properties[:, 0]) * Config.property_weights['charge_std'] + \
                        Config.property_weights['charge_mean']
        charge_loss = F.smooth_l1_loss(pred_properties[:, 0], charge_target)

        gravy_target = torch.randn_like(pred_properties[:, 1]) * Config.property_weights['gravy_std'] + \
                       Config.property_weights['gravy_mean']
        gravy_loss = F.smooth_l1_loss(pred_properties[:, 1], gravy_target)

        seqs = [AminoAcid.seq_from_logits(s, l.item())
                for s, l in zip(seq_logits, gen_lengths)]
        pos_ratio = torch.tensor([
            sum(AminoAcid.charge_table.get(aa, 0) > 0 for aa in seq) / len(seq)
            for seq in seqs
        ], device=self.device)
        neg_ratio = torch.tensor([
            sum(AminoAcid.charge_table.get(aa, 0) < 0 for aa in seq) / len(seq)
            for seq in seqs
        ], device=self.device)

        comp_loss = F.mse_loss(pos_ratio, torch.full_like(pos_ratio, 0.15)) + \
                    F.mse_loss(neg_ratio, torch.full_like(neg_ratio, 0.15))

        return {
            'charge': charge_loss,
            'gravy': gravy_loss,
            'composition': comp_loss,
            'pos_ratio': torch.mean(pos_ratio),
            'neg_ratio': torch.mean(neg_ratio)
        }

    def train(self):
        print(f"Training data statistics: {self.dataset.stats}")
        print(f"Starting training for {Config.epochs} epochs...")

        self.G.train()
        self.D.train()

        for epoch in range(Config.epochs):
            self.current_epoch = epoch
            for real_seqs in self.dataloader:
                real_seqs = real_seqs.to(self.device)
                batch_size = real_seqs.size(0)

                # Train Discriminator
                self.optim_D.zero_grad()

                z = torch.randn(batch_size, Config.noise_dim, device=self.device)
                gen_lengths = torch.randint(15, 61, (batch_size,), device=self.device)
                with torch.no_grad():
                    fake_seqs, _ = self.G(z, gen_lengths)

                d_real = self.D(real_seqs)
                d_fake = self.D(fake_seqs.detach())
                gp = self._calc_gradient_penalty(real_seqs, fake_seqs)

                d_real_loss = torch.mean((d_real - 1) ** 2)
                d_fake_loss = torch.mean(d_fake ** 2)
                d_loss = 0.5 * (d_real_loss + d_fake_loss) + gp

                d_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.D.parameters(), 1.0)
                self.optim_D.step()

                # Train Generator
                if epoch % 3 == 0:
                    self.optim_G.zero_grad()

                    z = torch.randn(batch_size, Config.noise_dim, device=self.device)
                    gen_lengths = torch.randint(15, 61, (batch_size,), device=self.device)
                    fake_seqs, pred_properties = self.G(z, gen_lengths)

                    d_fake = self.D(fake_seqs)
                    adv_loss = -torch.mean(torch.log(d_fake + 1e-8))

                    prop_losses = self._property_loss(pred_properties, gen_lengths, fake_seqs)

                    g_loss = adv_loss + prop_losses['charge'] + prop_losses['gravy'] + prop_losses['composition']

                    g_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.G.parameters(), 0.5)
                    self.optim_G.step()

            # Logging and saving
            if epoch % 100 == 0:
                self._log_progress(epoch, d_loss, g_loss, prop_losses)

            if epoch % 500 == 0:
                self._save_checkpoint(epoch)
                self._sample_sequences(epoch)

    def _log_progress(self, epoch, d_loss, g_loss, prop_losses):
        self.history['d_loss'].append(d_loss.item())
        self.history['g_loss'].append(g_loss.item())

        log_str = (
            f"Epoch {epoch}: D_loss={d_loss.item():.3f}, G_loss={g_loss.item():.3f} | "
            f"Charge={prop_losses['charge'].item():.3f}, Gravy={prop_losses['gravy'].item():.3f} | "
            f"Pos/Neg={prop_losses['pos_ratio'].item():.2%}/{prop_losses['neg_ratio'].item():.2%}"
        )
        print(log_str)

        # Save best model
        if g_loss.item() < getattr(self, 'best_g_loss', float('inf')):
            self.best_g_loss = g_loss.item()
            torch.save({
                'G_state_dict': self.G.state_dict(),
                'D_state_dict': self.D.state_dict(),
                'epoch': epoch,
                'loss': g_loss.item()
            }, os.path.join(Config.output_dir, "best_model.pth"))

    def _sample_sequences(self, epoch):
        print(f"\n=== Generated samples at epoch {epoch} ===")
        self.G.eval()

        with torch.no_grad():
            z = torch.randn(3, Config.noise_dim, device=self.device)
            lengths = torch.tensor([20, 35, 50], device=self.device)
            seq_logits, _ = self.G(z, lengths)

            for i, (seq_logit, length) in enumerate(zip(seq_logits, lengths)):
                seq = AminoAcid.seq_from_logits(seq_logit, length.item())
                charge, gravy, aromatic = AminoAcid.calc_physchem(seq)
                print(f"Sample {i + 1} (L={length.item()}): {seq}")
                print(f"  Properties: charge={charge:.2f}, gravy={gravy:.2f}, aromatic={aromatic:.2%}")

        self.G.train()

    def _save_checkpoint(self, epoch):
        state = {
            'G_state': self.G.state_dict(),
            'D_state': self.D.state_dict(),
            'optim_G': self.optim_G.state_dict(),
            'optim_D': self.optim_D.state_dict(),
            'epoch': epoch,
            'history': dict(self.history)
        }
        torch.save(state, os.path.join(Config.output_dir, f"checkpoint_{epoch}.pth"))

        # Plot training curves
        if len(self.history['d_loss']) > 1:
            plt.figure(figsize=(12, 4))
            plt.subplot(1, 2, 1)
            plt.plot(self.history['d_loss'], label='Discriminator Loss')
            plt.plot(self.history['g_loss'], label='Generator Loss')
            plt.legend()
            plt.title('Training Loss')
            plt.savefig(os.path.join(Config.output_dir, 'training_loss.png'))
            plt.close()


def main():
    parser = argparse.ArgumentParser(description='Train MaliGANNEXT model')
    parser.add_argument('--data_path', type=str, default=Config.train_data_path)
    parser.add_argument('--output_dir', type=str, default=Config.output_dir)
    parser.add_argument('--epochs', type=int, default=Config.epochs)
    parser.add_argument('--batch_size', type=int, default=Config.batch_size)

    args = parser.parse_args()

    # Update config
    Config.train_data_path = args.data_path
    Config.output_dir = args.output_dir
    Config.epochs = args.epochs
    Config.batch_size = args.batch_size

    os.makedirs(Config.output_dir, exist_ok=True)

    # Set random seeds
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    trainer = Trainer(Config.train_data_path)
    trainer.train()


if __name__ == "__main__":
    main()