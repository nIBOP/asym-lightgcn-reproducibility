import torch
from recbole.trainer import Trainer
from torch.nn.utils import clip_grad_norm_


class CustomTrainer(Trainer):
    def _cfg(self, key, default=None):
        try:
            return self.config[key]
        except Exception:
            return default

    def _train_epoch(self, train_data, epoch_idx, loss_func=None, show_progress=False):
        self.model.train()
        loss_func = loss_func or self.model.calculate_loss

        total_loss_tensor = torch.tensor(0.0, device=self.model.device)
        total_bpr_tensor = torch.tensor(0.0, device=self.model.device)
        total_sem_tensor = torch.tensor(0.0, device=self.model.device)

        batch_interval = int(self._cfg("batch_log_interval", 50) or 0)
        log_epoch_summary = bool(self._cfg("log_epoch_summary", True))
        total_batches = len(train_data)

        for batch_idx, interaction in enumerate(train_data):
            if hasattr(train_data, "pr_end") and batch_idx == total_batches - 1:
                train_data.pr_end()

            interaction = interaction.to(self.device)
            self.optimizer.zero_grad()
            losses = loss_func(interaction)

            if isinstance(losses, tuple):
                loss = losses[0]
                total_loss_tensor = total_loss_tensor + losses[0].detach().squeeze()
                total_bpr_tensor = total_bpr_tensor + losses[1].detach().squeeze()
                total_sem_tensor = total_sem_tensor + losses[2].detach().squeeze()
            else:
                loss = losses
                total_loss_tensor = total_loss_tensor + losses.detach().squeeze()
                total_bpr_tensor = total_bpr_tensor + losses.detach().squeeze()

            self._check_nan(loss)
            loss.backward()

            if self.clip_grad_norm:
                clip_grad_norm_(self.model.parameters(), **self.clip_grad_norm)
            self.optimizer.step()

            if batch_interval > 0 and batch_idx > 0 and batch_idx % batch_interval == 0:
                if isinstance(losses, tuple):
                    print(
                        f"   [Epoch {epoch_idx}] batch {batch_idx}/{total_batches} | "
                        f"Loss: {losses[0].item():.4f} (BPR: {losses[1].item():.4f}, Sem: {losses[2].item():.4f})"
                    )
                else:
                    print(f"   [Epoch {epoch_idx}] batch {batch_idx}/{total_batches} | Loss: {loss.item():.4f}")

        final_loss = total_loss_tensor.item()
        final_bpr = total_bpr_tensor.item()
        final_sem = total_sem_tensor.item()

        if isinstance(losses, tuple):
            if log_epoch_summary:
                print(
                    f"[Epoch {epoch_idx}] finished | total_loss={final_loss:.4f} "
                    f"(BPR: {final_bpr:.4f}, Sem: {final_sem:.4f})"
                )
            return (final_loss, final_bpr, final_sem)

        if log_epoch_summary:
            print(f"[Epoch {epoch_idx}] finished | total_loss={final_loss:.4f}")
        return final_loss

    def _check_nan(self, loss):
        # Avoid a forced CPU/GPU sync on every batch.
        pass

    def _valid_epoch(self, valid_data, show_progress=False):
        valid_score, valid_result = super()._valid_epoch(valid_data, show_progress=False)
        if self._cfg("log_validation_metrics", True):
            print(f"[Validation] metrics: {valid_result}")
        return valid_score, valid_result
