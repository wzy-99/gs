
class LogMaxin:
    def log_loss(self, losses: dict, prefix: str = "") -> None:
        for k, v in losses.items():
            self.log(f"{prefix}/{k}", v, on_step=True, on_epoch=True, prog_bar=True)