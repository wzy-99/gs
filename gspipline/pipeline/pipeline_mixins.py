
class LogMixin:
    def log_loss(self, losses: dict, prefix: str | None = None) -> None:
        for k, v in losses.items():
            if prefix is not None:
                k = f"{prefix}_{k}"
            self.log(k, v, on_step=True, on_epoch=True, prog_bar=True)