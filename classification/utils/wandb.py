import os
import wandb


def wandb_login_ensure_personal_account() -> None:
    os.system("wandb login")
    login_details = wandb.api.viewer()
    assert login_details["entity"] == "szafrixxx", "Wrong wandb account!"
