import os
import wandb


def wandb_login_ensure_personal_account():
    os.system("wandb login")
    login_details = wandb.api.viewer()
    assert login_details["entity"] == os.environ.get(
        "WANDB_PERSONAL_USERNAME"
    ), "Wrong wandb account!"
