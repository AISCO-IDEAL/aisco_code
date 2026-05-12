import wandb

RUN_PATH = "aisco/GOAT_for_WP4/YOUR_RUN_ID"
FOLDER = "media/images"

run = wandb.Api().run(RUN_PATH)
folder = FOLDER.replace("\\", "/").strip("/")

for file in run.files(per_page=1000):
    name = file.name.replace("\\", "/").strip("/")
    if name == folder or name.startswith(folder + "/"):
        print("Deleting", file.name)
        file.delete()
