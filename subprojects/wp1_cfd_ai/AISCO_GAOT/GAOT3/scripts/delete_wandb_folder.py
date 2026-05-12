import wandb

ENTITY = "aisco"
PROJECT = "GOAT_for_WP4"
ARTIFACT_TYPE = "model"
DRY_RUN = False
TIMEOUT = 3000
RETRIES = 3

api = wandb.Api(timeout=TIMEOUT)
collections = api.artifact_type(ARTIFACT_TYPE, project=f"{ENTITY}/{PROJECT}").collections(per_page=1000)

for collection in collections:
    path = f"{ENTITY}/{PROJECT}/artifacts/{ARTIFACT_TYPE}/{collection.name}"
    for artifact in collection.artifacts(per_page=100):
        print(("Would delete" if DRY_RUN else "Deleting"), path, artifact.name)
        if DRY_RUN:
            continue
        for attempt in range(RETRIES):
            try:
                artifact.delete(delete_aliases=True)
                break
            except Exception as exc:
                if attempt == RETRIES - 1:
                    print("Failed:", artifact.name, exc)
