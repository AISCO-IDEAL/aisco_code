import importlib

from subprojects.wp4_aeroshell_optimizer.v1.src.models.model_baseclass import BaseModel
from subprojects.wp4_aeroshell_optimizer.v1.src.utils.config_utils import load_general_config
from subprojects.wp4_aeroshell_optimizer.v1.src.utils.dynamic_loading import dynamic_type_loading
# we want to dynamically run an entire optimization process for the WIRE model.
# we need the model, config and (losses)
# we need paths to store model (aka the sdf) in the end

# use the train config file


def model_train(config_name: str = "training_v0_config.yaml"):
    # to be replaced by generalized util function
    config = load_general_config(cfg_name=config_name)
    model_name = config["model"]
    model_file_name = config["model_filename"]
    module = importlib.import_module("subprojects.wp4_aeroshell_optimizer.v1.src.models." + model_file_name)
    ModelClass = getattr(module, model_name)  # noqa: N806
    print("type(ModelClass):", type(ModelClass))

    return ModelClass()

def check_pretrained(model: BaseModel) -> bool:

    # create hash for model

    # get path to pretrained

    # check if pretrained model exists
    hash_path =
    return False
    """
    When we
    Inputs:
        Config.yaml dictonary, including a model key

    Outputs:


    """
def initilise_model():

    # gehe in general config
    model_name = general_config["model"]
    model_definition = general_config["model_file_path"]

    # import model
    model = dynamic_type_loading(file_path= model_definition, class_name = model_name)

   # initilise model with corresponding config
   config =
   model = model(**config )

    # if check_pretrained
    if not check_pretrained(model):
        # pretrain the model
        # model = ... to be used further on
        raise("Not implemented yes")

def main() -> None:
    # start training
    initilise_model()


if __name__ == "__main__":
    main()
