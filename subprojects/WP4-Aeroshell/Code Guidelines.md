#aalthaus 
This markdown file should serve as a code guideline we would like to enforce in the WP4 submodule (aeroshell generation)


## Example code on how to use typing and document codes

```python
import typing as T
  
#Typing does the following: flags what types are in/outputted
def train_sdf_model(model: SDFNetwork, config: T.Dict) -> SDFNetwork:
	


	#At the beginning we put the PRE and POST Conditions for documentation purposes
    """
    Train the given SDFNetwork model based on the provided configuration.

    Pre-conditions:
        - model: An instance of SDFNetwork, already initialized and ready for training.

        - config: A dictionary containing at least the following keys:
            - learning_rate: Positive float (default: 1e-3).
            - num_epochs: Positive integer (default: 1000).
            - batch_size: Positive integer (default: 1024).
            - (Optional) Any additional parameters required by SDFLosses or the training process.
              
    Post-conditions:
        - The model's parameters are updated via gradient descent for num_epochs iterations.
        - The loss is printed every 100 epochs.
        - No side effects on the input config or model other than parameter updates.
        - The trained model is returned.

    """

	#Functional Code goes here, i.e. thte training loop
	
	
	
	#Return type is also satisfied
	return model
```


