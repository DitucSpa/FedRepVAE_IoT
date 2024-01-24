# FedRepVAE_IoT
Federated (Reptile) Class Informed VAE for IoT.

## How it works
To simulate a Federated Learning environment, each client is emulated using its own terminal. Each terminal will receive the client files to use and the respective name, which will later allow for loading the dataset and model. Finally, the simulation will be executed by training the respective models. The computational cost with this method is very high because each terminal/client has its own model and dataset.

## Future steps (theory)
- Differential Privacy;
- Quantization on VAE;
- Transfer Learning with MNIST;
- GAN and DoppelGANger (for generating float32);
- Knowledge Distillation (Server as teacher and client as student).


## Future steps (for code)
- Merge the two different server strategies (FedRep and FedAVG) into a single file;
- Rewrite the part related to the Client to ensure that the choice of the dataset appears as an argument to be passed to the script;
- Reduce the complexity of the simulation (PyTorch?);
