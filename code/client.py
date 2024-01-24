from utils import *
from FEMNIST import *
from HAR import *
from sklearn.metrics import classification_report
import flwr as fl
from filelock import FileLock

# there are three possible training types: FL (Federated) and LOCAL, else it trains the classifier
train_type="FL"

seed = 42
random.seed(seed)
np.random.seed(seed)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
     
    
# FEMNIST PART (comment to disable)
"""  
path = r"Desktop\FEMNIST Dataset\all_data_0.json"
person = sys.argv[1]
angle = int(sys.argv[2])
image_shape = (28,28)
print("User: {0}, angle: {1}".format(person, angle))
classes = 10
test_size = 0.40
femnist = FEMNIST(person, angle, classes, path)
dataset = femnist.get_dataset(test_size=test_size, shots=0)
#CI_VAE = femnist.create_CH_model(k_size=32)
CI_VAE = femnist.create_model(latent_dim=7)
print("N. Params:", CI_VAE.count_params())
"""
FEMNIST_selection = False


# HAR PART (comment to disable) 
person = int(sys.argv[1])
classes = 6
har = HAR(r"HAR_dataset.csv")
dataset = har.get_dataset(person)
print("User: {0}".format(person))
# MODEL SELECTION (comment to disable)
CI_VAE = har.create_model(latent_dim=3, dense_size=64)
#CI_VAE = har.create_model_CH(dense_size=128)
image_shape = (12,12)


# try to load the model if exists
model_path = r"model_weights\model.h5"
try: 
    CI_VAE.load_weights(model_path)
    print("Model checkpoint loaded")
except Exception as ex: print("No model checkpoint:", ex)




class KerasClient(fl.client.Client):
    def __init__(self, dataset_type, dataset, model, train_type="FL"):
        super().__init__()
        self.dataset_type = dataset_type
        self.dataset = dataset
        self.train_type = train_type
        self.x_train, self.y_train, self.x_test, self.y_test = self.dataset
        self.x_train, self.x_test = har.preprocessing(self.x_train, self.x_test) # COMMENT IF YOU USE FEMNIST
        self.model = model
        
    def get_parameters(self, config):
        return self.model.get_weights()
    
    def set_parameters(self, parameters):
        self.model.set_weights(parameters)
        
    def fit(self, parameters, config):
        self.set_parameters(parameters)
        
        if self.train_type == "FL": self.FL_train()
        elif self.train_type == "LOCAL": self.LOCAL_train()
        else: self.CH_train()

        # return the updated model weights
        return self.model.get_weights(), len(self.x_train), {} 
    
    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        
        if self.train_type == "FL": evaluation = self.FL_evaluate() 
        elif self.train_type == "LOCAL": evaluation = self.LOCAL_evaluate() 
        elif self.train_type == "CH": evaluation = self.CH_evaluate() 
        else: evaluation = self.REPTILE_evaluate()
        
        train_loss, test_loss, f1 = evaluation
        train_loss = float(train_loss.numpy())
        test_loss = float(test_loss.numpy())

        print("Train loss: {0}, Test loss: {1}, f1 test: {2}".format(train_loss, test_loss, f1)) 
        
        # save the model (attenzione alla concomitanza dei modelli che provano ad accadere al file contemporaneamente)
        lock = FileLock('model_weights\model.lock')
        with lock:
            self.model.save('model_weights\model.h5')
        return float(train_loss), len(self.x_train), {"train_loss": train_loss, "val_loss":test_loss}
    
    def compute_loss(self, x, y, training=True):
        z_mean, z_log_var, z = self.model.get_layer('encoder')(x, training=training)
        reconstruction = self.model.get_layer('decoder')(z, training=training)
        mae_loss = tf.reduce_mean(keras.losses.mean_squared_error(x, reconstruction))
        mae_loss *= image_shape[0] * image_shape[1]
        kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
        kl_loss = tf.reduce_mean(kl_loss)
        kl_loss *= -0.5
        class_loss = self.model.get_layer('class_model')(z, training=training)
        class_loss = tf.reduce_mean(keras.losses.categorical_crossentropy(y, class_loss))
        return mae_loss, kl_loss, class_loss

    def compute_loss_CH(self, x, y, training=True):
        class_loss = self.model(x, training=training)
        class_loss = tf.reduce_mean(keras.losses.categorical_crossentropy(y, class_loss))
        return class_loss

    def LOCAL_train(self):
        # load the VAE weights
        encoder_path = "model_weights\{0}_encoder_weights.h5".format(person)
        decoder_path = "model_weights\{0}_decoder_weights.h5".format(person)
        if os.path.exists(encoder_path): self.model.get_layer("encoder").load_weights(encoder_path)
        if os.path.exists(decoder_path): self.model.get_layer("decoder").load_weights(decoder_path)
        self.FL_train()
        # save the VAE weights
        self.model.get_layer('encoder').save_weights(encoder_path)
        self.model.get_layer('decoder').save_weights(decoder_path)

    def FL_train(self):
        learning_rate = 0.001
        class_loss_weight = 3
        batch = 4
        epoch = 1
        optimizer = Adam(learning_rate=learning_rate)
        self.model.compile(optimizer=optimizer)
        for _ in range(epoch):
            for batch_data in range(len(self.y_train)//batch+1):
                y = self.y_train[batch_data*batch:batch_data*batch+batch]
                x = self.x_train[batch_data*batch:batch_data*batch+batch]
                if x.shape[0] == 0: continue
                with tf.GradientTape() as tape:
                    mae_loss, kl_loss, class_loss = self.compute_loss(x, y)
                    total_loss = mae_loss + kl_loss + class_loss_weight * class_loss
                grads = tape.gradient(total_loss, self.model.trainable_weights)
                optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        return

    def CH_train(self):
        learning_rate = 0.001
        batch = 4 if FEMNIST_selection else 8
        optimizer = Adam(learning_rate=learning_rate)
        self.model.compile(optimizer=optimizer)
        for batch_data in range(len(self.y_train)//batch):
            y = self.y_train[batch_data*batch:batch_data*batch+batch]
            x = self.x_train[batch_data*batch:batch_data*batch+batch]
            if x.shape[0] == 0: continue
            with tf.GradientTape() as tape:
                total_loss = self.compute_loss_CH(x, y)
            grads = tape.gradient(total_loss, self.model.trainable_weights)
            optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        return       

    def LOCAL_evaluate(self):
        # load the VAE weights
        encoder_path = "model_weights\{0}_encoder_weights.h5".format(person)
        decoder_path = "model_weights\{0}_decoder_weights.h5".format(person)
        if os.path.exists(encoder_path): self.model.get_layer("encoder").load_weights(encoder_path)
        if os.path.exists(decoder_path): self.model.get_layer("decoder").load_weights(decoder_path)
        return self.FL_evaluate()

    def FL_evaluate(self):
        batch = 4
        learning_rate = 0.0005
        optimizer = Adam(learning_rate=learning_rate)
        self.model.compile(optimizer=optimizer)
        
        train_loss = []
        cont = 0
        for batch_data in range(len(self.y_train)//batch+1):
            y = self.y_train[batch_data*batch:batch_data*batch+batch]
            x = self.x_train[batch_data*batch:batch_data*batch+batch]
            if x.shape[0] == 0: continue
            train_loss.append(sum(self.compute_loss(x, y)))
            cont += 1
        train_loss = sum(train_loss)/cont
        test_loss = []
        cont = 0
        for batch_data in range(len(self.y_test)//batch+1):
            y = self.y_test[batch_data*batch:batch_data*batch+batch]
            x = self.x_test[batch_data*batch:batch_data*batch+batch]
            if x.shape[0] == 0: continue
            test_loss.append(sum(self.compute_loss(x, y, training=False)))
            cont += 1
        test_loss = sum(test_loss)/cont
        predicts = self.model.predict(self.x_test, batch_size=batch, verbose=0)
        f1 = f1_score(self.y_test.argmax(1), predicts[1].argmax(1), average='micro')
        return train_loss, test_loss, round(f1,2)

    def CH_evaluate(self):
        batch =local
        learning_rate = 0.0005
        optimizer = Adam(learning_rate=learning_rate)
        self.model.compile(optimizer=optimizer)
        
        train_loss = []
        cont = 0
        for batch_data in range(len(self.y_train)//batch+1):
            y = self.y_train[batch_data*batch:batch_data*batch+batch]
            x = self.x_train[batch_data*batch:batch_data*batch+batch]
            if x.shape[0] == 0: continue
            train_loss.append(self.compute_loss_CH(x, y))
            cont += 1
        train_loss = sum(train_loss)/cont
        
        test_loss = []
        cont = 0
        for batch_data in range(len(self.y_test)//batch+1):
            y = self.y_test[batch_data*batch:batch_data*batch+batch]
            x = self.x_test[batch_data*batch:batch_data*batch+batch]
            if x.shape[0] == 0: continue
            test_loss.append(self.compute_loss_CH(x, y, training=False))
            cont += 1
        test_loss = sum(test_loss)/cont
        predicts = self.model.predict(self.x_test, batch_size=batch, verbose=0)
        f1 = f1_score(self.y_test.argmax(1), predicts.argmax(1), average='micro')
        return train_loss, test_loss, round(f1,2)
    
    
fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=KerasClient(har, dataset, CI_VAE, train_type=train_type))