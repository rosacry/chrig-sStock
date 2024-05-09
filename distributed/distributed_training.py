import ray
from models.model_training import optimize_and_train, load_or_initialize_model, save_checkpoint, load_checkpoint, continuous_update
import os
import time

ray.init()  # Automatically uses resources like GPUs if available

@ray.remote
class ContinuousTrainer:
    def __init__(self, model_path):
        self.model_path = model_path
        # Load or initialize the model and optimizer based on the existence of a checkpoint
        if os.path.exists(model_path):
            self.model, self.optimizer = load_checkpoint(model_path)
        else:
            self.model, self.optimizer = load_or_initialize_model(model_path)

    def train(self, train_type='historical'):
        # Train the model continuously with the ability to specify the type of training
        if train_type == 'real_time':
            continuous_update(self.model_path)  # This will train using real-time data
        else:
            self.model, self.optimizer = optimize_and_train(self.model_path)
            save_checkpoint(self.model, self.optimizer, self.model_path)  # Save state frequently

    def update_model_path(self, new_model_path):
        self.model_path = new_model_path
        if os.path.exists(new_model_path):
            self.model, self.optimizer = load_checkpoint(new_model_path)
        else:
            self.model, self.optimizer = load_or_initialize_model(new_model_path)

def manage_training_sessions():
    trainer = ContinuousTrainer.remote('path/to/initial/model.pth')
    training_task = trainer.train.remote(train_type='historical')

    while True:
        # Check if training should stop
        if not training_should_continue():
            print("Halting distributed training as per the stop signal.")
            break
        ready, _ = ray.wait([training_task], num_returns=1, timeout=None)
        for result in ready:
            print(f"Training session updated or completed: {result}")
            # Restart or update the training task based on conditions
            training_task = trainer.train.remote(train_type='historical')
        time.sleep(10)  # Check every 10 seconds

def training_should_continue():
    with open("training_control.txt", "r") as file:
        status = file.read().strip()
    return status.lower() == 'start'
    
if __name__ == "__main__":
    manage_training_sessions()
