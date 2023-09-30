import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import wandb
from datamodel import FloatImageDataset, train_test_split
from torch.utils.data import DataLoader
from train_test_suite import train_and_test_model
from sklearn.metrics import classification_report
import yaml
from pprint import pprint
from pathlib import Path

HOME_DIRECTORY = Path.home()
SEED = 42

with open("resnet_sweep.yml", "r") as yaml_file:
    sweep_config = yaml.safe_load(yaml_file)

sweep_id = wandb.sweep(sweep=sweep_config)


class CustomResNetClassifier(nn.Module):
    """
    Resnet18 tweaked classifer

    """

    def __init__(self,
                 tail_train_percentage=0.25,
                 number_of_classes=2):
        super(CustomResNetClassifier, self).__init__()

        self.resnet18 = models.resnet18(pretrained=True)

        # now freeze the last tail_train_percentage of layers
        depth = len(list(self.resnet18.parameters()))
        transition_level = depth * (1 - tail_train_percentage)
        for i, param in enumerate(self.resnet18.parameters()):
            if i <= transition_level:
                param.requires_grad = False

        # remove the last layer - we don't want it to classify, that's for us to do
        self.resnet18 = nn.Sequential(*list(self.resnet18.children())[:-1])

        self.classifier = nn.Linear(512, number_of_classes)

    def forward(self, x):
        # Extract features using the ResNet-18 backbone
        features = self.resnet18(x)

        # Flatten the features if needed (e.g., if you have spatial dimensions)
        features = features.view(features.size(0), -1)

        # Apply the fully connected layer to get class logits
        logits = self.classifier(features)

        return logits





def find_best_model():
    # config for wandb

    # Initialize wandb
    wandb.init(project="ResNet18")
    config = wandb.config

    # creating the model stuff
    learning_rate = config.learning_rate
    epochs = wandb.config.epochs

    print('HYPER PARAMETERS:')
    pprint(config)

    # create a custom resnet, retraining the last percentage of the layers
    model = CustomResNetClassifier(tail_train_percentage=config.tail_train_percentage)

    print('Model Architecture:')
    print(model)

    path = f"{HOME_DIRECTORY}/data/0.35_reduced_then_balanced/data_{config.input_size}"

    dataset = FloatImageDataset(directory_path=path,
                                true_folder_name="entangled", false_folder_name="not_entangled"
                                )

    training_dataset, testing_dataset = train_test_split(dataset, train_size=0.75, random_state=SEED)
    batch_size = config.batch_size

    # create the dataloaders
    train_dataloader = DataLoader(training_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(testing_dataset, batch_size=batch_size)

    # Define loss function
    loss_fn = nn.CrossEntropyLoss()

    # optimzer parsing logic:
    if config.optimizer == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    else:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    history = train_and_test_model(train_dataloader=train_dataloader, test_dataloader=test_dataloader,
                                   model=model, loss_fn=loss_fn, optimizer=optimizer, epochs=epochs,
                                   device="cpu", wandb=wandb, verbose=False)

    y_true, y_pred = history['y_true'], history['y_pred']
    print(classification_report(y_true=y_true, y_pred=y_pred))

    # Log test accuracy to wandb

    # Log hyperparameters to wandb
    wandb.log(dict(config))


if __name__ == "__main__":
    wandb.agent(sweep_id, function=find_best_model)

    # Specify your W&B project and sweep ID
    project_name = "ResNet18"

    # Fetch sweep runs
    api = wandb.Api()
    sweep = api.sweep(f"{project_name}/{sweep_id}")
    runs = list(sweep.runs)

    # Find the best run based on the metric you care about (e.g., lowest validation loss)
    best_run = None
    best_metric_value = float("inf")

    for run in runs:
        if run.summary["accuracy"] > best_metric_value:
            best_run = run
            best_metric_value = run.summary["accuracy"]

    # Print the best run and its hyperparameters
    print("Best Run:")
    print(f"Run ID: {best_run.id}")
    print(f"Test Accuracy: {best_run.summary['accuracy']}")
    print(f"Hyperparameters: {best_run.config}")
