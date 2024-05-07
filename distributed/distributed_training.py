import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from models.model_training import initialize_or_update_model
from models.model_tuning import tune_and_save_model
from models.optuna_optimization import optimize_and_save_study

def initialize_tune_optimize(num_epochs=10, incremental=False, model_path='models/model/aiModel.pth', optuna_trials=50):
    """Initialize or update the model, tune it, and then run Optuna optimization."""
    # Step 1: Initialize or update the model
    initialize_or_update_model(model_path, num_epochs, incremental)

    # Step 2: Tune the model with grid search
    tune_and_save_model(model_path)

    # Step 3: Optimize with Optuna
    optimize_and_save_study(n_trials=optuna_trials)


def distributed_train_with_ray():
    """Distributed training setup and launch with Ray."""
    ray.init(ignore_reinit_error=True)

    # ASHAScheduler for efficient resource management
    scheduler = ASHAScheduler(
        metric="val_loss",
        mode="min",
        max_t=10,
        grace_period=1,
        reduction_factor=2
    )

    # Ray Tune hyperparameter search
    analysis = tune.run(
        initialize_tune_optimize,
        resources_per_trial={"cpu": 2, "gpu": 1},
        config={
            "num_epochs": tune.grid_search([10, 20, 30]),
            "incremental": tune.choice([True, False]),
            "model_path": "/mnt/data/best_tuned_model.pth",
            "optuna_trials": tune.grid_search([30, 50, 70])
        },
        scheduler=scheduler,
        num_samples=10
    )

    best_config = analysis.best_config
    print(f"Best configuration found: {best_config}")
    return best_config
