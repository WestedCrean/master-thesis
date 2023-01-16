import wandb

# define sweep parameters
base_sweep_config = {
    "method": "random",
    "metric": {"goal": "maximize", "name": "val_accuracy"},
    "parameters": {
        "batch_size": {"values": [32, 128, 256]},
        "epochs": {"value": 1},
        "learning_rate": {
            "values": [
                1e-2,
                1e-3,
                # 1e-4,
            ]
        },
        "optimizer": {
            "values": [
                "adam",
                # "sgd", "adagrad", "adadelta"
            ]
        },
    },
}


def launch_sweep(train_fn, number_of_runs=20, sweep_config=base_sweep_config):
    # launch sweep controller

    print("Launching sweep...")

    sweep_id = wandb.sweep(sweep_config, project="master-thesis")

    wandb.agent(sweep_id, train_fn, count=number_of_runs)
