"""Run the logistic-regression baseline without training all five models."""

from models.train_baselines import main


if __name__ == "__main__":
    main(default_model="logreg")
