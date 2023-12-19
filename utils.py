# Early Stopping
class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve for a given patience."""
    def __init__(self, patience=3):
        self.patience = patience
        self.best_loss = None
        self.counter = 0

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
            return False
        elif val_loss >= self.best_loss:
            self.counter += 1
            if self.counter >= self.patience:
                return True
            else:
                return False
        else:
            self.best_loss = val_loss
            self.counter = 0
            return False
