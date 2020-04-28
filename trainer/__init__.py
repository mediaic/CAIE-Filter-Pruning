from trainer.normal import NormalTrainer

def get_trainer(cfg):
    pair = {
        'normal': NormalTrainer
    }
    assert (cfg.train.trainer in pair)

    return pair[cfg.train.trainer](cfg)
