import wandb
wandb.login()

from ultralytics import YOLO
import ultralytics
from wandb.integration.ultralytics import add_wandb_callback

wandb.init(project='sample_project', job_type='training')

sweep_config = {
    'method': 'random',
    'metric': {'goal': 'maximize', 'name': 'metrics/mAP50(B)'},
    'parameters': {
        'batch_size': {
            'values': [16, 32, 64]
        },
        'learning_rate': {
            'distribution': 'uniform',
            'min':0.0001,
            'max':0.001
        }
        
    }
}

sweep_id = wandb.sweep(sweep=sweep_config, project='sds')

def yolo_train():
    with wandb.init() as run:
        config = wandb.config

        # 모델 정의 및 하이퍼파라미터 사용
        model = YOLO('yolov8n.pt')
        add_wandb_callback(model, enable_model_checkpointing=True)
        results = model.train(data='/data/sds/datasets/version_6_3756/data.yaml', name='version_6_3756_w',
                              plots=True, batch=config.batch_size, epochs=10, imgsz=(640,480),
                              cos_lr=True, lrf=0.01, lr0=config.learning_rate)
        wandb.log({
            'epoch': epoch, 
            'val_loss': val/dfl_loss
        })

wandb.agent(sweep_id, function=yolo_train)