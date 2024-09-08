import json
import yaml
import tarfile
import logging
from pathlib import Path
from ultralytics import YOLO

logger = logging.getLogger(__name__)


if __name__ == '__main__':
    model_path = '/opt/ml/processing/model/model.tar.gz'
    logging.info(f'Loading model from {model_path}...')

    with tarfile.open(model_path) as tar:
        tar.extractall(path='/opt/ml/processing/input/code')

    logger.info('Loading model...')
    model = YOLO(f'/opt/ml/processing/input/code/model.pt')

    # modify data conf
    with open('/opt/ml/processing/input/code/data.yaml', 'w') as fp:
        data_conf = {
            'train': '/opt/ml/processing/input/images',
            'val': '/opt/ml/processing/input/images',
            'test': '/opt/ml/processing/input/images',
            'names': {
                0: 'Hammer', 
          1: 'SSD',
          2: 'Alcohol',
          3: 'Spanner',
          4: 'Axe',
          5: 'Awl',
          6: 'Throwing Knife',
          7: 'Firecracker',
          8: 'Thinner',
          9: 'Plier',
          10: 'Match',
          11: 'Smart Phone',
          12: 'Scissors',
          13: 'Tablet PC',
          14: 'Solid Fuel',
          15: 'Bat',
          16: 'Portable Gas',
          17: 'Nail Clippers',
          18: 'Knife',
          19: 'Metal Pipe',
          20: 'Electronic Cigarettes(Liquid)',
          21: 'Supplymentary Battery',
          22: 'Bullet',
          23: 'Gun Parts',
          24: 'USB',
          25: 'Liquid',
          26: 'Aerosol',
          27: 'Screwdriver',
          28: 'Chisel',
          29: 'Handcuffs',
          30: 'Lighter',
          31: 'HDD',
          32: 'Electronic Cigarettes',
          33: 'Battery',
          34: 'Gun',
          35: 'Laptop',
          36: 'Saw',
          37: 'Zippo Oil',
          38: 'Stun Gun',
          39: 'Camera',
          40: 'Camcorder',
          41: 'SD Card',
            }
        }
        yaml.dump(data_conf, fp)
        logger.info(f'Updated data conf: {json.dumps(data_conf, indent=2)}')

    metrics = model.val('/opt/ml/processing/input/code/data.yaml')
    precision, recall, mAP50, mAP95 = metrics.mean_results()
    fitness = metrics.fitness
    logger.info(f'precision: {precision}')
    logger.info(f'recall: {recall}')
    logger.info(f'mAP50: {mAP50}')
    logger.info(f'mAP95: {mAP95}')
    logger.info(f'fitness: {fitness}')

    report_dict = {
        'multiclass_classification_metrics': {
            'weighted_precision': {'value': precision, 'standard_deviation': 'NaN'},
            'weighted_recall': {'value': recall, 'standard_deviation': 'NaN'},
            'mAP50': {'value': mAP50, 'standard_deviation': 'NaN'},
            'mAP95': {'value': mAP95, 'standard_deviation': 'NaN'},
            'fitness': {'value': fitness, 'standard_deviation': 'NaN'},
        },
    }

    output_dir = '/opt/ml/processing/evaluation'
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    evaluation_path = f'{output_dir}/evaluation.json'
    with open(evaluation_path, 'w') as fp:
        fp.write(json.dumps(report_dict))

    logger.info('Evaluation complete!')