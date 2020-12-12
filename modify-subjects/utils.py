import json
import sys

def load_dataset(split):
    data = {}
    if split == 'synthetic':
        with open('tasks/R2R-pano/data/R2R_literal_speaker_data_augmentation_paths.json') as f:
            data = json.load(f)
    else:
        with open('tasks/R2R-pano/data/R2R_%s.json' % split) as f:
            data = json.load(f)

    # Return de dictionary
    return data

def save_dataset(split, data, folder):
    if split == 'synthetic':
        with open(f'modify-subjects/{folder}/R2R_literal_speaker_data_augmentation_paths.json', "w") as f:
            f.write(json.dumps(data, indent=4))
    else:
        with open(f'modify-subjects/{folder}/R2R_%s.json' % split, "w") as f:
            f.write(json.dumps(data, indent=4))
    print(f"Correctly saved in {folder} split {split}")


def print_progress(iteration, total, prefix='', suffix='', decimals=1, bar_length=100):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        bar_length  - Optional  : character length of bar (Int)
    """
    str_format = "{0:." + str(decimals) + "f}"
    percents = str_format.format(100 * (iteration / float(total)))
    filled_length = int(round(bar_length * iteration / float(total)))
    bar = '█' * filled_length + '-' * (bar_length - filled_length)

    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix)),

    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()

