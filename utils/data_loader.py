from glob import glob
import pandas as pd

def load_subtitles_dataset(dataset_path):
    subtitles_path = glob(dataset_path + '/*.ass')

    scripts = []
    episode_num = []

    for path in subtitles_path:

        # Read lines
        with open(path, 'r', encoding='utf8') as file:
            lines = file.readlines()
            lines = lines[27:]
            lines = [",".join(line.split(',')[9:]) for line in lines]
        
        # Clean data
        lines = [line.replace('\\N', ' ') for line in lines]
        script = " ".join(lines)

        episode = int(path.split('-')[-1].split('.')[0].strip())

        scripts.append(script)
        episode_num.append(episode)

    df = pd.DataFrame.from_dict({"episode": episode_num, "script": scripts})
    return df