import json
import base64
from pathlib import Path

def dict_from_json(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    return data

def write_markdown(cell, f):
    print(file=f)
    for line in cell['source']:
        print(line, file=f)

def write_source(cell, f):
    print("```python", file=f)
    for line in cell['source']:
        print(line, file=f, end='')
    print("\n```", file=f)

def get_img_filename(img_path):
    create_dir(img_path)
    return len(list(img_path.iterdir()))

def create_dir(path):
    if not path.exists():
        path.mkdir()

def create_file(path):
    if not path.exists():
        path.touch()

def write_image_to_disk(img_data, p, img_dir='images/'):
    img_path = Path(f"{p}/{img_dir}{get_img_filename(p/img_dir)}.png")
    create_file(img_path) 
    with open(img_path, 'wb') as img_file:
        img_file.write(base64.decodebytes(img_data.encode('ascii')))
    return img_path

def write_image(img_data, p, img_path='images/'):
    img_path = write_image_to_disk(img_data, p, img_dir='images/')
    img_str = f'![{img_path.name}]({str(img_path.resolve())})'
    return img_str

def write_out(cell, f, p):
    for output in cell['outputs']:
        if output['output_type'] == 'stream':
            print('\n`output:`',file=f)
            print('```python', file=f)
            print("\"\"\"", file=f)
            for line in output['text']:
                print(line, file=f)
            print("\"\"\"", file=f)
            print('```', file=f)
        elif output['output_type'] == 'display_data':
            img_str = write_image(output['data']['image/png'], p)
            print(img_str, file=f)

def write_code(cell, f, p):
    write_source(cell, f)
    write_out(cell, f, p)

def md_from_dict(json_dict, path, filename='README.md'):
    p = Path(path)
    with open(f"{str(p)}/{filename}", 'w') as f:
        write_metadata(f)
        for cell in json_dict['cells']:
            if cell['cell_type'] == 'markdown':
                write_markdown(cell, f)
            elif cell['cell_type'] == 'code':
                write_code(cell, f, p)

def write_metadata(f):
    print("---", file=f)
    print("geometry: margin=2cm", file=f)
    print("---", file=f)


if __name__ == '__main__':
    json_path = 'tps/tp3/tp3.ipynb'
    data = dict_from_json(json_path)
    md_path = 'tps/tp3/'
    md_from_dict(data, md_path)
