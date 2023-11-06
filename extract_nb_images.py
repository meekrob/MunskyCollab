#!/usr/bin/env python3
import base64
import json
import os, sys
from typing import Dict, List, Optional

def debug(func):
    def wrapper(*args, **kwargs):
        # print the fucntion name and arguments
        kwkeys = list(kwargs.keys())
        str_kwkeys = ",".join(kwkeys)
        if len(kwkeys) > 100: kwkeys = kwkeys[:100]
        print(f"Calling {func.__name__} with {len(args)=} args:")
        for arg in args:
            str_arg = str(arg)
            if len(str_arg) > 100:
                str_arg = str_arg[:100] + "..."
            print("\t", str_arg)

        print(f"and {len(kwkeys)=} kwargs: {str_kwkeys}")
        # call the function
        result = func(*args, **kwargs)
        # print the results
        str_result = str(result)
        if len(str_result) > 100:
            str_result = str_result[:100] + "..."
        print(f"{func.__name__} returned: {str_result}")
        return result
    return wrapper

def main():
    export_images( sys.argv[1] )
    #export_images("viewResults.ipynb")
    


@debug
def get_images(notebook_dict: Dict) -> List[Dict]:

    return [
        {
            "cell_idx": cell_idx,
            "output_idx": output_idx,
            "content_type": content_type,
            "img_data": decode_img_data(content),
        }
        for cell_idx, cell in enumerate(notebook_dict.get("cells", ()))
        for output_idx, output in enumerate(cell.get("outputs", ()))
        for content_type, content in output.get("data", {}).items()
        if content_type.startswith("image/")
    ]

@debug
def decode_img_data(content):

    if isinstance(content, list):
        return "".join(content).encode("utf-8")
    else:
        return base64.b64decode(content)

@debug
def write_contents(contents: Dict):

    for filepath, content in contents.items():

        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        with open(filepath, "wb") as fw:
            fw.write(content)

@debug
def get_export_contents(images: List[Dict], prefix: str, output_dir: str) -> Dict:

    contents = {}

    for image_dict in images:

        file_ext = image_dict["content_type"].split("/", 1)[1].split("+", 1)[0]
        filename = "{}_cell_{}_output_{}.{}".format(
            prefix, image_dict["cell_idx"], image_dict["output_idx"], file_ext
        )
        filepath = output_dir + os.sep + filename

        contents.update({filepath: image_dict["img_data"]})

    return contents

@debug
def export_images(
    filepath: str, 
    output_dir: Optional[str] = None, 
    prefix: Optional[str] = None) -> Dict:

    with open(filepath, "r") as fr:
        notebook_dict = json.load(fr)

    if prefix is None:
        prefix = os.path.basename(filepath).split(".")[0]

    if output_dir is None:
        output_dir = "."

    images = get_images(notebook_dict=notebook_dict)
    export_contents = get_export_contents( images=images, prefix=prefix, output_dir=output_dir)

    write_contents(contents=export_contents)

    return export_contents

if __name__ == "__main__": main()
