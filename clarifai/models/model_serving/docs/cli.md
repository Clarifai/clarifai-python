# CLI helpers

Supported cli:

```bash
$ clarifai -h
upload              Upload component to Clarifai platform
create              Create component of Clarifai platform
login               Login to Clarifai and save PAT locally
example             Download/List examples of model upload
build               Build clarifai model for uploading
```

1. Login

```bash
$ clarifai login
Get your PAT from https://clarifai.com/settings/security and pass it here: <your pat>
```

2. Create model repository

Initialize template for specify model type in provided directory

* `From scratch`:

```bash
$ clarifai create model --type <model-type> --working-dir <your_working_dir>
```

* `From example`:

```bash
$ clarifai create model --from-example --working-dir <your_working_dir>
? Select an example:
❯ multimodal_embedder/clip
  text_classifier/xlm-roberta
  text_embedder/instructor-xl
  ...
```

Then will see below output

```bash
---------------------------------------------------------------------------
* Created repository at: ./<your_working_dir>
<your_working_dir>
├── clarifai_config.yaml
├── inference.py
├── requirements.txt
└── test.py

0 directories, 4 files

* Please make sure your code is tested using `test.py` before uploading
---------------------------------------------------------------------------
```

> NOTE: if working-dir exists, need to set --overwrite flag otherwise an error arises

Full arguments

```bash
$ clarifai create model -h
--working-dir         Path to your working dir. Create new dir if it does not exist
--from-example        Create repository from example
--example-id          Example id, run `clarifai example list` to list of examples
--type                Clarifai supported model types.
--image-shape         list of H W dims for models with an image input type. H and W each have a max value of 1024
--max-bs              Max batch size
--overwrite           Overwrite working-dir if exists
```

3. See available examples

```bash
$ clarifai example list
Found 11 examples
 * multimodal_embedder/clip
 * text_classifier/xlm-roberta
 * text_embedder/instructor-xl
  ....
```

4. Build

This step will run `test.py` in provided working dir as default before building

```
$ clarifai build model <your_working_dir> --name model-name
$ tree <your_working_dir> -a
<your_working_dir>
├── .cache # (*)
│   ├── 1
│   │   ├── clarifai_config.yaml
│   │   ├── inference.py
│   │   ├── model.py
│   │   ├── test.py
│   │   └── ...
│   ├── config.pbtxt
│   └── requirements.txt
├── clarifai_config.yaml
├── inference.py
├── model-name.clarifai # (**)
├── requirements.txt
├── test.py
└── ...
```

**NOTE:**

(*): Build cache, user can simply ignore it.

(**): zipped of .cache

Full arguments

```bash
$ clarifai build model -h
positional arguments:
  path               Path to working directory, default is current directory
optional arguments:
--out-path           Output path of built model
--name               Name of built file, default is `clarifai_model_id` in config if set or`model`
--no-test            Trigger this flag to skip testing before uploading
```

5. Upload

This step will execute test.py in the specified working directory by default before proceeding with the build. You can upload your built file directly from the working directory to the platform or upload it to cloud storage and provide the direct URL during the upload process.

Use the following command to upload your built file directly to the platform. It will upload the `*.clarifai` file. *Note*: Only support file size from 5MiB to 5GiB

```bash
$ clarifai upload model <your_working_dir>
```

or upload with direct download url

```bash
$ clarifai upload model <your_working_dir> --url <your url>
```

Full arguments

```bash
$ clarifai upload model -h
positional arguments:
  path                  Path to working dir to get clarifai_config.yaml or path to yaml. Default is current directory

optional arguments:
  -h, --help           show this help message and exit
  --url URL            Direct download url of zip file
  --file FILE          Local built file
  --id ID              Model ID
  --user-app USER_APP  User ID and App ID separated by '/', e.g., <user_id>/<app_id>
  --no-test            Trigger this flag to skip testing before uploading
  --no-resume          Trigger this flag to not resume uploading local file
  --update-version     Update exist model with new version

```
