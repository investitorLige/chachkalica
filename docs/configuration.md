# Configuration

## Instance Config

`configs/instance/local.yaml` controls the local Label Studio instance.

Local Docker mode:

```yaml
label_studio:
  mode: local
  container_name: label-studio
  image_name: heartexlabs/label-studio:latest
  url: http://localhost:8080
  port: 8080
  volume_name: label-studio-data
  data_root: ./label_data
```

Remote/Kubernetes mode:

```yaml
label_studio:
  mode: remote
  url: https://label-studio.example.com
  data_root: ./label_data
```

`mode: remote` skips Docker start/inspect and only checks that the URL responds.

## Project Config

Project configs point to an instance config and define dataset paths, storage mode, and API auth.

```yaml
instance_config: configs/instance/local.yaml

project:
  title: Mock PPE Dataset

paths:
  dataset_dir: mock_dataset/images
  classes_file: mock_dataset/classes.txt

storage:
  type: local

auth:
  token: YOUR_LABEL_STUDIO_USER_API_TOKEN
```

`dataset_dir` and `classes_file` are resolved relative to `label_studio.data_root` unless absolute.

With:

```yaml
label_studio:
  data_root: ./label_data
```

these resolve to:

```text
label_data/mock_dataset/images
label_data/mock_dataset/classes.txt
```

## Storage Modes

## Labeling Schema

The label config is generated from `classes.txt` (one `<Label>` per non-empty
line) and includes bbox, polygon, and keypoint controls over the image:

```xml
<View>
  <Image name="image" value="$image"/>
  <RectangleLabels name="bbox" toName="image">...</RectangleLabels>
  <PolygonLabels name="segmentation" toName="image">...</PolygonLabels>
  <KeyPointLabels name="sam_point" toName="image" smart="true">...</KeyPointLabels>
</View>
```

The `sam_point` keypoint control is wired for the interactive SAM ML backend
(`ml_backends/sam`); connect that backend under *Project Settings → Model* to
use it. SAM is configured to return a **polygon** on the `segmentation` control
(`SAM_OUTPUT=polygon`). Brush is not offered: the export pipeline (`coco_sync/`,
`fleet.py sync`) only handles bbox + polygon, with no mask format.

## Storage Modes

Local storage:

```yaml
storage:
  type: local
```

The script creates a Label Studio local-files storage pointing at:

```text
/label-studio/data/local/<dataset_dir>
```

Cloud storage:

```yaml
storage:
  type: cloud
  root: https://my-bucket.example.com
```

The script still enumerates local files from:

```text
data_root + dataset_dir
```

but imports Label Studio task image URLs like:

```text
https://my-bucket.example.com/<dataset_dir>/<filename>
```

## Auth Token

`auth.token` is a Label Studio API token for a user on that Label Studio instance.

The script sends:

```http
Authorization: Token <token>
```

The token's user must have permission to create projects and import tasks.
