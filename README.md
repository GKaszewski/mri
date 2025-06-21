# How to run this?

Best way to run this is to use uv package manager. and then just run the command:

```bash
uv run main.py # for training model
uv run inference.py <model_path> <img_path> # for inference (tumor or not tumor for given image)
uv run grad.py <model_path> <img_path> # for gradient visualization
```
