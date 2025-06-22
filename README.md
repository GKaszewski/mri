# How to run this?

Best way to run this is to use uv package manager. and then just run the command:

```bash
uv run main.py # for training model
uv run inference.py <model_path> <img_path> # for inference (tumor or not tumor for given image)
uv run grad.py <model_path> <img_path> # for gradient visualization
```

By the way, this code is not suited for production use, it is just a proof of concept. There are no standard practices applied, no logging, no error handling, no tests, no documentation, etc. Simplified deployment, no correct handling of cors, no security, etc.
