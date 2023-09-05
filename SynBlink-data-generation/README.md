# SynBlink data generation



## Preparation
- 3D Model Preparation: 
  - BuyRealistic Human 100 Pack from Reallusion Inc., which includes 100 3D heads. 
  - Use two base body models which are fully rigged in Character Creator (one male and one female).
  - Replace each 3D head with the base model's original head in Character Creator separately. 
  - A total of 100 fully-rigged models are exported and saved as Fbx files.
- Background Preparation:
  - COCO-train2017
  - Describable Textures Dataset (DTD)
  - HDRI images from Poly Haven

## Requirnements
```
Blender 3.3.1
python==3.9.12
bpy==3.6.0
numpy==1.22.4
```

## Run

```
python run.py
```
