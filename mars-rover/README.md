# Mars Rover

Project layout for the rover terrain-analysis dashboard.

Place the source terrain image at `data/mars_terrain.jpg`. The files in
`outputs/` are generated artifacts and will be created when the processing
pipeline is implemented and run.

```text
mars-rover/
├── config.py
├── perception.py
├── planner.py
├── science.py
├── app.py
├── README.md
├── data/
│   └── mars_terrain.jpg
└── outputs/
    ├── terrain_classes.npy
    ├── confidence_map.npy
    ├── terrain_map.png
    └── hazard_map.png
```
