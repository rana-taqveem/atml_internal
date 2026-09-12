task1/
│
├── config.py                 <-- Replaced directory with a single file
├── make_subset.py
├── make_cue_conflicts.py
├── transforms.py
│
├── data/                     <-- Stores raw STL-10 and generated inputs
├── models/
│   └── backbones.py          <-- ResNet, ViT, CLIP wrappers + Linear Heads
├── analysis/
│   ├── evaluate_bias.py      <-- Clean, Color, and Cue-Conflict evaluation loops
│   ├── feature_similarity.py
│   └── representation.py
│
├── scripts/
│   └── run_task1.py          <-- Master orchestration script
├── results/                  <-- Logs, Accuracy Tables, and Visualizations
└── README.md