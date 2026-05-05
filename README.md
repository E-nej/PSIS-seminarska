## Running the Project

All code lives in `Deep-Qlearning-TSC/`. The main entry point is:

```bash
cd Deep-Qlearning-TSC
python tl_main.py
```

**Key toggles at the top of `tl_main.py`:**
- `training_enabled = True` — set to `False` to run evaluation only
- `gui = False` — set to `True` to open the SUMO GUI (requires `sumo-gui`)
- `total_episodes`, `max_steps`, `num_experiments` — training hyperparameters

**SUMO dependency:** SUMO must be installed and the TraCI tools path must be set. The file currently has a hardcoded Windows path:
```python
sys.path.append(os.path.join('c:', os.sep, 'Users','Desktop','Work','Sumo','tools'))
```
Update this to your local SUMO installation (e.g., `/usr/share/sumo/tools` on Linux).

The `sumoBinary` variable must also point to the correct SUMO executable (`sumo` on Linux, `sumo.exe` on Windows).
